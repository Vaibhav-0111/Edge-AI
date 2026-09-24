"""
ONNX Runtime inference engine for YOLOv8 PPE detection.

Wraps onnxruntime.InferenceSession with:
  - Letterbox preprocessing (keep aspect ratio, pad to 640x640)
  - FP32 normalization
  - YOLOv8 output decoding + Non-Maximum Suppression
  - Per-stage timing (preprocess / inference / postprocess) for the dashboard

Output contract per ARCHITECTURE.md §2.2:
    [
        {"class": "Person",   "confidence": 0.91, "bbox": [x1, y1, x2, y2], "track_id": None},
        {"class": "Hardhat",  "confidence": 0.88, "bbox": [x1, y1, x2, y2], "track_id": None},
        ...
    ]
"""

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort

logger = logging.getLogger("edge_ai.inference")

# Class labels as trained in Hansung-Cho/yolov8-ppe-detection (verified Phase 1)
DEFAULT_CLASS_NAMES: Dict[int, str] = {
    0: "Hardhat",
    1: "Mask",
    2: "NO-Hardhat",
    3: "NO-Mask",
    4: "NO-Safety Vest",
    5: "Person",
    6: "Safety Cone",
    7: "Safety Vest",
    8: "machinery",
    9: "vehicle",
}


@dataclass
class Detection:
    """Single object detection result from one frame."""
    class_name: str
    confidence: float
    bbox: List[float]           # [x1, y1, x2, y2] in original image pixel coords
    track_id: Optional[int] = None

    def to_dict(self) -> dict:
        """Serializes to the output contract defined in ARCHITECTURE.md §2.2."""
        return {
            "class": self.class_name,
            "confidence": round(self.confidence, 4),
            "bbox": [round(v, 1) for v in self.bbox],
            "track_id": self.track_id,
        }


@dataclass
class InferenceTiming:
    """Per-frame timing breakdown in milliseconds."""
    preprocess_ms: float = 0.0
    inference_ms: float = 0.0
    postprocess_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        return self.preprocess_ms + self.inference_ms + self.postprocess_ms

    def to_dict(self) -> dict:
        return {
            "preprocess_ms": round(self.preprocess_ms, 2),
            "inference_ms": round(self.inference_ms, 2),
            "postprocess_ms": round(self.postprocess_ms, 2),
            "total_ms": round(self.total_ms, 2),
        }


class ONNXInferenceEngine:
    """
    Loads a YOLOv8 ONNX model and runs CPU inference via onnxruntime.
    Reports per-stage timing so the dashboard can show real numbers.
    """

    def __init__(
        self,
        model_path: str,
        input_size: Tuple[int, int] = (640, 640),
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        class_names: Optional[Dict[int, str]] = None,
    ):
        """
        Initializes the ONNX inference session.

        Args:
            model_path: Path to the .onnx model file.
            input_size: (height, width) for model input. Default 640x640.
            conf_threshold: Minimum confidence to keep a detection.
            iou_threshold: IoU threshold for NMS.
            class_names: Map of class index → label string.
        """
        self.model_path = model_path
        self.input_h, self.input_w = input_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.class_names = class_names or DEFAULT_CLASS_NAMES

        self._ensure_model_exists(model_path)

        logger.info("Loading ONNX model: %s", model_path)
        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )

        # Cache IO metadata
        self._input_name = self.session.get_inputs()[0].name
        self._output_name = self.session.get_outputs()[0].name
        logger.info(
            "Model loaded — input: %s %s | output: %s %s",
            self._input_name,
            self.session.get_inputs()[0].shape,
            self._output_name,
            self.session.get_outputs()[0].shape,
        )

        # Warm up the session so first-frame latency is representative
        self._warmup()

    def _ensure_model_exists(self, model_path: str) -> None:
        """Automatically downloads and exports the PPE model if it does not exist locally."""
        if os.path.exists(model_path):
            return

        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        pt_path = "backend/models/Hansung-Cho_yolov8-ppe-detection.pt"
        if not os.path.exists(pt_path):
            url = "https://huggingface.co/Hansung-Cho/yolov8-ppe-detection/resolve/main/best.pt"
            logger.info("Downloading PPE model weights from Hugging Face: %s", url)
            try:
                import urllib.request
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=60) as resp, open(pt_path, "wb") as f:
                    f.write(resp.read())
                logger.info("Downloaded %s successfully.", pt_path)
            except Exception as e:
                logger.error("Failed to download model weights: %s", e)
                raise

        logger.info("Exporting %s to ONNX format...", pt_path)
        from ultralytics import YOLO
        import shutil

        model = YOLO(pt_path)
        exported = model.export(
            format="onnx",
            imgsz=self.input_w,
            opset=17,
            simplify=True,
            dynamic=False,
            half=False,
            verbose=False,
        )

        default_output = pt_path.replace(".pt", ".onnx")
        if default_output != model_path and os.path.exists(default_output):
            shutil.move(default_output, model_path)
        elif exported and str(exported) != model_path and os.path.exists(str(exported)):
            shutil.move(str(exported), model_path)

        logger.info("ONNX model ready at %s", model_path)

    def _warmup(self, rounds: int = 3) -> None:
        """Runs a few dummy inferences to warm up the ONNX runtime JIT."""
        dummy = np.zeros((1, 3, self.input_h, self.input_w), dtype=np.float32)
        for _ in range(rounds):
            self.session.run([self._output_name], {self._input_name: dummy})
        logger.info("Inference engine warmed up (%d rounds).", rounds)

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _preprocess(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, float, Tuple[float, float]]:
        """
        Letterbox-resizes and normalizes the frame.
        Returns (input_tensor NCHW float32, scale_ratio, (pad_left, pad_top)).
        """
        ih, iw = frame_bgr.shape[:2]
        # Compute scale to fit inside (input_h, input_w) maintaining aspect ratio
        scale = min(self.input_h / ih, self.input_w / iw)
        new_h, new_w = int(ih * scale), int(iw * scale)

        # Resize
        resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Pad to exact model input size with grey (114)
        pad_top = (self.input_h - new_h) // 2
        pad_left = (self.input_w - new_w) // 2
        pad_bottom = self.input_h - new_h - pad_top
        pad_right = self.input_w - new_w - pad_left
        padded = cv2.copyMakeBorder(
            resized, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )

        # BGR → RGB, HWC → CHW, normalize to [0, 1], add batch dim
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))          # (C, H, W)
        tensor = np.expand_dims(tensor, axis=0)            # (1, C, H, W)
        return tensor, scale, (pad_left, pad_top)

    # ------------------------------------------------------------------
    # Postprocessing
    # ------------------------------------------------------------------

    def _postprocess(
        self,
        raw_output: np.ndarray,
        orig_shape: Tuple[int, int],
        scale: float,
        pad: Tuple[float, float],
    ) -> List[Detection]:
        """
        Decodes YOLOv8 output tensor, applies NMS, and maps boxes back to
        original image coordinates.

        YOLOv8 ONNX output shape: (1, num_classes+4, num_anchors)
        Box encoding: [cx, cy, w, h, cls0_score, cls1_score, ...]
        """
        # raw_output shape: (1, 14, 8400) or similar depending on classes
        preds = raw_output[0]  # shape: (num_classes+4, num_anchors)

        # Transpose: (num_anchors, num_classes+4)
        preds = preds.T  # (8400, 14)

        num_classes = len(self.class_names)
        boxes_xywh = preds[:, :4]                  # cx, cy, w, h
        class_scores = preds[:, 4:4 + num_classes] # class probabilities

        # Class confidence = max score per anchor
        class_ids = np.argmax(class_scores, axis=1)
        confidences = class_scores[np.arange(len(class_scores)), class_ids]

        # Filter by confidence threshold
        mask = confidences >= self.conf_threshold
        if not mask.any():
            return []

        boxes_xywh = boxes_xywh[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        # Convert cx,cy,w,h → x1,y1,x2,y2
        x1 = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
        y1 = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
        x2 = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
        y2 = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2

        # OpenCV NMS wants [x, y, w, h]
        pad_left, pad_top = pad
        orig_h, orig_w = orig_shape

        boxes_for_nms = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
        indices = cv2.dnn.NMSBoxes(
            boxes_for_nms,
            confidences.tolist(),
            self.conf_threshold,
            self.iou_threshold,
        )

        detections: List[Detection] = []
        if len(indices) == 0:
            return detections

        # cv2.dnn.NMSBoxes returns shape (N,) in OpenCV 4.x
        indices = np.array(indices).flatten()
        for idx in indices:
            # Map letterbox coords back to original frame
            bx1 = float((x1[idx] - pad_left) / scale)
            by1 = float((y1[idx] - pad_top) / scale)
            bx2 = float((x2[idx] - pad_left) / scale)
            by2 = float((y2[idx] - pad_top) / scale)

            # Clamp to frame boundaries
            bx1 = max(0.0, min(bx1, float(orig_w)))
            by1 = max(0.0, min(by1, float(orig_h)))
            bx2 = max(0.0, min(bx2, float(orig_w)))
            by2 = max(0.0, min(by2, float(orig_h)))

            cls_name = self.class_names.get(int(class_ids[idx]), f"cls_{class_ids[idx]}")
            detections.append(Detection(
                class_name=cls_name,
                confidence=float(confidences[idx]),
                bbox=[bx1, by1, bx2, by2],
            ))

        return detections

    # ------------------------------------------------------------------
    # Public inference method
    # ------------------------------------------------------------------

    def infer(self, frame_bgr: np.ndarray) -> Tuple[List[Detection], InferenceTiming]:
        """
        Runs the full detection pipeline on one BGR frame.
        Returns (list of Detection, InferenceTiming with per-stage ms).
        Degrades gracefully: logs exceptions and returns empty list on failure.
        """
        timing = InferenceTiming()
        try:
            orig_shape = frame_bgr.shape[:2]  # (h, w)

            t0 = time.perf_counter()
            tensor, scale, pad = self._preprocess(frame_bgr)
            timing.preprocess_ms = (time.perf_counter() - t0) * 1000

            t1 = time.perf_counter()
            raw_output = self.session.run(
                [self._output_name], {self._input_name: tensor}
            )
            timing.inference_ms = (time.perf_counter() - t1) * 1000

            t2 = time.perf_counter()
            detections = self._postprocess(raw_output[0], orig_shape, scale, pad)
            timing.postprocess_ms = (time.perf_counter() - t2) * 1000

        except Exception as exc:
            logger.error("Inference failed on frame: %s", exc, exc_info=True)
            return [], timing

        return detections, timing

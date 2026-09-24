"""
End-to-End Edge AI PPE & Hazard Triage Real-Time Pipeline.

Wires together:
  VideoSource → ONNXInferenceEngine → PPEAssociator →
  ComplianceEngine → HazardEngine → TemporalConfirmationBuffer

Per ARCHITECTURE.md §3 and WORKFLOW.md Phase 5.
"""

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from backend.app.compliance.association import PPEAssociator
from backend.app.compliance.engine import ComplianceEngine, ComplianceResult
from backend.app.hazard.engine import HazardEngine, HazardResult
from backend.app.hazard.temporal import ConfirmedAlert, TemporalConfirmationBuffer
from backend.app.inference.engine import Detection, ONNXInferenceEngine
from backend.app.utils.video_source import VideoSource

logger = logging.getLogger("edge_ai.pipeline")


@dataclass
class FrameProcessingResult:
    """Consolidated results from processing a single video frame."""
    frame_idx: int
    timestamp: str
    detections: List[Detection]
    compliances: List[ComplianceResult]
    hazards: List[HazardResult]
    confirmed_alerts: List[ConfirmedAlert]
    metrics: Dict[str, float]
    annotated_frame: Optional[np.ndarray] = None


class EdgeAIPipeline:
    """
    Main orchestration class for the real-time edge processing pipeline.
    """

    def __init__(
        self,
        model_path: str = "backend/models/yolov8n_ppe.onnx",
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        required_ppe: Optional[set] = None,
        restricted_zones: Optional[List[List[Tuple[float, float]]]] = None,
        window_size: int = 5,
        confirm_threshold: int = 4,
    ):
        logger.info("Initializing EdgeAIPipeline with model: %s", model_path)
        self.inference_engine = ONNXInferenceEngine(
            model_path=model_path,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
        )

        self.associator = PPEAssociator()
        self.compliance_engine = ComplianceEngine(
            required_ppe=required_ppe or {"helmet", "vest"}
        )

        # Default restricted zone (from synthetic demo scene or site config)
        if restricted_zones is None:
            # (380, 180) to (620, 450)
            restricted_zones = [
                [(380.0, 180.0), (620.0, 180.0), (620.0, 450.0), (380.0, 450.0)]
            ]

        self.hazard_engine = HazardEngine(
            restricted_zones=restricted_zones,
            machinery_proximity_px=160.0,
        )

        self.temporal_buffer = TemporalConfirmationBuffer(
            window_size=window_size,
            confirm_threshold=confirm_threshold,
            alert_cooldown_seconds=3.0,
        )

        # Performance tracking metrics
        self.fps_tracker = deque_fps(maxlen=30)
        self.total_frames_processed = 0

    def process_frame(
        self,
        frame: np.ndarray,
        frame_idx: int = 0,
        annotate: bool = True,
    ) -> FrameProcessingResult:
        """
        Runs the complete 6-stage pipeline on one BGR frame.
        """
        t_start = time.perf_counter()
        timestamp_str = time.strftime("%H:%M:%S")

        # 1. Inference Engine
        detections, timing = self.inference_engine.infer(frame)

        # 2. PPE Association
        person_records = self.associator.associate(detections)

        # 3. Compliance Engine
        compliances = self.compliance_engine.score_all(person_records)

        # 4. Hazard Triage Engine
        hazards = self.hazard_engine.score_all(compliances, detections)

        # 5. Temporal Confirmation
        confirmed_alerts = self.temporal_buffer.update(hazards, timestamp_str)

        # 6. Overall Metrics
        t_end = time.perf_counter()
        total_latency_ms = (t_end - t_start) * 1000.0
        self.fps_tracker.update(t_end - t_start)
        current_fps = self.fps_tracker.get_fps()
        self.total_frames_processed += 1

        metrics = {
            "fps": round(current_fps, 1),
            "total_latency_ms": round(total_latency_ms, 1),
            "preprocess_ms": round(timing.preprocess_ms, 1),
            "inference_ms": round(timing.inference_ms, 1),
            "postprocess_ms": round(timing.postprocess_ms, 1),
            "persons_detected": len(hazards),
            "violations_active": sum(1 for h in hazards if h.severity in ("MEDIUM", "HIGH", "CRITICAL")),
            "confirmed_alerts_count": len(confirmed_alerts),
        }

        # Optional visual annotation
        annotated_frame = None
        if annotate:
            annotated_frame = self.annotate_frame(
                frame.copy(), detections, hazards, confirmed_alerts, metrics
            )

        return FrameProcessingResult(
            frame_idx=frame_idx,
            timestamp=timestamp_str,
            detections=detections,
            compliances=compliances,
            hazards=hazards,
            confirmed_alerts=confirmed_alerts,
            metrics=metrics,
            annotated_frame=annotated_frame,
        )

    def annotate_frame(
        self,
        img: np.ndarray,
        detections: List[Detection],
        hazards: List[HazardResult],
        confirmed_alerts: List[ConfirmedAlert],
        metrics: Dict[str, float],
    ) -> np.ndarray:
        """
        Draws professional real-time HUD and detection boxes onto frame.
        """
        h, w = img.shape[:2]

        # Draw Restricted Zone polygon overlay
        for zone in self.hazard_engine.restricted_zones:
            pts = np.array(zone, dtype=np.int32)
            overlay = img.copy()
            cv2.fillPoly(overlay, [pts], (0, 0, 160))
            cv2.addWeighted(overlay, 0.25, img, 0.75, 0, dst=img)
            cv2.polylines(img, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
            cx = int(np.mean([p[0] for p in zone]))
            cy = int(zone[0][1] + 20)
            cv2.putText(
                img, "ZONE RESTRICTED (DANGER)", (cx - 100, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA
            )

        # Draw Person Bounding Boxes with Severity Color
        severity_colors = {
            "SAFE": (46, 204, 113),       # Emerald Green
            "LOW": (52, 152, 219),        # Dodger Blue
            "MEDIUM": (34, 126, 230),     # Amber / Orange
            "HIGH": (60, 76, 231),        # Crimson Red
            "CRITICAL": (0, 0, 255),      # Pure Red
        }

        for hazard in hazards:
            bx1, by1, bx2, by2 = [int(v) for v in hazard.person_bbox]
            color = severity_colors.get(hazard.severity, (200, 200, 200))

            # Bounding box
            thickness = 3 if hazard.severity in ("HIGH", "CRITICAL") else 2
            cv2.rectangle(img, (bx1, by1), (bx2, by2), color, thickness)

            # Badge banner on top
            badge_text = f"WORKER | {hazard.severity} (Score: {hazard.risk_score})"
            if hazard.critical_override:
                badge_text = "CRITICAL: FALL/HAZARD OVERRIDE"

            (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(img, (bx1, by1 - 22), (bx1 + tw + 10, by1), color, -1)
            cv2.putText(
                img, badge_text, (bx1 + 5, by1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA
            )

            # PPE Checklist below box
            checklist = []
            checklist.append(f"H:{'OK' if hazard.helmet else 'NO'}")
            checklist.append(f"V:{'OK' if hazard.vest else 'NO'}")
            if hazard.gloves is not None:
                checklist.append(f"G:{'OK' if hazard.gloves else 'NO'}")
            checklist_text = " | ".join(checklist)

            cv2.rectangle(img, (bx1, by2), (bx1 + 140, by2 + 18), (30, 30, 30), -1)
            cv2.putText(
                img, checklist_text, (bx1 + 4, by2 + 13),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1, cv2.LINE_AA
            )

        # Draw HUD bar at top of frame
        cv2.rectangle(img, (0, 0), (w, 36), (18, 20, 24), -1)
        cv2.line(img, (0, 36), (w, 36), (60, 65, 75), 1)

        hud_left = (
            f"FPS: {metrics['fps']:.1f}  |  "
            f"Latency: {metrics['total_latency_ms']:.1f}ms  "
            f"(Inf: {metrics['inference_ms']:.1f}ms)  |  "
            f"Active: {metrics['persons_detected']}  |  "
            f"Violations: {metrics['violations_active']}"
        )
        cv2.putText(
            img, hud_left, (12, 23),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (240, 240, 240), 1, cv2.LINE_AA
        )

        return img


class deque_fps:
    """Helper to compute smoothed rolling FPS."""
    def __init__(self, maxlen: int = 30):
        self.times: List[float] = []
        self.maxlen = maxlen

    def update(self, dt: float):
        self.times.append(dt)
        if len(self.times) > self.maxlen:
            self.times.pop(0)

    def get_fps(self) -> float:
        if not self.times:
            return 0.0
        avg_dt = sum(self.times) / len(self.times)
        return (1.0 / avg_dt) if avg_dt > 0 else 0.0

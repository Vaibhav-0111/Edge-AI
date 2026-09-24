"""Exports the verified YOLOv8 PPE checkpoint to ONNX for onnxruntime inference."""

import os
import sys

# Add project root to path so we can import from backend/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def export_onnx(
    pt_path: str = "backend/models/Hansung-Cho_yolov8-ppe-detection.pt",
    output_path: str = "backend/models/yolov8n_ppe.onnx",
    imgsz: int = 640,
    opset: int = 17,
    simplify: bool = True,
    dynamic: bool = False,
) -> str:
    """
    Converts a YOLOv8 .pt model to ONNX using the ultralytics export API.
    Returns the path to the exported ONNX file.
    """
    from ultralytics import YOLO

    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"Checkpoint not found: {pt_path}. Run check_ppe_model.py first.")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Loading checkpoint from {pt_path}...")
    model = YOLO(pt_path)

    print(f"Exporting to ONNX  →  opset={opset}, imgsz={imgsz}, simplify={simplify}")
    exported = model.export(
        format="onnx",
        imgsz=imgsz,
        opset=opset,
        simplify=simplify,
        dynamic=dynamic,
        half=False,       # keep FP32 for CPU inference — half only helps on GPU
        int8=False,
        verbose=False,
    )

    # ultralytics places the file next to the .pt by default; move it if needed
    default_output = pt_path.replace(".pt", ".onnx")
    if default_output != output_path and os.path.exists(default_output):
        import shutil
        shutil.move(default_output, output_path)
        print(f"Moved exported file to {output_path}")
    elif exported and str(exported) != output_path and os.path.exists(str(exported)):
        import shutil
        shutil.move(str(exported), output_path)
        print(f"Moved exported file to {output_path}")

    if not os.path.exists(output_path):
        raise RuntimeError(f"Export appeared to succeed but ONNX file not found at {output_path}")

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Export successful! ONNX model saved to: {output_path}  ({size_mb:.2f} MB)")

    # Quick sanity-check the ONNX graph with onnxruntime
    print("Running onnxruntime sanity check...")
    import numpy as np
    import onnxruntime as ort

    sess = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    out = sess.get_outputs()
    print(f"  Input  : {inp.name}  shape={inp.shape}  dtype={inp.type}")
    for o in out:
        print(f"  Output : {o.name}  shape={o.shape}  dtype={o.type}")

    dummy = np.zeros([1, 3, imgsz, imgsz], dtype=np.float32)
    result = sess.run(None, {inp.name: dummy})
    print(f"  Dummy run output shape: {result[0].shape}")
    print("Sanity check passed.")
    return output_path


if __name__ == "__main__":
    export_onnx()

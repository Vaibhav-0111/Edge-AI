"""
Benchmark script — measures real CPU preprocess/inference/postprocess latency and FPS.

Run from project root:
    .\\venv\\Scripts\\python.exe scripts/benchmark.py

Numbers printed here are the ones that appear on the dashboard.
Never fabricate these — RULES.md §1 is explicit about this.
"""

import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from backend.app.inference.engine import ONNXInferenceEngine, InferenceTiming

ONNX_MODEL = "backend/models/yolov8n_ppe.onnx"
WARMUP_FRAMES = 10
BENCHMARK_FRAMES = 100
INPUT_VIDEO = "dataset/raw/real_ppe_test.mp4"


def run_benchmark() -> None:
    if not os.path.exists(ONNX_MODEL):
        print(f"ONNX model not found at {ONNX_MODEL}. Run scripts/export_onnx.py first.")
        sys.exit(1)

    print("=" * 60)
    print(f"  Edge AI PPE — CPU Inference Benchmark")
    print(f"  Model : {ONNX_MODEL}")
    print(f"  Frames: {BENCHMARK_FRAMES} (after {WARMUP_FRAMES} warmup)")
    print("=" * 60)

    engine = ONNXInferenceEngine(
        model_path=ONNX_MODEL,
        input_size=(640, 640),
        conf_threshold=0.35,
        iou_threshold=0.45,
    )

    # Prefer real footage, fall back to generated demo video or dummy frames
    if os.path.exists(INPUT_VIDEO):
        cap = cv2.VideoCapture(INPUT_VIDEO)
        use_video = True
    else:
        print(f"Warning: {INPUT_VIDEO} not found, using randomly generated frames.")
        use_video = False
        cap = None

    def next_frame():
        if use_video and cap is not None:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                _, frame = cap.read()
            return frame
        else:
            return (np.random.rand(480, 640, 3) * 255).astype(np.uint8)

    print("\nRunning warmup frames (not counted)...")
    for _ in range(WARMUP_FRAMES):
        f = next_frame()
        engine.infer(f)

    print(f"Running {BENCHMARK_FRAMES} benchmark frames...")
    timings: list[InferenceTiming] = []
    total_wall_start = time.perf_counter()

    for i in range(BENCHMARK_FRAMES):
        f = next_frame()
        if f is None:
            continue
        _, t = engine.infer(f)
        timings.append(t)

    total_wall_ms = (time.perf_counter() - total_wall_start) * 1000

    if cap is not None:
        cap.release()

    if not timings:
        print("No timing data collected.")
        return

    pre_ms  = [t.preprocess_ms  for t in timings]
    inf_ms  = [t.inference_ms   for t in timings]
    post_ms = [t.postprocess_ms for t in timings]
    total_ms = [t.total_ms      for t in timings]

    def stats(vals):
        arr = np.array(vals)
        return arr.mean(), arr.std(), arr.min(), np.percentile(arr, 95), arr.max()

    print("\n" + "=" * 60)
    print(f"  RESULTS ({len(timings)} frames on CPU)")
    print("=" * 60)
    for label, vals in [("Preprocess", pre_ms), ("Inference", inf_ms), ("Postprocess", post_ms), ("Total", total_ms)]:
        mean, std, mn, p95, mx = stats(vals)
        print(f"  {label:<12}: mean={mean:.1f}ms  std={std:.1f}  min={mn:.1f}  p95={p95:.1f}  max={mx:.1f}")

    mean_total = np.mean(total_ms)
    fps = 1000.0 / mean_total if mean_total > 0 else 0
    wall_fps = (len(timings) / total_wall_ms) * 1000.0

    print("=" * 60)
    print(f"  Effective FPS (1000/mean_total)  : {fps:.1f}")
    print(f"  Wall-clock FPS (incl. overhead)  : {wall_fps:.1f}")
    print("=" * 60)

    # Save results to a file so they can be referenced in the README / dashboard
    result_path = "docs/benchmark_results.txt"
    os.makedirs("docs", exist_ok=True)
    with open(result_path, "w") as f:
        f.write(f"Edge AI PPE — CPU Benchmark Results\n")
        f.write(f"Model: {ONNX_MODEL}\n")
        f.write(f"Frames: {len(timings)}\n\n")
        for label, vals in [("Preprocess", pre_ms), ("Inference", inf_ms), ("Postprocess", post_ms), ("Total", total_ms)]:
            mean, std, mn, p95, mx = stats(vals)
            f.write(f"{label}: mean={mean:.1f}ms  std={std:.1f}  min={mn:.1f}  p95={p95:.1f}  max={mx:.1f}\n")
        f.write(f"\nEffective FPS: {fps:.1f}\n")
        f.write(f"Wall-clock FPS: {wall_fps:.1f}\n")
    print(f"\nResults saved to {result_path}")


if __name__ == "__main__":
    run_benchmark()

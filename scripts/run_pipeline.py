"""
Run Pipeline script — processes MP4 video end-to-end through EdgeAIPipeline.
Validates Phase 5 before adding WebSocket / Frontend layers.
"""

import os
import sys
import time

# Ensure UTF-8 output encoding for Windows shells
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "replace")
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, "replace")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app.pipeline import EdgeAIPipeline
from backend.app.utils.video_source import VideoSource


def main():
    # Prefer real_ppe_test.mp4 if present for authentic camera detections
    default_vid = "dataset/raw/real_ppe_test.mp4"
    if not os.path.exists(default_vid):
        default_vid = "dataset/raw/demo_sample.mp4"

    video_path = sys.argv[1] if len(sys.argv) > 1 else default_vid

    if not os.path.exists(video_path):
        print(f"ERROR: Video file {video_path} not found.")
        sys.exit(1)

    print("=" * 65)
    print("  EDGE AI PPE SAFETY & HAZARD TRIAGE PIPELINE — REAL-TIME RUNNER")
    print("=" * 65)

    source = VideoSource(source=video_path, loop=False, max_fps=30)
    pipeline = EdgeAIPipeline(
        model_path="backend/models/yolov8n_ppe.onnx",
        conf_threshold=0.35,
        window_size=5,
        confirm_threshold=4,
    )

    total_alerts_emitted = 0
    frame_count = 0
    start_time = time.time()

    print(f"Ingesting: {video_path} ({source.resolution[0]}x{source.resolution[1]} @ {source.fps} FPS)")
    print("Processing frames...\n")

    for frame_idx, frame in source.frames():
        result = pipeline.process_frame(frame, frame_idx=frame_idx, annotate=False)
        frame_count += 1

        # Check for confirmed temporal alerts
        if result.confirmed_alerts:
            for alert in result.confirmed_alerts:
                total_alerts_emitted += 1
                print(
                    f"  [ALERT | FRAME {frame_idx:03d} | {result.timestamp}] "
                    f"{alert.severity} (Risk: {alert.risk_score}) | "
                    f"Reasons: {alert.reasons} | Ratio: {alert.confirmed_ratio:.1%}"
                )

        if frame_idx % 25 == 0:
            m = result.metrics
            print(
                f"[F{frame_idx:03d}] FPS: {m['fps']:.1f} | "
                f"Latency: {m['total_latency_ms']:.1f}ms (Inf: {m['inference_ms']:.1f}ms) | "
                f"Active Persons: {m['persons_detected']} | Violations: {m['violations_active']}"
            )

    elapsed = time.time() - start_time
    avg_fps = frame_count / elapsed if elapsed > 0 else 0

    print("\n" + "=" * 65)
    print("  PIPELINE PROCESSING COMPLETED")
    print("=" * 65)
    print(f"Total Frames Processed : {frame_count}")
    print(f"Elapsed Time           : {elapsed:.2f} s")
    print(f"Average Pipeline FPS   : {avg_fps:.1f} FPS")
    print(f"Confirmed Alerts Fired : {total_alerts_emitted}")
    print("=" * 65)

    assert frame_count > 0, "No frames processed!"
    print("[SUCCESS] Phase 5 real-time pipeline verified successfully!")


if __name__ == "__main__":
    main()

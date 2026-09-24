"""Validates detection quality on sample frames using the verified checkpoint."""

import os
import cv2
from ultralytics import YOLO

def main():
    model_path = "backend/models/Hansung-Cho_yolov8-ppe-detection.pt"
    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}. Run check_ppe_model.py first.")
        return

    model = YOLO(model_path)
    video_path = "dataset/raw/real_ppe_test.mp4"
    if not os.path.exists(video_path):
        video_path = "dataset/raw/demo_sample.mp4"

    cap = cv2.VideoCapture(video_path)
    test_indices = [15, 35, 55]
    os.makedirs("docs/diagrams", exist_ok=True)

    for idx in test_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        results = model.predict(frame, conf=0.35, verbose=False)
        res = results[0]
        annotated_frame = res.plot()

        out_img = f"docs/diagrams/val_frame_{idx}.jpg"
        cv2.imwrite(out_img, annotated_frame)
        print(f"Frame {idx}: Detected {len(res.boxes)} objects:")
        for b in res.boxes:
            cls_id = int(b.cls[0].item())
            cls_name = model.names[cls_id]
            conf = float(b.conf[0].item())
            box = [round(x, 1) for x in b.xyxy[0].tolist()]
            print(f"  - {cls_name} ({conf:.2f}): {box}")

    cap.release()
    print("Visual validation complete. Sample images saved in docs/diagrams/")

if __name__ == "__main__":
    main()

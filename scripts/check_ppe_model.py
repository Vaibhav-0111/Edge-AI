"""Checks public pretrained YOLOv8 PPE models and inspects class labels."""

import os
import sys

def main():
    print("Checking YOLOv8 PPE model availability and classes...")
    from ultralytics import YOLO

    candidates = [
        ("Hansung-Cho/yolov8-ppe-detection", "https://huggingface.co/Hansung-Cho/yolov8-ppe-detection/resolve/main/best.pt"),
        ("keremberke/yolov8n-protective-equipment-detection", "https://huggingface.co/keremberke/yolov8n-protective-equipment-detection/resolve/main/best.pt"),
        ("Hexmon/vyra-yolo-ppe-detection", "https://huggingface.co/Hexmon/vyra-yolo-ppe-detection/resolve/main/best.pt"),
    ]

    os.makedirs("backend/models", exist_ok=True)
    loaded = False

    for name, url in candidates:
        print(f"\n--- Checking candidate: {name} ---")
        pt_path = f"backend/models/{name.replace('/', '_')}.pt"
        if not os.path.exists(pt_path):
            print(f"Downloading {url} to {pt_path}...")
            try:
                import urllib.request
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=30) as resp, open(pt_path, "wb") as f:
                    f.write(resp.read())
                print(f"Downloaded successfully: {os.path.getsize(pt_path)} bytes")
            except Exception as e:
                print(f"Download failed for {name}: {e}")
                continue

        try:
            model = YOLO(pt_path)
            print(f"Successfully loaded {name}!")
            print(f"Model task: {getattr(model, 'task', 'unknown')}")
            print(f"Model class names ({len(model.names)}):")
            for idx, cname in model.names.items():
                print(f"  [{idx}]: {cname}")
            loaded = True
            break
        except Exception as e:
            print(f"Failed to load {pt_path} with YOLO: {e}")

    if not loaded:
        print("\nChecking baseline YOLOv8n (COCO standard)...")
        try:
            model = YOLO("yolov8n.pt")
            print("Loaded default yolov8n.pt successfully!")
            print(f"Sample classes: {list(model.names.items())[:10]}")
        except Exception as e:
            print(f"Failed to load yolov8n: {e}")

if __name__ == "__main__":
    main()

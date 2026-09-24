# Architecture — Edge AI PPE Compliance & Hazard Triaging

## 1. High-Level Pipeline

```
CCTV / IP Camera / MP4
        │
        ▼
    OpenCV (frame capture + preprocessing)
        │
        ▼
    YOLOv8 model → exported ONNX → ONNX Runtime (CPU inference)
        │
        ▼
    Object Detection output (person, helmet, vest, gloves, boots, goggles,
    fire/smoke, fallen-person)
        │
        ▼
    PPE Association (match each PPE box to nearest person box, IoU/distance)
        │
        ▼
    ┌───────────────────────────┐
    │ Compliance Engine         │  → per-person PPE completeness %
    │ Hazard/Risk Triage Engine │  → zone + proximity + hazard-type scoring
    └───────────────────────────┘
        │
        ▼
    Temporal Confirmation (N consecutive frames before an alert fires)
        │
        ▼
    WebSocket Broadcaster (JSON event) → FastAPI backend
        │
        ▼
    Browser Dashboard (live video overlay, alert feed, system metrics)
```

## 2. Module Breakdown

### 2.1 Video Ingestion (`backend/app/utils/video_source.py`)
- Wraps OpenCV `VideoCapture` for MP4 files and RTSP streams behind one
  interface so the rest of the pipeline doesn't care about the source.
- Responsible for frame resizing/normalization before inference.

### 2.2 Inference Engine (`backend/app/inference/`)
- Loads the exported `.onnx` YOLOv8 model via `onnxruntime.InferenceSession`.
- Runs preprocessing (letterbox resize, normalize) → session.run() →
  postprocessing (NMS, box decoding).
- Reports per-stage timing (preprocess / inference / postprocess) so the
  dashboard can show real latency numbers.
- Output contract (per frame):
  ```python
  [
    {"class": "person", "confidence": 0.94, "bbox": [x1, y1, x2, y2], "track_id": 12},
    {"class": "helmet", "confidence": 0.91, "bbox": [x1, y1, x2, y2]},
    ...
  ]
  ```

### 2.3 PPE Association (`backend/app/compliance/association.py`)
- For each `person` box, find PPE boxes whose center falls within the
  person's region (or nearest by distance) and whose vertical position is
  plausible for that PPE item (e.g. helmet near the top of the box).
- Produces a per-person PPE checklist: `{helmet: bool, vest: bool, gloves: bool, boots: bool}`.

### 2.4 Compliance Engine (`backend/app/compliance/engine.py`)
- Converts the checklist into a compliance percentage and a simple
  COMPLIANT / VIOLATION status.
- Required PPE set is configurable per site (e.g. construction site vs.
  chemical plant require different items).

### 2.5 Hazard Triage Engine (`backend/app/hazard/engine.py`)
- Computes a **risk score** by summing weighted contributions:

  | Factor                          | Weight |
  |----------------------------------|--------|
  | Missing helmet                   | +30    |
  | Missing vest                     | +20    |
  | Missing gloves near machinery    | +20    |
  | Inside restricted zone           | +30    |
  | Near machinery/hazard source     | +20    |
  | Person fallen                    | → CRITICAL override |
  | Fire/smoke detected              | → CRITICAL override |

- Maps score ranges to severity:
  - `0` → SAFE
  - `1–39` → LOW
  - `40–69` → MEDIUM
  - `70–99` → HIGH
  - `100+` or override event → CRITICAL

### 2.6 Temporal Confirmation (`backend/app/hazard/temporal.py`)
- Maintains a short rolling buffer (e.g. last 5 frames) per tracked person.
- A violation is only "confirmed" (and broadcast as an alert) once it
  appears in a configurable threshold (e.g. 4 of the last 5 frames).
- This is what separates this system from a naive "YOLO says no helmet in
  frame 1" demo — it's explicitly called out in the pitch as **temporal
  violation confirmation**.

### 2.7 WebSocket Layer (`backend/app/websocket/`)
- `manager.py` — tracks connected dashboard clients, handles connect/
  disconnect, broadcasts JSON messages to all clients.
- Message schema:
  ```json
  {
    "camera": "CAM_01",
    "person_id": 12,
    "helmet": false,
    "vest": true,
    "gloves": false,
    "risk_score": 65,
    "severity": "HIGH",
    "reason": ["missing_helmet", "near_machinery"],
    "timestamp": "12:31:08"
  }
  ```
- System metrics are broadcast on a separate lightweight channel/message
  type: `{"type": "metrics", "fps": 21, "latency_ms": 47, "cpu_pct": 64}`.

### 2.8 Dashboard (`frontend/`)
- Plain HTML/CSS/JS (no framework) to keep build time low.
- Left panel: live video feed with bounding-box overlay drawn on a
  `<canvas>` synced to detection coordinates.
- Right panel: system metrics (FPS, latency, CPU) + scrolling alert feed
  color-coded by severity (🟢 🟠 🔴).
- Connects to the backend via `new WebSocket("ws://<host>:8000/ws")` and
  renders each incoming message as it arrives.

## 3. Data Flow Summary

```
Frame → Inference → Association → Compliance + Hazard scoring
      → Temporal confirmation → WebSocket event → Dashboard render
```

## 4. Model & Dataset Strategy

```
Dataset (PPE-labeled images/video)
   ↓
YOLOv8 fine-tuning (scripts/train_yolo.py)
   ↓
Validation
   ↓
Export to ONNX (scripts/export_onnx.py)
   ↓
Benchmark on CPU (scripts/benchmark.py) — record real preprocess/inference/
postprocess timings, do not fabricate numbers for the demo
   ↓
Wire into backend/app/inference/
```

If a pretrained PPE-detection YOLOv8 checkpoint already covers the needed
classes (helmet, vest, gloves, boots), fine-tuning can be skipped or
minimal — but this must be verified early (see `WORKFLOW.md` Phase 1),
since dataset availability is the single biggest project risk.

## 5. Deployment Target

- Demo: any laptop, CPU-only, MP4 input.
- Stretch goal: Raspberry Pi / Jetson with RTSP camera input, to make the
  "edge" claim concrete during judging.

## 6. Non-Functional Targets

- Inference latency: sub-100 ms per frame on CPU (report actual measured
  numbers on the dashboard, never a hardcoded/fake value).
- Dashboard update latency: near-real-time (<300 ms from detection to
  render, given local network).
- False-alarm reduction via temporal confirmation (target: no single-frame
  flicker alerts in the demo recording).

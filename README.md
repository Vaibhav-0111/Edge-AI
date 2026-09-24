# Edge AI Vision Pipeline for Real-Time PPE Compliance and Hazard Triaging

**Category:** Computer Vision · Edge AI · Real-Time Systems · Workplace Safety
**Stack:** OpenCV · YOLOv8 · ONNX Runtime · FastAPI · WebSockets · HTML/CSS/JS

---

## 1. The Pitch (30 seconds)

> Our system converts existing CCTV infrastructure into an edge-based AI safety
> system. Video is processed locally using OpenCV and an optimized YOLOv8 model
> running through ONNX Runtime. The system detects workers and their PPE,
> verifies compliance, identifies hazardous situations such as restricted-zone
> entry, and assigns a risk level. Confirmed violations are pushed through
> WebSockets to a live safety dashboard. Because inference happens at the edge,
> the system minimizes cloud dependency, reduces latency, and can continue
> operating even with limited connectivity.

## 2. What It Does

- Ingests a video stream (MP4 file for demo, RTSP/CCTV for real deployment).
- Runs a YOLOv8 model (exported to ONNX, served via ONNX Runtime) to detect
  people and PPE items (helmet, vest, gloves, boots, goggles).
- Associates detected PPE items with the nearest detected person.
- Runs a **Compliance Engine** that scores each worker's PPE completeness.
- Runs a **Hazard Triage Engine** that layers in zone violations, proximity to
  machinery, fall detection, and fire/smoke detection to produce a **risk
  score** and severity level (LOW / MEDIUM / HIGH / CRITICAL).
- Uses **temporal violation confirmation** — a violation must persist across
  several consecutive frames before it is confirmed and alerted, to avoid
  false alarms from single bad frames.
- Streams every confirmed event to a browser dashboard over WebSockets in
  real time, along with live FPS/latency/CPU metrics.

## 3. Why This Wins (Differentiators)

1. **It's a pipeline, not a model demo.** Detection is the input to a real
   decision system (compliance scoring + risk triage), not the end product.
2. **Edge-first story.** ONNX Runtime + CPU inference means no GPU server,
   no cloud round-trip, works on a factory floor with bad internet.
3. **Temporal confirmation** reduces false positives — a detail judges notice.
4. **Live, explainable dashboard** — every alert shows *why* it fired
   (which PPE is missing, which zone, what the risk score components were).

## 4. Tech Stack

```
AI          → YOLOv8, OpenCV, ONNX Runtime
Backend     → Python, FastAPI, WebSockets
Frontend    → HTML, CSS, JavaScript (no framework — keep it fast to build)
Deployment  → CPU / Edge PC / Mini PC (optional: Raspberry Pi, Jetson)
Optional    → SQLite (event log), Docker, Telegram/Email alert webhook
```

## 5. Folder Structure

```
edge-ai-ppe-safety/
├── README.md              → this file
├── RULES.md                → coding standards & engineering rules
├── ARCHITECTURE.md         → detailed system + data-flow architecture
├── WORKFLOW.md             → phased build plan for the hackathon
├── PROMPT.md               → master prompt to hand to an LLM/coding agent
├── requirements.txt
├── .gitignore
├── backend/
│   ├── app/
│   │   ├── main.py                → FastAPI entrypoint
│   │   ├── inference/              → YOLOv8 + ONNX Runtime wrapper
│   │   ├── compliance/             → PPE compliance scoring engine
│   │   ├── hazard/                 → hazard/risk triage engine
│   │   ├── websocket/              → WS connection manager + broadcaster
│   │   └── utils/                  → shared helpers (video I/O, config)
│   └── models/                     → exported .onnx model files (gitignored)
├── frontend/
│   ├── index.html                  → dashboard shell
│   ├── css/                        → dashboard styling
│   └── js/                         → WebSocket client + render logic
├── dataset/
│   ├── raw/                        → source images/videos (gitignored)
│   ├── processed/                  → cleaned/split dataset
│   └── annotations/                → YOLO-format labels
├── scripts/
│   ├── train_yolo.py               → fine-tuning entrypoint
│   ├── export_onnx.py              → PyTorch → ONNX conversion
│   └── benchmark.py                → latency/FPS benchmarking
└── docs/
    └── diagrams/                   → architecture diagrams, screenshots
```

## 6. Setup (once code exists)

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run the backend (serves API + WebSocket)
uvicorn backend.app.main:app --reload --port 8000

# Open the dashboard
# frontend/index.html served by the backend, or open directly in browser
```

## 7. Demo Flow (for judges)

1. **Scene 1 — Compliant worker:** all PPE detected → dashboard shows 🟢.
2. **Scene 2 — Worker removes helmet:** dashboard flips to 🟠 after temporal
   confirmation, shows exactly what's missing and the running risk score.
3. **Scene 3 — Worker enters restricted zone:** risk score stacks (missing
   PPE + zone + proximity) → 🔴 CRITICAL alert pushed instantly.
4. **Scene 4 — Metrics panel:** show live FPS/latency numbers to prove
   real edge inference, not a pre-recorded result.

See `WORKFLOW.md` for the phase-by-phase build plan and `ARCHITECTURE.md`
for full technical detail.

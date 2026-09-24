# Build Workflow — Phased Plan

Build in this order. Each phase should end with something runnable, even
if rough — never leave the pipeline in a broken, non-runnable state
between phases.

## Phase 0 — Setup
- [x] Initialize repo, folder structure (this scaffold), `requirements.txt`,
      `.gitignore`.
- [x] Confirm dev environment: Python version, OpenCV, `ultralytics`
      (YOLOv8), `onnxruntime`, `fastapi`, `uvicorn`, `websockets`.
- [x] Decide the demo video source (MP4 clip) to standardize testing.

## Phase 1 — Dataset & Model Reality Check (do this FIRST — biggest risk)
- [x] Check whether a pretrained YOLOv8 checkpoint (e.g. a public PPE
      detection model) already covers the required classes: person,
      helmet, vest, gloves, boots, goggles.
- [x] If yes → skip/minimize fine-tuning, go straight to export.
- [ ] If no → source a labeled PPE dataset (Roboflow Universe and similar
      public sources are a good first stop), fine-tune YOLOv8 on it.
- [x] Validate detection quality visually on a few sample frames before
      moving on.

## Phase 2 — Inference Engine
- [ ] Export the chosen YOLOv8 model to ONNX (`scripts/export_onnx.py`).
- [ ] Build `backend/app/inference/` to load the ONNX model via
      `onnxruntime` and run detection on a single static image.
- [ ] Confirm output format matches the contract in `ARCHITECTURE.md`
      §2.2.
- [ ] Run `scripts/benchmark.py` to get real preprocess/inference/
      postprocess timings on CPU.

## Phase 3 — Compliance Engine
- [ ] Build PPE association logic (`backend/app/compliance/association.py`).
- [ ] Build compliance scoring (`backend/app/compliance/engine.py`).
- [ ] Test with hand-crafted fake detections first (no model needed) to
      validate scoring logic in isolation.

## Phase 4 — Hazard Triage Engine
- [ ] Build the weighted risk-score calculator
      (`backend/app/hazard/engine.py`) per the table in `ARCHITECTURE.md`.
- [ ] Build temporal confirmation (`backend/app/hazard/temporal.py`).
- [ ] Test severity classification with fake sequences of detections.

## Phase 5 — Real-Time OpenCV Pipeline
- [ ] Wire `utils/video_source.py` → inference → association → compliance
      → hazard → temporal confirmation into one loop processing an MP4
      file frame by frame.
- [ ] Print/log confirmed violation events to the console first, before
      adding WebSockets — confirm the full logic pipeline works
      end-to-end.

## Phase 6 — WebSocket Backend
- [ ] Build `backend/app/websocket/manager.py` (connection tracking,
      broadcast).
- [ ] Wire `main.py` (FastAPI) to serve the WebSocket endpoint and push
      confirmed events + periodic system metrics.

## Phase 7 — Dashboard
- [ ] Build `frontend/index.html` + `css/` + `js/` — video/overlay panel,
      metrics panel, alert feed.
- [ ] Connect dashboard to the WebSocket endpoint, render incoming events
      live.
- [ ] Color-code alerts by severity (🟢 🟠 🔴) per `ARCHITECTURE.md`.

## Phase 8 — Demo Polish
- [ ] Prepare/record the 4-scene demo flow from `README.md` §7 (compliant
      worker → missing PPE → restricted zone entry → live metrics proof).
- [ ] Rehearse the 30-second pitch.
- [ ] Sanity-check: no fabricated numbers on screen, no crashes on
      WebSocket disconnect, works on a clean machine from
      `README.md` §6 setup steps alone.

## Stretch Goals (only after Phase 8 is solid)
- [ ] RTSP/live camera input instead of MP4.
- [ ] Raspberry Pi / Jetson deployment for a literal "edge device" demo.
- [ ] SQLite event log + simple history view.
- [ ] Telegram/email webhook for CRITICAL alerts.
- [ ] Docker packaging for one-command setup.

# Engineering Rules

These are the working rules for anyone (human or LLM agent) contributing
code to this project. Follow them for every phase in `WORKFLOW.md`.

## 1. General Principles

- **Working software over impressive-looking code.** For a hackathon, a
  pipeline that runs end-to-end on Day 1 beats a perfect module that isn't
  wired in yet. Build vertical slices, not one finished layer at a time.
- **No fabricated numbers.** Latency, FPS, and accuracy figures shown on
  the dashboard or in the pitch must come from real measured runs
  (`scripts/benchmark.py`). Never hardcode a "looks good" number.
- **Keep modules independently testable.** Each module in
  `backend/app/{inference,compliance,hazard,websocket}` should be runnable
  and testable on its own with a small script or notebook before being
  wired into the full pipeline.
- **Config over hardcoding.** Required PPE sets, risk weights, and
  temporal-confirmation thresholds live in one config file
  (`backend/app/utils/config.py` or a `.yaml`), not scattered magic numbers.

## 2. Code Style

- **Python:** follow PEP 8. Type-hint function signatures. Use `black` for
  formatting if time allows.
- **File/function naming:** descriptive, no abbreviations that aren't
  obvious (`compliance_engine.py`, not `ce.py`).
- **Docstrings:** every public function gets a one-line docstring
  explaining what it does and what it returns — this matters a lot when
  an LLM agent is picking up work mid-project and needs context fast.
- **JS/HTML/CSS:** keep the frontend framework-free and dependency-light;
  plain `fetch`/`WebSocket` APIs, no build step, so it can be opened and
  debugged instantly during the demo.

## 3. Project Structure Discipline

- New backend logic goes in the module it belongs to (`inference/`,
  `compliance/`, `hazard/`, `websocket/`, `utils/`) — don't dump
  everything into `main.py`.
- Anything model-related (weights, `.onnx` files) stays in
  `backend/models/` and is **gitignored** — never commit large binary
  model files to the repository.
- Raw datasets stay in `dataset/raw/` and are **gitignored** for the same
  reason; only annotation/label files and small samples are tracked.

## 4. Error Handling & Robustness

- The pipeline must degrade gracefully: if the video source drops a frame
  or the model fails on a frame, log it and continue — never crash the
  whole stream mid-demo.
- WebSocket disconnects/reconnects from the dashboard must not crash the
  backend; the connection manager should just drop the dead client.

## 5. Testing Before Integration

- Test the inference module against a single static image before wiring
  it into the video loop.
- Test the compliance + hazard engines with a handful of hand-crafted
  fake detection outputs (no model needed) to confirm the scoring logic
  is correct before trusting real model output.
- Only after both of the above pass, connect the full pipeline to
  WebSockets and the dashboard.

## 6. Documentation Discipline

- If a module's behavior changes in a way that affects the architecture
  (e.g. a new risk factor, a new PPE class, a changed message schema),
  update `ARCHITECTURE.md` in the same work session — don't let the docs
  drift from the code during a hackathon crunch.
- Keep `WORKFLOW.md`'s phase checklist up to date as phases complete, so
  it's always clear what's left before the demo.

## 7. Scope Discipline

- Resist scope creep. The differentiators that matter for judging are:
  compliance engine, hazard/risk triage, temporal confirmation, edge
  (ONNX/CPU) inference, and a live dashboard. Anything beyond that
  (auth, multi-camera orchestration, cloud storage, mobile app) is a
  stretch goal only — build it last, if at all.

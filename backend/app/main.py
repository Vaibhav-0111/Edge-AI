"""
FastAPI Backend Application for Edge AI PPE Safety & Hazard Triage System.

Features:
  - WebSocket endpoint `/ws` for live alert broadcast and system telemetry
  - MJPEG live stream `/api/video_feed` for instant browser video display
  - REST endpoints for system status, recent alerts, and runtime configuration
  - Background async worker processing EdgeAIPipeline in real-time
  - Serves static frontend dashboard
"""

import asyncio
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.app.pipeline import EdgeAIPipeline
from backend.app.utils.video_source import VideoSource
from backend.app.websocket.manager import ConnectionManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("edge_ai.main")

# State & globals
manager = ConnectionManager(max_history=100)
pipeline: Optional[EdgeAIPipeline] = None
latest_jpeg_frame: Optional[bytes] = None
pipeline_running: bool = False
pipeline_stats: Dict[str, float] = {
    "fps": 0.0,
    "latency_ms": 0.0,
    "inference_ms": 0.0,
    "frames_processed": 0,
    "total_alerts": 0,
    "active_persons": 0,
    "active_violations": 0,
}


class PPEConfigRequest(BaseModel):
    required_ppe: List[str]


async def video_pipeline_worker():
    """
    Background worker that runs the EdgeAIPipeline frame-by-frame,
    encodes JPEG frames for the video feed, and broadcasts alerts / metrics.
    """
    global latest_jpeg_frame, pipeline_running, pipeline_stats

    # Choose video source
    video_path = "dataset/raw/real_ppe_test.mp4"
    if not os.path.exists(video_path):
        video_path = "dataset/raw/demo_sample.mp4"

    logger.info("Starting pipeline worker on: %s", video_path)
    source = VideoSource(source=video_path, loop=True, max_fps=25)

    pipeline_running = True
    frame_counter = 0

    while pipeline_running:
        for frame_idx, frame in source.frames():
            if not pipeline_running:
                break

            # Process frame through full edge pipeline
            result = pipeline.process_frame(frame, frame_idx=frame_counter, annotate=True)
            frame_counter += 1

            # Update stats
            pipeline_stats["fps"] = result.metrics["fps"]
            pipeline_stats["latency_ms"] = result.metrics["total_latency_ms"]
            pipeline_stats["inference_ms"] = result.metrics["inference_ms"]
            pipeline_stats["frames_processed"] = frame_counter
            pipeline_stats["active_persons"] = result.metrics["persons_detected"]
            pipeline_stats["active_violations"] = result.metrics["violations_active"]

            # Encode annotated frame as JPEG for MJPEG stream
            if result.annotated_frame is not None:
                ret, buffer = cv2.imencode(".jpg", result.annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ret:
                    latest_jpeg_frame = buffer.tobytes()

            # Broadcast confirmed alerts if any fired
            if result.confirmed_alerts:
                for alert in result.confirmed_alerts:
                    pipeline_stats["total_alerts"] += 1
                    alert_dict = alert.to_dict()
                    alert_dict["camera"] = "CAM_01"
                    await manager.broadcast_alert(alert_dict)
                    logger.info("BROADCAST ALERT: %s (Risk: %d)", alert.severity, alert.risk_score)

            # Broadcast telemetry / metrics at ~5Hz (every 5 frames)
            if frame_counter % 5 == 0:
                await manager.broadcast_metrics({
                    **result.metrics,
                    "total_alerts": pipeline_stats["total_alerts"],
                })

            # Broadcast frame bounding box sync
            hazard_dicts = [h.to_dict() for h in result.hazards]
            await manager.broadcast_frame_data({
                "frame_idx": frame_counter,
                "timestamp": result.timestamp,
                "hazards": hazard_dicts,
                "metrics": result.metrics,
            })

            # Cooperate with asyncio event loop
            await asyncio.sleep(0.001)

        await asyncio.sleep(0.05)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline, pipeline_running
    logger.info("Initializing Edge AI Pipeline...")
    pipeline = EdgeAIPipeline(
        model_path="backend/models/yolov8n_ppe.onnx",
        conf_threshold=0.35,
        window_size=5,
        confirm_threshold=4,
    )
    # Start video processing in background task
    task = asyncio.create_task(video_pipeline_worker())
    yield
    # Cleanup on shutdown
    logger.info("Shutting down pipeline worker...")
    pipeline_running = False
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


app = FastAPI(
    title="Edge AI PPE Safety & Hazard Triage Backend",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Real-time WebSocket endpoint for alerts and telemetry."""
    await manager.connect(websocket)
    try:
        while True:
            # Keepalive / incoming ping handler
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        await manager.disconnect(websocket)
    except Exception as e:
        logger.warning("WebSocket exception: %s", e)
        await manager.disconnect(websocket)


def generate_mjpeg_frames():
    """Generator yielding multipart JPEG frames for video feed."""
    global latest_jpeg_frame
    while True:
        if latest_jpeg_frame is not None:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + latest_jpeg_frame + b"\r\n"
            )
        time.sleep(0.035)  # ~28 FPS stream limiter


@app.get("/api/video_feed")
async def video_feed():
    """MJPEG live camera stream endpoint."""
    return StreamingResponse(
        generate_mjpeg_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/status")
async def get_status():
    """System health, pipeline statistics, and performance numbers."""
    return JSONResponse({
        "status": "ONLINE" if pipeline_running else "INITIALIZING",
        "pipeline": pipeline_stats,
        "active_clients": len(manager.active_connections),
        "model": "yolov8n_ppe.onnx",
        "hardware": "Edge CPU (ONNXRuntime)",
    })


@app.get("/api/alerts/recent")
async def get_recent_alerts():
    """Returns recent confirmed alerts for initial dashboard load."""
    return JSONResponse({
        "total": len(manager.alert_history),
        "alerts": manager.alert_history[-30:],
    })


@app.post("/api/config/ppe")
async def update_required_ppe(config: PPEConfigRequest):
    """Updates required PPE set at runtime."""
    global pipeline
    if pipeline:
        pipeline.compliance_engine.required_ppe = set(config.required_ppe)
        logger.info("Updated required PPE set to: %s", config.required_ppe)
        return {"status": "ok", "required_ppe": list(pipeline.compliance_engine.required_ppe)}
    return JSONResponse(status_code=500, content={"error": "Pipeline not initialized"})


# Mount frontend static files
frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "frontend"))
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")

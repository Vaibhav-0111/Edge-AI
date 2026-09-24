"""
WebSocket Connection Manager — per ARCHITECTURE.md §2.7.

Handles client connections, disconnections, keepalives, and thread-safe
JSON broadcasting for alerts and real-time telemetry.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Set
from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger("edge_ai.websocket")


class ConnectionManager:
    """Manages active WebSocket connections to dashboard clients."""

    def __init__(self, max_history: int = 50):
        self.active_connections: Set[WebSocket] = set()
        self.alert_history: List[Dict[str, Any]] = []
        self.max_history = max_history
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        """Accepts and registers a new WebSocket connection."""
        await websocket.accept()
        async with self._lock:
            self.active_connections.add(websocket)
        logger.info("Client connected. Total active clients: %d", len(self.active_connections))

        # Send initial recent alert history on connect so dashboard isn't blank
        if self.alert_history:
            try:
                await websocket.send_json({
                    "type": "history",
                    "alerts": self.alert_history[-20:],
                })
            except Exception as e:
                logger.warning("Failed to send initial history to client: %s", e)

    async def disconnect(self, websocket: WebSocket) -> None:
        """Removes a disconnected client."""
        async with self._lock:
            self.active_connections.discard(websocket)
        logger.info("Client disconnected. Remaining clients: %d", len(self.active_connections))

    async def broadcast_alert(self, alert_data: Dict[str, Any]) -> None:
        """Broadcasts a confirmed violation alert to all connected dashboards."""
        msg = {
            "type": "alert",
            "camera": alert_data.get("camera", "CAM_01"),
            "person_id": alert_data.get("track_id"),
            "person_bbox": alert_data.get("person_bbox"),
            "helmet": alert_data.get("helmet"),
            "vest": alert_data.get("vest"),
            "gloves": alert_data.get("gloves"),
            "risk_score": alert_data.get("risk_score"),
            "severity": alert_data.get("severity"),
            "reasons": alert_data.get("reasons"),
            "compliance_pct": alert_data.get("compliance_pct"),
            "confirmed_ratio": alert_data.get("confirmed_ratio"),
            "timestamp": alert_data.get("timestamp"),
        }

        # Save to rolling history
        self.alert_history.append(msg)
        if len(self.alert_history) > self.max_history:
            self.alert_history.pop(0)

        await self._broadcast_json(msg)

    async def broadcast_metrics(self, metrics: Dict[str, Any]) -> None:
        """Broadcasts lightweight real-time system and inference metrics."""
        msg = {
            "type": "metrics",
            "fps": metrics.get("fps", 0.0),
            "latency_ms": metrics.get("total_latency_ms", 0.0),
            "preprocess_ms": metrics.get("preprocess_ms", 0.0),
            "inference_ms": metrics.get("inference_ms", 0.0),
            "postprocess_ms": metrics.get("postprocess_ms", 0.0),
            "persons_detected": metrics.get("persons_detected", 0),
            "violations_active": metrics.get("violations_active", 0),
            "cpu_pct": metrics.get("cpu_pct", 0.0),
        }
        await self._broadcast_json(msg)

    async def broadcast_frame_data(self, frame_data: Dict[str, Any]) -> None:
        """Broadcasts per-frame detection boxes and state for canvas overlays."""
        msg = {
            "type": "frame_sync",
            "frame_idx": frame_data.get("frame_idx"),
            "timestamp": frame_data.get("timestamp"),
            "hazards": frame_data.get("hazards", []),
            "metrics": frame_data.get("metrics", {}),
        }
        await self._broadcast_json(msg)

    async def _broadcast_json(self, payload: Dict[str, Any]) -> None:
        """Internal helper to broadcast JSON payload safely to all clients."""
        if not self.active_connections:
            return

        dead_connections = []
        for connection in list(self.active_connections):
            try:
                await connection.send_json(payload)
            except Exception:
                dead_connections.append(connection)

        if dead_connections:
            async with self._lock:
                for dead in dead_connections:
                    self.active_connections.discard(dead)

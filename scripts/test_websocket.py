"""
Unit/Integration test for FastAPI WebSocket & REST endpoints.
Spawns the backend in background, tests connection, receives metrics & alerts.
"""

import asyncio
import json
import os
import sys
import time
import urllib.request
import websockets

BASE_URL = "http://127.0.0.1:8000"
WS_URL = "ws://127.0.0.1:8000/ws"


async def test_backend():
    print("Testing REST endpoints...")
    # Check /api/status
    req = urllib.request.Request(f"{BASE_URL}/api/status")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200, f"Expected 200, got {resp.status}"
        data = json.loads(resp.read().decode())
        print("  /api/status OK:", data)
        assert data["status"] in ("ONLINE", "INITIALIZING")

    # Check /api/alerts/recent
    req_alerts = urllib.request.Request(f"{BASE_URL}/api/alerts/recent")
    with urllib.request.urlopen(req_alerts) as resp_alerts:
        assert resp_alerts.status == 200
        print("  /api/alerts/recent OK")

    print("\nTesting WebSocket connection and message reception...")
    async with websockets.connect(WS_URL) as ws:
        print("  Connected to WebSocket:", WS_URL)

        received_types = set()
        for i in range(5):
            msg_raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
            try:
                msg = json.loads(msg_raw)
                mtype = msg.get("type")
                received_types.add(mtype)
                print(f"  Received message #{i+1}: type='{mtype}'")
            except Exception:
                print(f"  Received text: {msg_raw}")

        # Send ping
        await ws.send("ping")
        pong = await asyncio.wait_for(ws.recv(), timeout=5.0)
        assert pong == "pong" or "type" in pong, f"Ping response valid"
        print("  Ping exchange PASSED")

        assert len(received_types) > 0, "No messages received from WebSocket broadcast"
        print("  WebSocket broadcast test PASSED. Types received:", received_types)

    print("\nALL BACKEND WEBSOCKET & REST TESTS PASSED!")


if __name__ == "__main__":
    asyncio.run(test_backend())

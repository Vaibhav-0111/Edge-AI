# =======================================================
# Edge AI PPE Safety & Hazard Triage System — Dockerfile
# Lean, production-ready container for Render & cloud deployment.
# Includes pre-exported ONNX model (no heavy PyTorch needed).
# Memory footprint: ~70MB (Runs smoothly on Render 512MB tier).
# =======================================================

FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PORT=8000

# Install OS libraries required for OpenCV & healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install lightweight dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code, static assets, and pre-exported model
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY dataset/ ./dataset/

# Verify that the ONNX model is present
RUN python -c "import os; p='backend/models/yolov8n_ppe.onnx'; assert os.path.exists(p), f'Missing model: {p}'; print(f'Verified model present: {os.path.getsize(p)/1024/1024:.1f} MB')"

# Expose default port
EXPOSE 8000

# Start server directly via Python entrypoint (dynamically binds to Render's $PORT)
CMD ["python", "-m", "backend.app.main"]

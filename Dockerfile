# ────────────────────────────────────────────────
# SepsisPilot — Dockerfile
# HF Spaces Docker deployment (port 7860)
# ────────────────────────────────────────────────

FROM python:3.11-slim

# Metadata labels
LABEL org.opencontainers.image.title="SepsisPilot"
LABEL org.opencontainers.image.description="OpenEnv RL environment for sepsis treatment sequencing"
LABEL org.opencontainers.image.version="1.0.0"

# Security: run as non-root user
RUN groupadd -r sepsispilot && useradd -r -g sepsispilot sepsispilot

WORKDIR /app

# Install system deps first (cached layer)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY environment/ ./environment/
COPY app.py .
COPY inference.py .
COPY openenv.yaml .
COPY README.md .

# Ownership
RUN chown -R sepsispilot:sepsispilot /app

USER sepsispilot

# HF Spaces requires port 7860
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Start FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]

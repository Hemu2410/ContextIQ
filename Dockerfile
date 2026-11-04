# =============================================================================
# ContextIQ — Multi-Stage Production Dockerfile
# =============================================================================
# Optimized for AWS ECS / Fargate deployment with minimal image size.
# Runs the FastAPI REST API via Uvicorn on port 8000.
# =============================================================================

FROM python:3.12-slim AS base

# Prevent Python from writing .pyc files and enable unbuffered stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# ---------------------------------------------------------------------------
# Stage 1: Install dependencies
# ---------------------------------------------------------------------------
FROM base AS dependencies

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# Stage 2: Production image
# ---------------------------------------------------------------------------
FROM dependencies AS production

# Copy application source code
COPY contextiq/ ./contextiq/
COPY starter_kit/ ./starter_kit/
COPY api.py .
COPY main.py .

# Create non-root user for security
RUN adduser --disabled-password --no-create-home appuser
USER appuser

# Expose API port
EXPOSE 8000

# Health check for ECS / container orchestrators
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Launch Uvicorn (binds to $PORT if set by cloud host like Render/Cloud Run, defaults to 8000)
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000} --workers 2"]

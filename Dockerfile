# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Stage 1 — export the exact pinned dependencies from uv.lock
# (keeps uv.lock the single source of truth; no hand-written requirements.txt)
# ---------------------------------------------------------------------------
FROM python:3.13-slim AS deps

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /src
COPY pyproject.toml uv.lock ./
RUN uv export --frozen --no-dev --no-hashes -o requirements.txt

# ---------------------------------------------------------------------------
# Stage 2 — runtime image
# ---------------------------------------------------------------------------
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Runtime system libraries:
#   - libgl1/glib/sm/xext/xrender : OpenCV (opencv-python needs libGL.so.1 etc.)
#   - libgomp1                    : OpenMP backend for torch / OpenCV
#   - tzdata                      : zoneinfo for APScheduler (Asia/Jakarta)
#   - curl                        : healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        libsm6 \
        libxext6 \
        libxrender1 \
        tzdata \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=deps /src/requirements.txt ./

# Install torch + torchvision from the CPU-only index: PyPI's Linux wheels
# bundle the CUDA runtime (~2 GB+); the PyTorch CPU index ships the same
# versions without it. Installing the exact pinned versions first means the
# `pip install -r` below sees them already satisfied and skips them. The
# uv.lock also resolves CUDA companion packages (cuda-*/nvidia-*/triton) for
# Linux — those are only needed by GPU torch, so they are dropped here
# (nvidia-ml-py, a tiny pure-python ultralytics dep, is kept).
RUN TORCH_VERSION="$(grep -iE '^torch==' requirements.txt | head -1 | tr -d '\r')" \
    && TORCHVISION_VERSION="$(grep -iE '^torchvision==' requirements.txt | head -1 | tr -d '\r')" \
    && test -n "$TORCH_VERSION" && test -n "$TORCHVISION_VERSION" \
    && pip install --index-url https://download.pytorch.org/whl/cpu \
           --extra-index-url https://pypi.org/simple \
           "$TORCH_VERSION" "$TORCHVISION_VERSION" \
    && awk '!/^(cuda-|triton|nvidia-)/ || /^nvidia-ml-py/' requirements.txt > /tmp/requirements-linux.txt \
    && pip install -r /tmp/requirements-linux.txt \
    && rm -f requirements.txt /tmp/requirements-linux.txt

# Application code + data
COPY app ./app
COPY data/cctvs.json ./data/cctvs.json
COPY alembic.ini ./
COPY alembic ./alembic

# Trained flood model -> exactly the path FLOOD_MODEL_PATH resolves to inside
# the container (config.py resolves relative paths against the project root).
COPY runs/detect/flood_yolo11s_run/best.pt ./flood_yolo11s_run/best.pt

# Non-root user. /app stays writable so the daily scraper can rewrite data/cctvs.json.
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/ || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

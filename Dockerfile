# syntax=docker/dockerfile:1

# Ships uv + Python 3.13 together — one image for the whole toolchain.
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_NO_CACHE=1

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

# Pin dependencies straight from uv.lock (single source of truth).
COPY pyproject.toml uv.lock ./
RUN uv export --frozen --no-dev --no-hashes -o requirements.txt

# torch + torchvision from the CPU-only index first: PyPI's Linux wheels bundle
# the CUDA runtime (~2 GB+); the PyTorch CPU index ships the same versions
# without it. Installing the exact pinned versions first means the full install
# below sees them already satisfied and skips them. The lock also resolves CUDA
# companion packages (cuda-*/nvidia-*/triton) for Linux — dropped here, they're
# only needed by GPU torch (nvidia-ml-py, a pure-python ultralytics dep, kept).
RUN TORCH_VERSION="$(grep -iE '^torch==' requirements.txt | head -1)" \
    && TORCHVISION_VERSION="$(grep -iE '^torchvision==' requirements.txt | head -1)" \
    && test -n "$TORCH_VERSION" && test -n "$TORCHVISION_VERSION" \
    && uv pip install --index-url https://download.pytorch.org/whl/cpu \
           --extra-index-url https://pypi.org/simple \
           "$TORCH_VERSION" "$TORCHVISION_VERSION" \
    && awk '!/^(cuda-|triton|nvidia-)/ || /^nvidia-ml-py/' requirements.txt > /tmp/requirements-linux.txt \
    && uv pip install -r /tmp/requirements-linux.txt \
    && rm -f requirements.txt /tmp/requirements-linux.txt

# Application code + data + env (config.py loads .env via python-dotenv).
COPY app ./app
COPY data/cctvs.json ./data/cctvs.json
COPY alembic.ini ./
COPY alembic ./alembic
COPY .env ./.env

# Trained anomaly model -> exactly the path ANOMALY_MODEL_PATH resolves to.
FREECOPY models/best.pt ./models/best.pt

# Non-root user. /app stays writable so the daily scraper can rewrite data/cctvs.json.
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 9002

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://127.0.0.1:9002/api/health || exit 1

# python -m uvicorn (not the bare console script): the uv image's python is
# uv-managed, so its bin dir may not be on PATH.
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9002"]

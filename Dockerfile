# syntax=docker/dockerfile:1

# Multi-stage build. Dependencies are installed into a virtualenv in a builder
# stage and copied into a clean runtime stage, so pip, its cache and the build
# toolchain never reach the final image.
#
#   docker build --target cpu -t sentiment-analysis-api:local .
#   docker run --rm -p 8000:8000 sentiment-analysis-api:local

FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app


FROM base AS builder

COPY requirements/ requirements/

# The default PyPI torch wheel bundles CUDA and costs several gigabytes. The
# CPU index gives the same API in a fraction of the size, which matters for an
# image whose whole job is to answer HTTP requests on a CPU node.
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements/inference-cpu.txt


FROM base AS cpu

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY src/ src/
COPY api/ api/
COPY app/ app/
# Weights ship in the image so `docker run` is self-contained. They are the
# largest thing in it by far; for a real deployment they belong in object
# storage or on the Hugging Face Hub, pulled at start-up. docker-compose mounts
# ./models over this directory so local edits do not require a rebuild.
COPY models/ models/

# Run as a non-root user. Nothing here needs to write to the filesystem.
RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Liveness only: /health answers even when no model artifact is usable, which is
# the distinction the API draws deliberately. Use /ready to gate traffic.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

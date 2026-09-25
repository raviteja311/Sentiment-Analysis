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
# image whose whole job is to answer HTTP requests on a CPU node. The lock
# pins the transitive closure too, so two builds a month apart get the same
# starlette, not whichever one PyPI had that day.
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements/inference-cpu.lock


FROM base AS cpu

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# --chown at copy time: a later `chown -R` rewrites every file into a new
# layer, which with weights baked in stores them twice.
RUN useradd --create-home --uid 1000 appuser

COPY --chown=appuser:appuser src/ src/
COPY --chown=appuser:appuser api/ api/
COPY --chown=appuser:appuser app/ app/

# Only the small artifacts come from the build context - the LR pipeline, the
# tokenizers and the transformer's config and vocabulary. The large weights are
# excluded by .dockerignore and fetched from object storage instead.
COPY --chown=appuser:appuser models/ models/

# Which models' weights to bake into the image. Empty by default, which keeps
# the image small and leaves it serving Logistic Regression only; the rest can
# be mounted at run time (see docker-compose.yml) or baked in with:
#   docker build --target cpu --build-arg FETCH_MODELS="lstm gru roberta" .
ARG FETCH_MODELS=""

# Fetch as the runtime user, so the downloads need no ownership fix afterwards.
USER appuser
RUN if [ -n "$FETCH_MODELS" ]; then \
        python -m src.artifacts --models $FETCH_MODELS; \
    fi

EXPOSE 8000

# Liveness only: /health answers even when no model artifact is usable, which is
# the distinction the API draws deliberately. Use /ready to gate traffic.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

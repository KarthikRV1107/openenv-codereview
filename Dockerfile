# IncidentOps OpenEnv — Dockerfile
# Uses the official openenv-base image with plain pip for reliability.

ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE}

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Copy source first so pip can install from it
COPY . /app/env

WORKDIR /app/env

# Install Python dependencies with pip (no uv, no lock file needed)
RUN pip install --no-cache-dir \
    "fastapi>=0.110.0" \
    "uvicorn[standard]>=0.29.0" \
    "pydantic>=2.7.0" \
    "openai>=1.30.0" \
    "openenv-core>=0.1.0"

# Make the project importable
ENV PYTHONPATH="/app/env:${PYTHONPATH}"

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

EXPOSE 8000

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]

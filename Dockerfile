# infrastructure/docker/Dockerfile.api
# Multi-stage build — CPU-only PyTorch (~1.5 GB vs 5 GB GPU image)

FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git \
    && rm -rf /var/lib/apt/lists/*

# CPU-only PyTorch first (prevents pulling the full CUDA variant)
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

# Install production dependencies from requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .


FROM python:3.11-slim AS runtime

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /build /app

# Non-root user for security
RUN groupadd -r voltiq && useradd -r -g voltiq voltiq
RUN chown -R voltiq:voltiq /app
USER voltiq

RUN mkdir -p /app/data/raw /app/data/processed /app/data/artifacts

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
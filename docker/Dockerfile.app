FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for Gymnasium and kubectl
RUN apt-get update && apt-get install -y \
    libosmesa6-dev \
    freeglut3-dev \
    mesa-common-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LO "https://dl.k8s.io/release/v1.28.0/bin/linux/$(dpkg --print-architecture)/kubectl" && \
    chmod +x kubectl && \
    mv kubectl /usr/local/bin/

# --extra-index-url, not --index-url: the latter *replaces* PyPI, so anything
# pip needs that the PyTorch index does not serve becomes unresolvable. That
# started failing the build with "No matching distribution found for
# flit_core<4,>=3.11" -- a build dependency pulled in during resolution, which
# lives on PyPI and not on download.pytorch.org. With --extra-index-url the
# CPU wheel index is consulted in addition to PyPI rather than instead of it,
# which keeps the original intent (CPU-only torch, no CUDA payload) while
# leaving ordinary build dependencies resolvable.
RUN pip install --no-cache-dir torch --extra-index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir gymnasium numpy kfp==2.15.2 mlflow-skinny boto3 kubernetes==30.1.0 "temporalio>=1.7.0" "minio>=7.2.0"

COPY src/core/engine.py ./
COPY src/core/twin.py ./
COPY src/core/tracking.py ./
COPY src/core/worker_entrypoint.py ./
COPY src/core/aggregate.py ./
COPY src/core/metrics_csv.py ./
COPY src/orchestration/ ./src/orchestration/

# Entrypoint can be overridden by the Pipeline/Job command
ENTRYPOINT ["python"]

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

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir flwr gymnasium numpy kfp==2.15.2 mlflow-skinny boto3

COPY src/core/engine.py ./
COPY src/core/client.py ./
COPY src/core/server.py ./
COPY src/core/tracking.py ./

# Entrypoint can be overridden by PyTorchJob/Pipeline
ENTRYPOINT ["python"]

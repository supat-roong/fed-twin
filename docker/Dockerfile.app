FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for Gymnasium
RUN apt-get update && apt-get install -y \
    libosmesa6-dev \
    freeglut3-dev \
    mesa-common-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir flwr gymnasium numpy kfp mlflow-skinny boto3

COPY src/core/engine.py ./
COPY src/core/client.py ./
COPY src/core/server.py ./
COPY src/core/tracking.py ./

# Entrypoint can be overridden by PyTorchJob/Pipeline
ENTRYPOINT ["python"]

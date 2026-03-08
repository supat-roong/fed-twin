#!/usr/bin/env bash
# install_local.sh — Bootstraps the full Federated Digital Twin stack on a local kind cluster
# Prerequisites: kind, kubectl, docker

set -euo pipefail

CLUSTER_NAME="fed-twin-cluster"
IMAGE_NAME="fed-twin-app:v1"

echo "=================================================="
echo "  Federated Digital Twin Local Setup"
echo "=================================================="

# 0. Virtual Environment Setup
echo "🐍 Syncing uv environment..."
uv sync

# 1. Check/Create Cluster
if kind get clusters | grep -q "$CLUSTER_NAME"; then
    echo "✅ Cluster '$CLUSTER_NAME' already exists."
else
    echo "📦 Creating Kind cluster '$CLUSTER_NAME'..."
    SCRIPT_DIR=$(dirname "$0")
    kind create cluster --config "${SCRIPT_DIR}/kind-cluster.yaml"
fi

# 2. Install Kubeflow Training Operator (if not present)
echo "🛠 Checking Kubeflow Training Operator..."
if ! kubectl get crd pytorchjobs.kubeflow.org > /dev/null 2>&1; then
    echo "   Installing Training Operator..."
    kubectl apply -k "github.com/kubeflow/training-operator/manifests/overlays/standalone?ref=v1.7.0"
    echo "   Waiting for Training Operator..."
    sleep 5
    kubectl wait --for=condition=ready pod -l control-plane=kubeflow-training-operator -n kubeflow --timeout=120s
else
    echo "✅ Training Operator already installed."
fi

# 3. Install KFP (if not present)
echo "🛠 Checking Kubeflow Pipelines..."
if ! kubectl get deploy -n kubeflow ml-pipeline > /dev/null 2>&1; then
    echo "   Installing KFP CRDs..."
    kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=2.4.0&timeout=300s"
    kubectl wait --for condition=established --timeout=300s crd/applications.app.k8s.io || true
    
    echo "   Installing KFP Core..."
    kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic?ref=2.4.0&timeout=300s"
else
    echo "✅ KFP already installed."
fi

# 4. Build & Load Image
echo "🐳 Building Docker Image..."
docker build -t "$IMAGE_NAME" -f docker/Dockerfile.app .

# Get the Image ID to check if it's already in Kind
LOCAL_IMAGE_ID=$(docker image inspect "$IMAGE_NAME" --format '{{.Id}}' | cut -d':' -f2 | cut -c1-12)
KIND_IMAGE_ID=$(docker exec "$CLUSTER_NAME-control-plane" crictl images 2>/dev/null | grep "${IMAGE_NAME%:*}" | grep "${IMAGE_NAME#*:}" | awk '{print $3}' | head -n 1 || true)

if [ "$LOCAL_IMAGE_ID" == "${KIND_IMAGE_ID:0:12}" ] && [ -n "$KIND_IMAGE_ID" ]; then
    echo "✅ Image '$IMAGE_NAME' already exists in Kind and matches local version ($LOCAL_IMAGE_ID)."
else
    echo "🚚 Loading Image into Cluster ($LOCAL_IMAGE_ID)..."
    kind load docker-image "$IMAGE_NAME" --name "$CLUSTER_NAME"
fi

# 5. Permissions & Image Fix
echo "🔑 Configuring Permissions..."
# Ensure namespace exists
kubectl create namespace kubeflow --dry-run=client -o yaml | kubectl apply -f -
kubectl create clusterrolebinding pipeline-runner-extend --clusterrole=cluster-admin --serviceaccount=kubeflow:default --dry-run=client -o yaml | kubectl apply -f -

echo "🐳 Container Registry Migration Fix..."
for img in "ghcr.io/kubeflow/kfp-frontend:2.4.0" "ghcr.io/kubeflow/kfp-api-server:2.4.0" "ghcr.io/kubeflow/kfp-visualization-server:2.4.0" "ghcr.io/kubeflow/kfp-launcher:2.4.0"; do
    if ! docker exec "$CLUSTER_NAME-control-plane" crictl images | grep -q "${img%:*}"; then
        echo "   Pulling & Loading $img..."
        docker pull "$img"
        kind load docker-image "$img" --name "$CLUSTER_NAME"
    else
        echo "   ✅ $img is already in cluster."
    fi
done

# Update deployments
echo "🛠 Updating KFP Deployments to use GHCR images..."
kubectl set image deployment/ml-pipeline-ui ml-pipeline-ui=ghcr.io/kubeflow/kfp-frontend:2.4.0 -n kubeflow
kubectl set image deployment/ml-pipeline ml-pipeline-api-server=ghcr.io/kubeflow/kfp-api-server:2.4.0 -n kubeflow
kubectl set image deployment/ml-pipeline-visualizationserver ml-pipeline-visualizationserver=ghcr.io/kubeflow/kfp-visualization-server:2.4.0 -n kubeflow

# Fix Launcher Image (M1/Kind Stability)
kubectl set env deployment/ml-pipeline V2_LAUNCHER_IMAGE=ghcr.io/kubeflow/kfp-launcher:2.4.0 -n kubeflow

# Fix Minio Image for KFP (Kind/M1 issue)
echo "🔧 Fixing Minio Image..."
kubectl set image deployment/minio minio=minio/minio:latest -n kubeflow
echo "🔧 Patching Minio Deployment to expose Console..."
kubectl patch deployment minio -n kubeflow --type=json -p='[{"op": "add", "path": "/spec/template/spec/containers/0/ports/-", "value": {"containerPort": 9001}}]'
kubectl patch deployment minio -n kubeflow --type=json -p='[{"op": "replace", "path": "/spec/template/spec/containers/0/args", "value": ["server", "/data", "--console-address", ":9001"]}]'

# Fix Workflow Controller Executor Image (Init Container Issue)
echo "🔧 Patching Workflow Controller Executor Image..."
kubectl patch deployment workflow-controller -n kubeflow --type=json -p='[{"op": "replace", "path": "/spec/template/spec/containers/0/args/3", "value": "quay.io/argoproj/argoexec:v3.4.17"}]'

echo "⏳ Waiting for Kubeflow Deployments to become ready (this may take 5-10 minutes)..."
kubectl rollout status deployment/workflow-controller -n kubeflow --timeout=15m
kubectl rollout status deployment/minio -n kubeflow --timeout=15m
kubectl rollout status deployment/ml-pipeline -n kubeflow --timeout=15m
kubectl rollout status deployment/ml-pipeline-ui -n kubeflow --timeout=15m

# 6. Expose Services via NodePort
echo "🌐 Exposing Services via NodePort..."
# Find ml-pipeline-ui service and patch it
kubectl patch service ml-pipeline-ui -n kubeflow -p '{"spec": {"type": "NodePort", "ports": [{"port": 80, "targetPort": 3000, "nodePort": 30080}]}}' || true
# Find minio service and patch it
kubectl patch service minio-service -n kubeflow --type=json -p='[{"op": "replace", "path": "/spec/type", "value": "NodePort"}, {"op": "replace", "path": "/spec/ports", "value": [{"name": "api", "port": 9000, "protocol": "TCP", "targetPort": 9000, "nodePort": 30900}, {"name": "console", "port": 9001, "protocol": "TCP", "targetPort": 9001, "nodePort": 30901}]}]' || true

echo ""
echo "✅ Setup complete! Services are exposed locally via Docker/Kind port-mapping:"
echo "  → Kubeflow Pipelines UI : http://localhost:8080"
echo "  → MinIO API             : http://localhost:9000"
echo "  → MinIO Console         : http://localhost:9001"

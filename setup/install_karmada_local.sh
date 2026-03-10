#!/usr/bin/env bash
# install_karmada_local.sh — Bootstraps the Federated Digital Twin stack on Karmada multi-cluster
# Prerequisites: kind, kubectl, docker

set -euo pipefail

HOST_CLUSTER="fed-twin-host"
MEMBER_PREFIX="fed-twin-member"
IMAGE_NAME="fed-twin-app:v1"

echo "=================================================="
echo "  Federated Digital Twin Karmada Local Setup"
echo "=================================================="

# Make sure environment has standard paths
source ~/.zshrc 2>/dev/null || true
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"

# 0. Virtual Environment Setup & Config Load
echo "🐍 Syncing uv environment..."
if command -v uv &> /dev/null; then
    uv sync
else
    echo "⚠️  uv command not found, skipping sync."
fi

# Read num_workers from config
CONFIG_FILE="config/config.json"
if [ -f "$CONFIG_FILE" ]; then
    # Simple extraction using grep/awk, assuming {"num_workers": X} format
    NUM_WORKERS=$(grep -E '"num_workers"\s*:' "$CONFIG_FILE" | awk -F: '{print $2}' | tr -d ' ,')
else
    NUM_WORKERS=3
    echo "⚠️ Config file not found, defaulting to $NUM_WORKERS workers."
fi

if [ -z "$NUM_WORKERS" ]; then
    NUM_WORKERS=3
fi

NUM_MEMBER_CLUSTERS=$NUM_WORKERS
echo "⚙️  Configured for $NUM_WORKERS FL Workers -> Creating $NUM_MEMBER_CLUSTERS Member Clusters."

# 1. Check/Create Host Cluster
if kind get clusters | grep -q "^$HOST_CLUSTER$"; then
    echo "✅ Host Cluster '$HOST_CLUSTER' already exists."
else
    echo "📦 Creating Host Kind cluster '$HOST_CLUSTER'..."
    SCRIPT_DIR=$(dirname "$0")
    kind create cluster --name "$HOST_CLUSTER" --config "${SCRIPT_DIR}/kind-karmada-host.yaml"
fi

# 2. Check/Create Member Clusters
for i in $(seq 1 $NUM_MEMBER_CLUSTERS); do
    MEMBER_NAME="${MEMBER_PREFIX}${i}"
    if kind get clusters | grep -x "$MEMBER_NAME" > /dev/null 2>&1; then
        echo "✅ Member Cluster '$MEMBER_NAME' already exists."
    else
        echo "📦 Creating Member Kind cluster '$MEMBER_NAME'..."
        kind create cluster --name "$MEMBER_NAME"
    fi
    # Set inotify limits on member node
    docker exec "${MEMBER_NAME}-control-plane" sysctl -w fs.inotify.max_user_instances=512 fs.inotify.max_user_watches=524288 || true
done

# Set inotify limits on host node
docker exec "${HOST_CLUSTER}-control-plane" sysctl -w fs.inotify.max_user_instances=512 fs.inotify.max_user_watches=524288 || true

# 3. Install Karmadactl (if not present)
if ! command -v karmadactl &> /dev/null; then
    echo "🛠 Installing karmadactl..."
    curl -s --proto '=https' --tlsv1.2 -sSf https://raw.githubusercontent.com/karmada-io/karmada/master/hack/install-cli.sh | bash
fi

# 4. Pre-fetch Karmada Images
echo "🐳 Pre-fetching Karmada core images on host to avoid container networking timeouts..."
for img in \
    "docker.io/karmada/karmada-aggregated-apiserver:v1.17.0" \
    "docker.io/karmada/karmada-controller-manager:v1.17.0" \
    "docker.io/karmada/karmada-scheduler:v1.17.0" \
    "docker.io/karmada/karmada-webhook:v1.17.0"; do
    if ! docker exec "${HOST_CLUSTER}-control-plane" crictl images | grep -q "${img%:*}"; then
        docker pull --platform=linux/arm64 "$img" &>/dev/null || true
        kind load docker-image "$img" --name "$HOST_CLUSTER" &>/dev/null || true
    fi
done

# 5. Initialize Karmada Control Plane on Host
echo "🛠 Initializing Karmada Control Plane on Host Cluster..."
kubectl config use-context "kind-${HOST_CLUSTER}"
if ! kubectl get namespace karmada-system > /dev/null 2>&1; then
    echo "   Creating local Karmada data directories to avoid sudo..."
    mkdir -p "$HOME/.karmada/pki"
    karmadactl init \
        --karmada-data="$HOME/.karmada" \
        --karmada-pki="$HOME/.karmada/pki" \
        --cert-external-ip="127.0.0.1" \
        --cert-external-dns="localhost" \
        --karmada-apiserver-advertise-address="127.0.0.1" \
        --etcd-storage-mode="emptyDir" \
        --kube-image-tag="v1.35.0"
else
    echo "✅ Karmada already initialized."
fi

# 5. Join Clusters (Host and Members)
echo "🔗 Joining Clusters to Karmada..."
KARMADA_API_CONFIG="$HOME/.karmada/karmada-apiserver.config"
HOST_KUBECONFIG="$HOME/.kube/config"

join_and_patch() {
    local cluster_name=$1
    local context=$2
    
    # 1. Join using localhost (so karmadactl on host can reach it)
    if ! kubectl --kubeconfig="$KARMADA_API_CONFIG" get cluster "$cluster_name" > /dev/null 2>&1; then
        echo "   Joining $cluster_name..."
        karmadactl --kubeconfig="$KARMADA_API_CONFIG" \
            join "$cluster_name" \
            --cluster-kubeconfig="$HOST_KUBECONFIG" \
            --cluster-context="$context" || return 1
    else
        echo "   ✅ $cluster_name already joined."
    fi

    # 2. Get internal IP and Port
    local ip=$( (docker inspect "${cluster_name}-control-plane" --format '{{ .NetworkSettings.Networks.kind.IPAddress }}' 2>/dev/null || docker inspect "kind-${cluster_name}-control-plane" --format '{{ .NetworkSettings.Networks.kind.IPAddress }}' 2>/dev/null) | tr -d '\n' )
    if [ -z "$ip" ]; then return 0; fi

    echo "   📍 Patching $cluster_name to internal IP $ip..."
    
    # 3. Patch Cluster object
    kubectl --kubeconfig="$KARMADA_API_CONFIG" patch cluster "$cluster_name" --type=merge -p "{\"spec\":{\"apiEndpoint\":\"https://${ip}:6443\"}}" || true
    
    # 4. Patch the Secret (Karmada stores kubeconfig here)
    local secret_name=$(kubectl --kubeconfig="$KARMADA_API_CONFIG" get cluster "$cluster_name" -o jsonpath='{.spec.secretRef.name}')
    local secret_ns=$(kubectl --kubeconfig="$KARMADA_API_CONFIG" get cluster "$cluster_name" -o jsonpath='{.spec.secretRef.namespace}')
    
    if [ -n "$secret_name" ]; then
        # Use python to safely update the server URL in the base64 encoded secret data
        kubectl --kubeconfig="$KARMADA_API_CONFIG" get secret "$secret_name" -n "$secret_ns" -o jsonpath='{.data.kubeconfig}' | python3 -c "import sys, base64; sys.stdout.buffer.write(base64.b64decode(sys.stdin.read()))" > /tmp/temp_kubeconfig
        python3 -c "import sys; c=open('/tmp/temp_kubeconfig').read(); open('/tmp/temp_kubeconfig','w').write(c.replace('127.0.0.1', '$ip').replace('localhost', '$ip'))"
        local new_kubeconfig=$(python3 -c "import sys, base64; print(base64.b64encode(open('/tmp/temp_kubeconfig', 'rb').read()).decode('utf-8'))")
        kubectl --kubeconfig="$KARMADA_API_CONFIG" patch secret "$secret_name" -n "$secret_ns" -p "{\"data\":{\"kubeconfig\":\"$new_kubeconfig\"}}" || true
    fi
}

# Join Host
join_and_patch "$HOST_CLUSTER" "kind-$HOST_CLUSTER"

# Join Members
for i in $(seq 1 $NUM_MEMBER_CLUSTERS); do
    MEMBER_NAME="${MEMBER_PREFIX}${i}"
    join_and_patch "$MEMBER_NAME" "kind-$MEMBER_NAME"
done

# 6. Build & Load Image to All Clusters (Host + Members)
echo "🐳 Building Docker Images..."
docker build -t "$IMAGE_NAME" -f docker/Dockerfile.app .
docker build -t local-mlflow-boto3:v2.12.2 -f docker/Dockerfile.mlflow .

echo "🚚 Loading Image into Host Cluster..."
kind load docker-image "$IMAGE_NAME" --name "$HOST_CLUSTER"
kind load docker-image local-mlflow-boto3:v2.12.2 --name "$HOST_CLUSTER"

for i in $(seq 1 $NUM_MEMBER_CLUSTERS); do
    MEMBER_NAME="${MEMBER_PREFIX}${i}"
    echo "🚚 Loading Image into Member Cluster '$MEMBER_NAME'..."
    kind load docker-image "$IMAGE_NAME" --name "$MEMBER_NAME"
done

# 7. Install ML/KFP Stack on Host Cluster ONLY
echo "=================================================="
echo "  Deploying KFP & MLflow Services on Host"
echo "=================================================="
kubectl config use-context "kind-${HOST_CLUSTER}"

# Permissions
kubectl create namespace kubeflow --dry-run=client -o yaml | kubectl apply -f -
kubectl create clusterrolebinding pipeline-runner-extend --clusterrole=cluster-admin --serviceaccount=kubeflow:default --dry-run=client -o yaml | kubectl apply -f -

# Install KFP
if ! kubectl get deploy -n kubeflow ml-pipeline > /dev/null 2>&1; then
    echo "   Installing KFP CRDs..."
    kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=2.4.0&timeout=300s"
    kubectl wait --for condition=established --timeout=300s crd/applications.app.k8s.io || true
    
    echo "   Installing KFP Core..."
    kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic?ref=2.4.0&timeout=300s"
else
    echo "✅ KFP already installed on Host."
fi

# GHCR Migration
echo "🐳 Container Registry Migration Fix..."
for img in "ghcr.io/kubeflow/kfp-frontend:2.4.0" "ghcr.io/kubeflow/kfp-api-server:2.4.0" "ghcr.io/kubeflow/kfp-visualization-server:2.4.0" "ghcr.io/kubeflow/kfp-launcher:2.4.0"; do
    if ! docker exec "${HOST_CLUSTER}-control-plane" crictl images | grep -q "${img%:*}"; then
        docker pull "$img"
        kind load docker-image "$img" --name "$HOST_CLUSTER"
    fi
done

kubectl set image deployment/ml-pipeline-ui ml-pipeline-ui=ghcr.io/kubeflow/kfp-frontend:2.4.0 -n kubeflow
kubectl set image deployment/ml-pipeline ml-pipeline-api-server=ghcr.io/kubeflow/kfp-api-server:2.4.0 -n kubeflow
kubectl set image deployment/ml-pipeline-visualizationserver ml-pipeline-visualizationserver=ghcr.io/kubeflow/kfp-visualization-server:2.4.0 -n kubeflow
kubectl set env deployment/ml-pipeline V2_LAUNCHER_IMAGE=ghcr.io/kubeflow/kfp-launcher:2.4.0 -n kubeflow

# Minio Fix
kubectl set image deployment/minio minio=minio/minio:latest -n kubeflow
kubectl patch deployment minio -n kubeflow --type=json -p='[{"op": "add", "path": "/spec/template/spec/containers/0/ports/-", "value": {"containerPort": 9001}}]' || true
kubectl patch deployment minio -n kubeflow --type=json -p='[{"op": "replace", "path": "/spec/template/spec/containers/0/args", "value": ["server", "/data", "--console-address", ":9001"]}]' || true

# Workflow Controller Fix
kubectl patch deployment workflow-controller -n kubeflow --type=json -p='[{"op": "replace", "path": "/spec/template/spec/containers/0/args/3", "value": "quay.io/argoproj/argoexec:v3.4.17"}]' || true

# MLflow
kubectl apply -f k8s/mlflow-server.yaml
kubectl run minio-setup-mlflow --image=minio/mc:latest --namespace=kubeflow --restart=Never \
  --command -- sh -c "mc alias set minio http://minio-service.kubeflow.svc.cluster.local:9000 minio minio123 && mc mb minio/mlflow-artifacts --ignore-existing" 2>/dev/null || true
kubectl wait --for=condition=completed pod/minio-setup-mlflow -n kubeflow --timeout=60s 2>/dev/null || true
kubectl delete pod minio-setup-mlflow -n kubeflow 2>/dev/null || true

# NodePorts
kubectl patch service ml-pipeline-ui -n kubeflow -p '{"spec": {"type": "NodePort", "ports": [{"port": 80, "targetPort": 3000, "nodePort": 30080}]}}' || true
kubectl patch service minio-service -n kubeflow --type=json -p='[{"op": "replace", "path": "/spec/type", "value": "NodePort"}, {"op": "replace", "path": "/spec/ports", "value": [{"name": "api", "port": 9000, "protocol": "TCP", "targetPort": 9000, "nodePort": 30900}, {"name": "console", "port": 9001, "protocol": "TCP", "targetPort": 9001, "nodePort": 30901}]}]' || true
kubectl patch service mlflow-service -n kubeflow -p '{"spec": {"type": "NodePort", "ports": [{"port": 5000, "targetPort": 5000, "nodePort": 30500}]}}' || true

echo "⏳ Waiting for Host Services (this takes a while)..."
kubectl rollout status deployment/workflow-controller -n kubeflow --timeout=15m
kubectl rollout status deployment/ml-pipeline -n kubeflow --timeout=15m
kubectl rollout status deployment/ml-pipeline-ui -n kubeflow --timeout=15m

echo "✅ Karmada Setup complete! Active context is still the Karmada API or Host."
echo "Use 'kubectl --context=karmada-apiserver' to interact with the Federation control plane."

# 8. Install Karmada Dashboard
echo "=================================================="
echo "  Deploying Karmada Dashboard"
echo "=================================================="

# Switch back to the Karmada host to install the dashboard components
kubectl config use-context "kind-${HOST_CLUSTER}"

echo "   Applying Karmada Dashboard manifests..."
# We will use the official manifest for the dashboard
kubectl apply -f https://raw.githubusercontent.com/karmada-io/dashboard/main/deploy/karmada-dashboard.yaml

echo "   Deploying Secret for Karmada API configuration..."
# The dashboard needs the kubeconfig of Karmada API Server to communicate with it
# We grab it from ~/.karmada/karmada-apiserver.config and create a secret in karmada-system namespace
kubectl create secret generic karmada-kubeconfig --from-file=karmada-kubeconfig="$HOME/.karmada/karmada-apiserver.config" -n karmada-system --dry-run=client -o yaml | kubectl apply -f -
kubectl create secret generic karmada-kubeconfig-kf --from-file=karmada-kubeconfig="$HOME/.karmada/karmada-apiserver.config" -n kubeflow --dry-run=client -o yaml | kubectl apply -f -

echo "   Exposing Karmada Dashboard via NodePort 32000..."
# Use apply for a more robust definition
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Service
metadata:
  labels:
    app: frontend
  name: karmada-dashboard
  namespace: karmada-system
spec:
  ports:
  - name: http
    nodePort: 32000
    port: 80
    protocol: TCP
    targetPort: 80
  selector:
    app: frontend
  type: NodePort
EOF

echo "⏳ Waiting for Karmada Dashboard..."
kubectl rollout status deployment/karmada-dashboard -n karmada-system --timeout=5m || true

echo "✅ Karmada Dashboard is available at: http://localhost:32000"

# 9. Create Admin Token for Dashboard in the Karmada Federation
echo "=================================================="
echo "  Generating Karmada Dashboard Access Token"
echo "=================================================="
# We create the SA and binding IN THE FEDERATION context
kubectl --kubeconfig="$KARMADA_API_CONFIG" create serviceaccount karmada-admin-sa -n karmada-system --dry-run=client -o yaml | kubectl --kubeconfig="$KARMADA_API_CONFIG" apply -f -
kubectl --kubeconfig="$KARMADA_API_CONFIG" create clusterrolebinding karmada-admin-sa-binding --clusterrole=cluster-admin --serviceaccount=karmada-system:karmada-admin-sa --dry-run=client -o yaml | kubectl --kubeconfig="$KARMADA_API_CONFIG" apply -f -

DASHBOARD_TOKEN=$(kubectl --kubeconfig="$KARMADA_API_CONFIG" create token karmada-admin-sa -n karmada-system --duration=24h)
echo "🚀 Dashboard Access Token (expires in 24h):"
echo "--------------------------------------------------"
echo "$DASHBOARD_TOKEN"
echo "--------------------------------------------------"
echo ""
echo "📝 Note: Use the token above to log into the dashboard at http://localhost:32000"

#!/bin/bash
set -e # Exit on error

CLUSTER_NAME="fed-twin-cluster"
IMAGE_NAME="fed-twin-app:v1"

echo "🚀 Starting Federated Digital Twin Project Setup..."

# 0. Virtual Environment Setup
if [ -d "venv" ]; then
    echo "🐍 Activating Virtual Environment..."
    source venv/bin/activate
else
    echo "⚠️ venv not found. Creating and installing requirements..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
fi

# 1. Check/Create Cluster
if kind get clusters | grep -q "$CLUSTER_NAME"; then
    echo "✅ Cluster '$CLUSTER_NAME' already exists."
else
    echo "📦 Creating Kind cluster '$CLUSTER_NAME'..."
    kind create cluster --name "$CLUSTER_NAME"
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
KIND_IMAGE_ID=$(docker exec "$CLUSTER_NAME-control-plane" crictl images | grep "${IMAGE_NAME%:*}" | grep "${IMAGE_NAME#*:}" | awk '{print $3}' | head -n 1)

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

# Fix Workflow Controller Executor Image (Init Container Issue)
echo "🔧 Patching Workflow Controller Executor Image..."
kubectl patch deployment workflow-controller -n kubeflow --type=json -p='[{"op": "replace", "path": "/spec/template/spec/containers/0/args/3", "value": "quay.io/argoproj/argoexec:v3.4.17"}]'
kubectl rollout status deployment/workflow-controller -n kubeflow

# 6. Pipeline Execution Logic
PIPELINE_ARG=${1:-"all"}

run_pipeline() {
    local PIPELINE_TYPE=$1
    echo "============================================"
    echo "🎯 Running Strategy: $PIPELINE_TYPE"
    echo "============================================"

    # Determine paths
    if [ "$PIPELINE_TYPE" == "fl" ]; then
        PIPELINE_FILE="src/pipelines/fl_pipeline.py"
        PIPELINE_YAML="pipeline_specs/fl_pipeline.yaml"
    elif [ "$PIPELINE_TYPE" == "single" ]; then
        PIPELINE_FILE="src/pipelines/single_pipeline.py"
        PIPELINE_YAML="pipeline_specs/single_pipeline.yaml"
    else
        PIPELINE_FILE="src/pipelines/${PIPELINE_TYPE}_pipeline.py"
        PIPELINE_YAML="pipeline_specs/${PIPELINE_TYPE}_pipeline.yaml"
    fi

    echo "📄 Install dependencies & Compiling Pipeline for $PIPELINE_TYPE..."
    pip install kfp boto3 --quiet || echo "Warning: pip install failed"

    if [ -f "$PIPELINE_FILE" ]; then
        python "$PIPELINE_FILE"
        echo "✅ Generated $PIPELINE_YAML"
    else
        echo "❌ Pipeline file '$PIPELINE_FILE' not found!"
        return 1
    fi
    
    echo "   Submitting Pipeline Run..."
    python src/automate_run.py "$PIPELINE_TYPE"
    
    if [ $? -eq 0 ]; then
        echo "✅ Pipeline Run Completed."
        echo "📥 Fetching Results..."
        python src/fetch_results.py "$PIPELINE_TYPE"
        
        # Display Results
        if [ -f "metrics/last_run_id_$PIPELINE_TYPE.txt" ]; then
            RUN_ID=$(cat "metrics/last_run_id_$PIPELINE_TYPE.txt")
            echo "Last Run ID for $PIPELINE_TYPE: $RUN_ID"
            METRICS_FILE="metrics/metrics_${PIPELINE_TYPE}_${RUN_ID}.csv"
            if [ -f "$METRICS_FILE" ]; then
                echo "✅ Results saved to: $METRICS_FILE"
                echo "📊 Preview (last 5 lines):"
                tail -n 5 "$METRICS_FILE"
            fi
        fi
    else
        echo "❌ Pipeline Run Failed."
        return 1
    fi
}

echo "🧪 Preparing Execution Environment..."
kubectl wait --for=condition=ready pod -l app=ml-pipeline -n kubeflow --timeout=300s
kubectl wait --for=condition=ready pod -l app=ml-pipeline-ui -n kubeflow --timeout=300s

# Start Port Forwarding in background
echo "   Starting Robust Port Forwarding..."
# Disable job control to suppress "Killed" messages
set +m 2>/dev/null || true

# Deep Cleanup of zombie port-forwards and monitoring loops
pkill -f "kubectl port-forward" 2>/dev/null || true
pkill -f "curl.*healthz" 2>/dev/null || true
lsof -ti:8080,9000 | xargs kill -9 2>/dev/null || true

# Function to start PF and monitor (with logging)
start_pf_monitored() {
    local local_port=$1
    local remote_port=$2
    local target=$3
    local check_url=$4
    local log_file="port_forward_${local_port}.log"
    
    (
        while true; do
            local is_healthy=false
            if [ -n "$check_url" ]; then
                if curl -s --max-time 5 "$check_url" > /dev/null; then
                    is_healthy=true
                fi
            else
                if lsof -i :$local_port >/dev/null 2>&1; then
                    is_healthy=true
                fi
            fi
            
            if ! $is_healthy; then
                echo "[$(date)] Port $local_port unhealthy. Restarting..." >> "$log_file"
                lsof -ti:$local_port | xargs kill -9 2>/dev/null || true
                
                # Start PF with logging
                kubectl port-forward --address 127.0.0.1 -n kubeflow "$target" "$local_port:$remote_port" >> "$log_file" 2>&1 &
                
                # Fast check to see if it died immediately
                local pf_pid=$!
                sleep 2
                if ! kill -0 $pf_pid 2>/dev/null; then
                    echo "[$(date)] Port-forward died immediately. Check logs." >> "$log_file"
                fi
                sleep 5 # Wait a bit before next main check
            else
                 sleep 10 # Healthy, check again in 10s
            fi
        done
    ) >/dev/null 2>&1 &
}

# Start monitoring subshells
rm -f port_forward_*.log
# Forward Local 8080 -> Remote 80 (ml-pipeline-ui)
start_pf_monitored 8080 80 "svc/ml-pipeline-ui" "http://127.0.0.1:8080/apis/v1beta1/healthz"
MONITOR_8080_PID=$!
# Forward Local 9000 -> Remote 9000 (minio)
start_pf_monitored 9000 9000 "svc/minio-service" ""
MONITOR_9000_PID=$!

# Add Trap for cleanup on exit
cleanup() {
    echo ""
    echo "🧹 Cleaning up background processes..."
    kill $MONITOR_8080_PID $MONITOR_9000_PID 2>/dev/null || true
    pkill -f "kubectl port-forward" 2>/dev/null || true
    lsof -ti:8080,9000 | xargs kill -9 2>/dev/null || true
    echo "✅ Done."
}
trap cleanup EXIT INT TERM

echo "   Waiting for API to respond (logs at port_forward_8080.log)..."
for i in {1..60}; do
    if curl -s --max-time 2 http://127.0.0.1:8080/apis/v1beta1/healthz > /dev/null; then
        echo "✅ KFP API is UP."
        break
    fi
    if [ $i -eq 60 ]; then
        echo "❌ KFP API failed to start."
        echo "🔍 Log Tail (port_forward_8080.log):"
        tail -n 10 port_forward_8080.log
        exit 1
    fi
    sleep 2
done

# Run Logic
if [ "$PIPELINE_ARG" == "all" ]; then
    echo "🔁 Running ALL pipelines..."
    # Ensure generated pipelines are ready
    if [ ! -f "src/pipelines/fl_visual_pipeline.py" ]; then
          echo "⚠️ fl_visual_pipeline.py missing. Generating..."
          python src/pipelines/generate_visual_pipeline.py
    fi

    # Order: Single -> Single Visual -> FL -> FL Visual
    for pt in "single" "single_visual" "fl" "fl_visual"; do
        run_pipeline "$pt"
    done
else
    # Auto-generate if visual and missing (just in case)
    if [ "$PIPELINE_ARG" == "fl_visual" ] && [ ! -f "src/pipelines/fl_visual_pipeline.py" ]; then
        python src/pipelines/generate_visual_pipeline.py
    fi

    run_pipeline "$PIPELINE_ARG"
fi

# 7. Generate Visualizations
echo "📊 Generating Comparison Plots..."
python src/analysis/compare_results.py || echo "⚠️ Could not generate comparison plot (maybe missing data?)"
python src/analysis/worker_diversity.py || echo "⚠️ Could not generate worker diversity plot"

# Generate generalization gap analysis for both pipeline types
if [ "$PIPELINE_ARG" == "single" ] || [ "$PIPELINE_ARG" == "single_visual" ] || [ "$PIPELINE_ARG" == "all" ]; then
    python src/analysis/generalization_gap.py single || echo "⚠️ Could not generate single generalization gap"
fi
if [ "$PIPELINE_ARG" == "fl" ] || [ "$PIPELINE_ARG" == "fl_visual" ] || [ "$PIPELINE_ARG" == "all" ]; then
    python src/analysis/generalization_gap.py fl || echo "⚠️ Could not generate FL generalization gap"
fi

echo ""
echo "🎉 Process Complete!"

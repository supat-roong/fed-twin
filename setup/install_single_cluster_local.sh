#!/usr/bin/env bash
# install_single_cluster_local.sh — bootstrap the Fed-Twin stack via vendor/fed-infra.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FED_INFRA_ROOT="${ROOT_DIR}/vendor/fed-infra"
export FED_INFRA_ROOT
# shellcheck source=/dev/null
. "${FED_INFRA_ROOT}/lib/common.sh"
# shellcheck source=/dev/null
. "${FED_INFRA_ROOT}/lib/config.sh"
# shellcheck source=/dev/null
. "${FED_INFRA_ROOT}/lib/nodeport.sh"
fed_config_load "${ROOT_DIR}/infra.env"

echo "Syncing uv environment..."
command -v uv >/dev/null 2>&1 && uv sync || echo "uv not found, skipping sync"

echo "Building Fed-Twin image..."
docker build -t fed-twin-app:v1 -f "${ROOT_DIR}/docker/Dockerfile.app" "${ROOT_DIR}"

"${ROOT_DIR}/vendor/fed-infra/bin/fed-infra-up" --env "${ROOT_DIR}/infra.env"

echo "Granting the KFP default service account cluster-admin..."
kubectl create clusterrolebinding pipeline-runner-extend \
  --clusterrole=cluster-admin --serviceaccount=kubeflow:default \
  --dry-run=client -o yaml | kubectl apply -f -

# The Temporal Helm chart installs the server and waits for the frontend to
# roll out, but never registers a Temporal namespace. worker_main.py and (in
# Phase 3b) the pipeline's own Temporal client both use "default" (fed-twin's
# own k8s/temporal-worker.yaml sets TEMPORAL_NAMESPACE=default), and this
# chart configuration does not create it automatically -- every
# client.start_workflow() call would otherwise fail outright with
# NamespaceNotFound, even though the worker itself connects and polls fine
# regardless (active-fed hit and fixed this exact gap; same chart, same gap).
# Registered via the chart's own bundled temporal-admintools deployment.
# Idempotent: `namespace create` exits non-zero ("already exists") on a
# second run, so check first via `describe` and only create when absent --
# re-running setup against an already-provisioned cluster must not fail.
echo "Registering Temporal namespace 'default'..."
if kubectl exec -n kubeflow deploy/temporal-admintools -- \
    temporal operator namespace describe --namespace default >/dev/null 2>&1; then
  echo "Temporal namespace 'default' already registered."
else
  kubectl exec -n kubeflow deploy/temporal-admintools -- \
    temporal operator namespace create --namespace default
fi

echo "Applying fed-twin RBAC and the Temporal worker Deployment..."
kubectl apply -f "${ROOT_DIR}/k8s/rbac.yaml"
kubectl apply -f "${ROOT_DIR}/k8s/temporal-worker.yaml"

echo "Exposing KFP MinIO..."
fed_expose_nodeport minio-service kubeflow \
  "[{\"name\":\"api\",\"port\":9000,\"targetPort\":9000,\"nodePort\":${FED_NODEPORT_MINIO_API}},{\"name\":\"console\",\"port\":9001,\"targetPort\":9001,\"nodePort\":${FED_NODEPORT_MINIO_CONSOLE}}]"

echo "Setup complete."

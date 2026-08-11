#!/usr/bin/env bash
# install_single_cluster_local.sh — bootstrap the Fed-Twin stack via vendor/fed-infra.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Syncing uv environment..."
command -v uv >/dev/null 2>&1 && uv sync || echo "uv not found, skipping sync"

echo "Building Fed-Twin image..."
docker build -t fed-twin-app:v1 -f "${ROOT_DIR}/docker/Dockerfile.app" "${ROOT_DIR}"

"${ROOT_DIR}/vendor/fed-infra/bin/fed-infra-up" --env "${ROOT_DIR}/infra.env"

echo "Granting the KFP default service account cluster-admin..."
kubectl create clusterrolebinding pipeline-runner-extend \
  --clusterrole=cluster-admin --serviceaccount=kubeflow:default \
  --dry-run=client -o yaml | kubectl apply -f -

echo "Exposing KFP MinIO..."
kubectl patch service minio-service -n kubeflow --type=json -p='[
  {"op":"replace","path":"/spec/type","value":"NodePort"},
  {"op":"replace","path":"/spec/ports","value":[
    {"name":"api","port":9000,"protocol":"TCP","targetPort":9000,"nodePort":30900},
    {"name":"console","port":9001,"protocol":"TCP","targetPort":9001,"nodePort":30901}]}]'

echo "Setup complete."

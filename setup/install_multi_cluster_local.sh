#!/usr/bin/env bash
# install_multi_cluster_local.sh — bootstrap the Fed-Twin stack (host cluster
# + Karmada member clusters) via vendor/fed-infra. Mirrors
# install_single_cluster_local.sh; see there for shared setup this doesn't
# repeat.
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
fed_config_load "${ROOT_DIR}/infra.env.multi"

echo "Syncing uv environment..."
command -v uv >/dev/null 2>&1 && uv sync || echo "uv not found, skipping sync"

echo "Building Fed-Twin image..."
docker build -t fed-twin-app:v1 -f "${ROOT_DIR}/docker/Dockerfile.app" "${ROOT_DIR}"

# karmadactl isn't installed by fed-infra: fed_karmada_init/fed_karmada_join
# (vendor/fed-infra/lib/karmada.sh) require it already on PATH and fed_die if
# it's missing. Installing it here is generic to any Karmada consumer, not
# fed-twin-specific -- a promotion candidate for fed-infra itself (see
# docs/task-6-report.md).
if ! command -v karmadactl >/dev/null 2>&1; then
  # Pinned to FED_KARMADA_VERSION rather than installing "latest":
  # fed-infra's fed_karmada_init hard-fails when karmadactl's version differs
  # from FED_KARMADA_VERSION, so an unpinned install would succeed here and
  # then abort the bootstrap moments later with a version-mismatch error --
  # on a fresh machine, the most confusing possible time for it. install-cli.sh
  # wants the version without the leading "v" (it prepends one itself).
  echo "Installing karmadactl ${FED_KARMADA_VERSION}..."
  curl -s --proto '=https' --tlsv1.2 -sSf \
    https://raw.githubusercontent.com/karmada-io/karmada/master/hack/install-cli.sh \
    | INSTALL_CLI_VERSION="${FED_KARMADA_VERSION#v}" bash
fi

"${ROOT_DIR}/vendor/fed-infra/bin/fed-infra-up" --env "${ROOT_DIR}/infra.env.multi"

# fed-infra-up (fed_up_multi, vendor/fed-infra/lib/components.sh) leaves the
# current kubectl context on the host cluster once it returns, same as the
# single-profile path -- every kubectl call below is unqualified, same as
# install_single_cluster_local.sh.

echo "Granting the KFP default service account cluster-admin..."
kubectl create clusterrolebinding pipeline-runner-extend \
  --clusterrole=cluster-admin --serviceaccount=kubeflow:default \
  --dry-run=client -o yaml | kubectl apply -f -

echo "Exposing KFP MinIO..."
fed_expose_nodeport minio-service kubeflow \
  "[{\"name\":\"api\",\"port\":9000,\"targetPort\":9000,\"nodePort\":${FED_NODEPORT_MINIO_API}},{\"name\":\"console\",\"port\":9001,\"targetPort\":9001,\"nodePort\":${FED_NODEPORT_MINIO_CONSOLE}}]"

# ---- Karmada Dashboard ----
# Not a fed-infra component: it's genuinely optional tooling on top of the
# karmada component, not part of the consumer-agnostic bootstrap contract,
# so it stays here rather than being promoted into vendor/fed-infra (see
# docs/task-6-report.md).
echo "=================================================="
echo "  Deploying Karmada Dashboard"
echo "=================================================="
kubectl config use-context "kind-${FED_CLUSTER_NAME}"

echo "Applying Karmada Dashboard manifests..."
kubectl apply -f https://raw.githubusercontent.com/karmada-io/dashboard/main/deploy/karmada-dashboard.yaml

echo "Deploying Secret for Karmada API configuration..."
kubectl create secret generic karmada-kubeconfig \
  --from-file=karmada-kubeconfig="${FED_KARMADA_CONFIG}" -n karmada-system \
  --dry-run=client -o yaml | kubectl apply -f -
kubectl create secret generic karmada-kubeconfig-kf \
  --from-file=karmada-kubeconfig="${FED_KARMADA_CONFIG}" -n kubeflow \
  --dry-run=client -o yaml | kubectl apply -f -

echo "Exposing Karmada Dashboard via NodePort 32000..."
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

echo "Waiting for Karmada Dashboard..."
kubectl rollout status deployment/karmada-dashboard -n karmada-system --timeout=5m || true

# ---- Karmada Dashboard access token ----
echo "=================================================="
echo "  Generating Karmada Dashboard Access Token"
echo "=================================================="
kubectl --kubeconfig="${FED_KARMADA_CONFIG}" create serviceaccount karmada-admin-sa \
  -n karmada-system --dry-run=client -o yaml \
  | kubectl --kubeconfig="${FED_KARMADA_CONFIG}" apply -f -
kubectl --kubeconfig="${FED_KARMADA_CONFIG}" create clusterrolebinding karmada-admin-sa-binding \
  --clusterrole=cluster-admin --serviceaccount=karmada-system:karmada-admin-sa \
  --dry-run=client -o yaml | kubectl --kubeconfig="${FED_KARMADA_CONFIG}" apply -f -

DASHBOARD_TOKEN=$(kubectl --kubeconfig="${FED_KARMADA_CONFIG}" create token karmada-admin-sa \
  -n karmada-system --duration=24h)
echo "Dashboard Access Token (expires in 24h):"
echo "--------------------------------------------------"
echo "$DASHBOARD_TOKEN"
echo "--------------------------------------------------"

# NOTE: the multi-host kind cluster fed-infra creates (vendor/fed-infra/kind/
# multi-host.yaml.tpl) does not map hostPort 32000 (dashboard) or 32443 (the
# Karmada apiserver's own NodePort) out to this machine, unlike the old
# hand-rolled setup/kind-multi-cluster-host.yaml it replaces. See
# docs/task-6-report.md for why this is a fed-infra gap, not something fixed
# here. Until it's addressed, reach the dashboard with:
#   kubectl port-forward -n karmada-system svc/karmada-dashboard 32000:80
echo "Setup complete. Dashboard: use 'kubectl port-forward -n karmada-system svc/karmada-dashboard 32000:80', then http://localhost:32000"

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
# shellcheck source=/dev/null
. "${FED_INFRA_ROOT}/lib/dashboard.sh"
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

echo "Registering Temporal namespace 'default'..."
if kubectl exec -n "${FED_NAMESPACE}" deploy/temporal-admintools -- \
    temporal operator namespace describe --namespace default >/dev/null 2>&1; then
  echo "Temporal namespace 'default' already registered."
else
  kubectl exec -n "${FED_NAMESPACE}" deploy/temporal-admintools -- \
    temporal operator namespace create --namespace default
fi

echo "Applying fed-twin RBAC..."
kubectl apply -f "${ROOT_DIR}/k8s/rbac.yaml"

# The kubeconfig fed-infra writes points at https://127.0.0.1:<nodeport>,
# which is right for this machine (kind maps that port to the host) but
# meaningless inside a pod, where 127.0.0.1 is the pod's own loopback. The
# Temporal worker runs as a pod on the host cluster, so handing it that file
# verbatim would make every topology='multi' dispatch fail to connect.
# Rewrite the server to the Karmada apiserver's in-cluster Service DNS, whose
# port is read from the live Service rather than hardcoded so this cannot
# drift from whatever karmadactl actually created (the deleted legacy
# pipelines hardcoded :5443 here -- exactly the smell this replaces).
echo "Rewriting the Karmada kubeconfig for in-cluster use..."
KARMADA_SVC_PORT=$(kubectl -n karmada-system get svc karmada-apiserver \
  -o jsonpath='{.spec.ports[0].port}') || KARMADA_SVC_PORT=""
if [ -z "$KARMADA_SVC_PORT" ]; then
  echo "ERROR: could not read the karmada-apiserver Service port in karmada-system." >&2
  echo "       Is the Karmada control plane installed on this cluster?" >&2
  exit 1
fi

KARMADA_INCLUSTER_CONFIG=$(mktemp)
trap 'rm -f "$KARMADA_INCLUSTER_CONFIG"' EXIT
cp "${FED_KARMADA_CONFIG}" "$KARMADA_INCLUSTER_CONFIG"
KARMADA_CLUSTER_NAME=$(kubectl --kubeconfig="$KARMADA_INCLUSTER_CONFIG" \
  config view -o jsonpath='{.clusters[0].name}')
kubectl --kubeconfig="$KARMADA_INCLUSTER_CONFIG" config set-cluster \
  "$KARMADA_CLUSTER_NAME" \
  --server="https://karmada-apiserver.karmada-system.svc.cluster.local:${KARMADA_SVC_PORT}"

# Idempotent: `create secret --dry-run=client -o yaml | apply` re-running
# this against an already-provisioned cluster updates the Secret in place
# instead of failing (a plain `kubectl create secret` 409s on a second run).
echo "Creating/updating the Karmada kubeconfig Secret for the Temporal worker..."
kubectl create secret generic karmada-kubeconfig -n "${FED_NAMESPACE}" \
  --from-file=karmada-apiserver.config="$KARMADA_INCLUSTER_CONFIG" \
  --dry-run=client -o yaml | kubectl apply -f -

echo "Applying the Temporal worker Deployment..."
kubectl apply -f "${ROOT_DIR}/k8s/temporal-worker.yaml"

echo "Exposing KFP MinIO..."
fed_expose_nodeport minio-service kubeflow \
  "[{\"name\":\"api\",\"port\":9000,\"targetPort\":9000,\"nodePort\":${FED_NODEPORT_MINIO_API}},{\"name\":\"console\",\"port\":9001,\"targetPort\":9001,\"nodePort\":${FED_NODEPORT_MINIO_CONSOLE}}]"

# ---- Karmada Dashboard ----
# Promoted into vendor/fed-infra as the `karmada-dashboard` component (see
# docs/task-6-report.md for the history): fed-infra-up above already installed
# it, wired its Karmada-apiserver kubeconfig Secret into both karmada-system
# and kubeflow, and exposed it as a NodePort mapped out to this host at
# FED_HOSTPORT_KARMADA_DASHBOARD -- no port-forward needed anymore.
#
# fed_dashboard_token deliberately prints the token to stdout only (never
# through fed_log), so it cannot leak into a redirected log file -- which
# also means fed-infra-up itself never surfaces it. That token is the only
# way to log in, so this script must mint and print it explicitly, the same
# as the inlined block this replaces did.
echo "=================================================="
echo "  Generating Karmada Dashboard Access Token"
echo "=================================================="
DASHBOARD_TOKEN=$(KUBECONFIG="${FED_KARMADA_CONFIG}" fed_dashboard_token karmada-system karmada-admin-sa)
echo "Dashboard Access Token (expires in 24h):"
echo "--------------------------------------------------"
echo "$DASHBOARD_TOKEN"
echo "--------------------------------------------------"

echo "Setup complete. Karmada Dashboard: http://localhost:${FED_HOSTPORT_KARMADA_DASHBOARD}"

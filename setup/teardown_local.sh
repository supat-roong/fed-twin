#!/usr/bin/env bash
# teardown_local.sh — Destroys the Federated Digital Twin local kind cluster

set -euo pipefail

CLUSTER_NAME="fed-twin-cluster"

echo "🗑️ Destroying kind cluster '$CLUSTER_NAME'..."
kind delete cluster --name "$CLUSTER_NAME"
echo "✅ Teardown complete."

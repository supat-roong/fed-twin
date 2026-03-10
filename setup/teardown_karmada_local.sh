#!/usr/bin/env bash
# teardown_karmada_local.sh — Destroys the Federated Digital Twin Karmada local kind clusters

set -euo pipefail

# Make sure environment has standard paths
source ~/.zshrc 2>/dev/null || true
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"

HOST_CLUSTER="fed-twin-host"
MEMBER_PREFIX="fed-twin-member"

echo "🗑️ Destroying Host cluster '$HOST_CLUSTER'..."
kind delete cluster --name "$HOST_CLUSTER" || true

# We don't know exactly how many members exist, so we delete all that match the prefix
for member in $(kind get clusters 2>/dev/null | grep "^$MEMBER_PREFIX" || true); do
    echo "🗑️ Destroying Member cluster '$member'..."
    kind delete cluster --name "$member"
done

echo "✅ Teardown complete."

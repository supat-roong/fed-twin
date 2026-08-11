#!/usr/bin/env bash
# teardown_single_cluster_local.sh — destroy the local cluster via vendor/fed-infra.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"${ROOT_DIR}/vendor/fed-infra/bin/fed-infra-down" --env "${ROOT_DIR}/infra.env"

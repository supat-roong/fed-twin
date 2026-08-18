#!/usr/bin/env bash
# teardown_multi_cluster_local.sh — destroy the host + Karmada member
# clusters via vendor/fed-infra. Mirrors teardown_single_cluster_local.sh.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"${ROOT_DIR}/vendor/fed-infra/bin/fed-infra-down" --env "${ROOT_DIR}/infra.env.multi"

#!/bin/bash
# shellcheck shell=bash
set -euo pipefail

# Fast log scanner for ContraDiff jobs.
# Usage:
#   bash scripts/slurm/scan_contradiff_logs.sh
#   LOG_DIR=/path/to/logs bash scripts/slurm/scan_contradiff_logs.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"
PATTERN="${PATTERN:-STEP|credit_used|ERROR|Traceback|ContraDiff ideaA done|ProxyEval End|Summary}"

if [ ! -d "${LOG_DIR}" ]; then
  echo "[ERR] LOG_DIR not found: ${LOG_DIR}" >&2
  exit 2
fi

echo "=== scanning ${LOG_DIR} ==="
# -a avoids grep failure when logs contain non-UTF8 chars from progress bars.
grep -aE "${PATTERN}" "${LOG_DIR}"/contradiff*.out "${LOG_DIR}"/contradiff*.err 2>/dev/null | tail -n 200 || true


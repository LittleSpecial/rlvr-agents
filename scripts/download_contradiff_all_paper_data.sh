#!/bin/bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTRADIFF_DIR="${CONTRADIFF_DIR:-${ROOT_DIR}/contradiff}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "Using CONTRADIFF_DIR=${CONTRADIFF_DIR}"
echo "Using PYTHON_BIN=${PYTHON_BIN}"

export CONTRADIFF_DIR
export PYTHONPATH="${CONTRADIFF_DIR}/main:${CONTRADIFF_DIR}:${PYTHONPATH:-}"

# 1) locomotion (paper scope + medium-expert)
"${PYTHON_BIN}" "${ROOT_DIR}/scripts/download_locomotion_datasets.py" --skip-existing "$@"

# 2) maze2d + kitchen (paper non-locomotion scope)
"${PYTHON_BIN}" "${ROOT_DIR}/scripts/download_maze_kitchen_datasets.py" --skip-existing

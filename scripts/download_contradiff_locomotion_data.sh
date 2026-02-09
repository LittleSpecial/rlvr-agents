#!/bin/bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTRADIFF_DIR="${CONTRADIFF_DIR:-${ROOT_DIR}/contradiff}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [ ! -f "${CONTRADIFF_DIR}/main/scripts/download_locomotion_datasets.py" ]; then
  echo "[ERR] cannot find downloader script under ${CONTRADIFF_DIR}/main/scripts" >&2
  exit 2
fi

echo "Using CONTRADIFF_DIR=${CONTRADIFF_DIR}"
echo "Using PYTHON_BIN=${PYTHON_BIN}"

"${PYTHON_BIN}" "${CONTRADIFF_DIR}/main/scripts/download_locomotion_datasets.py" "$@"


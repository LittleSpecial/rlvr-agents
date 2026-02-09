#!/bin/bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTRADIFF_DIR="${CONTRADIFF_DIR:-${ROOT_DIR}/contradiff}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# Optional cluster module loading (same knobs as training scripts).
GCC_MODULE="${GCC_MODULE:-}"
CUDA_MODULE="${CUDA_MODULE:-}"
CUDNN_MODULE="${CUDNN_MODULE:-}"
NCCL_MODULE="${NCCL_MODULE:-}"
if ! type module >/dev/null 2>&1; then
  [ -f /etc/profile.d/modules.sh ] && source /etc/profile.d/modules.sh || true
  [ -f /usr/share/Modules/init/bash ] && source /usr/share/Modules/init/bash || true
fi
if type module >/dev/null 2>&1; then
  [ -n "${GCC_MODULE}" ] && module load "${GCC_MODULE}" || true
  [ -n "${CUDA_MODULE}" ] && module load "${CUDA_MODULE}" || true
  [ -n "${CUDNN_MODULE}" ] && module load "${CUDNN_MODULE}" || true
  [ -n "${NCCL_MODULE}" ] && module load "${NCCL_MODULE}" || true
fi

PRIMARY_DOWNLOADER="${ROOT_DIR}/scripts/download_locomotion_datasets.py"
FALLBACK_DOWNLOADER="${CONTRADIFF_DIR}/main/scripts/download_locomotion_datasets.py"
if [ -f "${PRIMARY_DOWNLOADER}" ]; then
  DOWNLOADER_SCRIPT="${PRIMARY_DOWNLOADER}"
elif [ -f "${FALLBACK_DOWNLOADER}" ]; then
  DOWNLOADER_SCRIPT="${FALLBACK_DOWNLOADER}"
  echo "[WARN] missing ${PRIMARY_DOWNLOADER}; fallback to ${FALLBACK_DOWNLOADER}" >&2
else
  echo "[ERR] cannot find downloader script." >&2
  echo "Checked: ${PRIMARY_DOWNLOADER}" >&2
  echo "Checked: ${FALLBACK_DOWNLOADER}" >&2
  exit 2
fi

echo "Using CONTRADIFF_DIR=${CONTRADIFF_DIR}"
echo "Using PYTHON_BIN=${PYTHON_BIN}"
echo "Using DOWNLOADER_SCRIPT=${DOWNLOADER_SCRIPT}"
export CONTRADIFF_DIR
export PYTHONPATH="${CONTRADIFF_DIR}/main:${CONTRADIFF_DIR}:${PYTHONPATH:-}"
echo "Using PYTHONPATH=${PYTHONPATH}"

"${PYTHON_BIN}" "${DOWNLOADER_SCRIPT}" "$@"

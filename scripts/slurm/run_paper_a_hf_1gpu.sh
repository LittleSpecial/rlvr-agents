#!/bin/bash
#SBATCH --job-name=paper_a_hf
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --output=logs/paper_a_hf_%j.out
#SBATCH --error=logs/paper_a_hf_%j.err
#SBATCH --export=ALL

# Paper A HF backend (single GPU debug/smoke).
# Usage:
#   MODEL_PATH=... TRAIN_DATA=... EVAL_DATA=... sbatch scripts/slurm/run_paper_a_hf_1gpu.sh

DEFAULT_NUM_GPUS=1
COMMON_SH=""
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/scripts/slurm/paper_a_hf_common.sh" ]; then
  COMMON_SH="${SLURM_SUBMIT_DIR}/scripts/slurm/paper_a_hf_common.sh"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  if [ -f "${SCRIPT_DIR}/paper_a_hf_common.sh" ]; then
    COMMON_SH="${SCRIPT_DIR}/paper_a_hf_common.sh"
  fi
fi

if [ -z "${COMMON_SH}" ]; then
  echo "[ERR] Cannot locate paper_a_hf_common.sh. Expected at ${SLURM_SUBMIT_DIR:-<unset>}/scripts/slurm/." >&2
  exit 2
fi

source "${COMMON_SH}"

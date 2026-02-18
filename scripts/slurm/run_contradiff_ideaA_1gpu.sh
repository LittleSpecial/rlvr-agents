#!/bin/bash
#SBATCH --job-name=contradiff_a
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --output=logs/contradiff_a_%j.out
#SBATCH --error=logs/contradiff_a_%j.err
#SBATCH --export=ALL

# Single-GPU ContraDiff + ideaA launcher.
# Use this script to avoid requesting 4 GPUs for single-process runs.

DEFAULT_NUM_GPUS=1
COMMON_SH=""
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/scripts/slurm/contradiff_ideaA_common.sh" ]; then
  COMMON_SH="${SLURM_SUBMIT_DIR}/scripts/slurm/contradiff_ideaA_common.sh"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  if [ -f "${SCRIPT_DIR}/contradiff_ideaA_common.sh" ]; then
    COMMON_SH="${SCRIPT_DIR}/contradiff_ideaA_common.sh"
  fi
fi

if [ -z "${COMMON_SH}" ]; then
  echo "[ERR] Cannot locate contradiff_ideaA_common.sh." >&2
  exit 2
fi

source "${COMMON_SH}"

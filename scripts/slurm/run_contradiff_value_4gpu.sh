#!/bin/bash
#SBATCH --job-name=contradiff_value
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=128
#SBATCH --time=48:00:00
#SBATCH --output=logs/contradiff_value_%j.out
#SBATCH --error=logs/contradiff_value_%j.err
#SBATCH --export=ALL

# Multi-GPU ContraDiff value launcher.

DEFAULT_NUM_GPUS=4
COMMON_SH=""
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/scripts/slurm/contradiff_value_common.sh" ]; then
  COMMON_SH="${SLURM_SUBMIT_DIR}/scripts/slurm/contradiff_value_common.sh"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  if [ -f "${SCRIPT_DIR}/contradiff_value_common.sh" ]; then
    COMMON_SH="${SCRIPT_DIR}/contradiff_value_common.sh"
  fi
fi

if [ -z "${COMMON_SH}" ]; then
  echo "[ERR] Cannot locate contradiff_value_common.sh." >&2
  exit 2
fi

source "${COMMON_SH}"

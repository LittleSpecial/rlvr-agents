#!/bin/bash
#SBATCH --job-name=contradiff_cacdp
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=128
#SBATCH --time=48:00:00
#SBATCH --output=logs/contradiff_cacdp_%j.out
#SBATCH --error=logs/contradiff_cacdp_%j.err
#SBATCH --export=ALL

DEFAULT_NUM_GPUS=4
COMMON_SH=""
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/scripts/slurm/contradiff_cacdp_common.sh" ]; then
  COMMON_SH="${SLURM_SUBMIT_DIR}/scripts/slurm/contradiff_cacdp_common.sh"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  if [ -f "${SCRIPT_DIR}/contradiff_cacdp_common.sh" ]; then
    COMMON_SH="${SCRIPT_DIR}/contradiff_cacdp_common.sh"
  fi
fi

if [ -z "${COMMON_SH}" ]; then
  echo "[ERR] Cannot locate contradiff_cacdp_common.sh." >&2
  exit 2
fi

source "${COMMON_SH}"

#!/bin/bash
#SBATCH --job-name=contradiff_value
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --output=logs/contradiff_value_%j.out
#SBATCH --error=logs/contradiff_value_%j.err
#SBATCH --export=ALL

# Train ContraDiff value model used by plan_guided evaluation.
# Required:
#   RUN_LOGBASE=/abs/path/to/contradiff/main/logs_runs/<experiment>
# Optional:
#   VALUE_BRANCH (default: plan1_diffuser)
#   VALUE_STEPS (default: 200000)

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

#!/bin/bash
#SBATCH --job-name=contradiff_a
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=128
#SBATCH --time=24:00:00
#SBATCH --output=logs/contradiff_a_%j.out
#SBATCH --error=logs/contradiff_a_%j.err
#SBATCH --export=ALL

# ContraDiff + ideaA (counterfactual-credit weighted loss).
# Usage:
#   CUDA_MODULE=compilers/cuda/11.8 GCC_MODULE=compilers/gcc/11.3.0 NCCL_MODULE=nccl/2.11.4-1_cuda11.8 \
#   CUDNN_MODULE=cudnn/8.6.0.163_cuda11.x CONDA_ENV=$HOME/.conda/envs/rlvr \
#   EXPERIMENT_NAME=contradiff_${USER}_$RANDOM MAX_STEPS=400 LOG_INTERVAL=20 EVAL_INTERVAL=100 SAVE_INTERVAL=100 \
#   LEARNING_RATE=5e-6 BATCH_SIZE=5 USE_COUNTERFACTUAL_CREDIT=1 COUNTERFACTUAL_K=2 \
#   DATASET=hopper-random-v2 EXPERT_RATIO=0.2 EXP_DATASET=expert \
#   sbatch scripts/slurm/run_contradiff_ideaA_4gpu.sh

DEFAULT_NUM_GPUS=4
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

#!/bin/bash
#SBATCH --job-name=contradiff_eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --output=logs/contradiff_eval_%j.out
#SBATCH --error=logs/contradiff_eval_%j.err
#SBATCH --export=ALL

# ContraDiff paper-style evaluation (guided planning rollout).
# Usage:
#   CUDA_MODULE=compilers/cuda/11.8 GCC_MODULE=compilers/gcc/11.3.0 NCCL_MODULE=nccl/2.11.4-1_cuda11.8 \
#   CUDNN_MODULE=cudnn/8.6.0.163_cuda11.x CONDA_ENV=$HOME/.conda/envs/rlvr \
#   CONTRADIFF_DIR=$RLVR/contradiff EXPERIMENT_NAME=cdiff_base_hopper_rand02_user_12345 \
#   DATASET=hopper-random-v2 BRANCH=plan2_hard LOAD_ITER=-1 NUMS_EVAL=50 \
#   VALBASE=/path/to/valuebase \
#   sbatch scripts/slurm/run_contradiff_eval_1gpu.sh

COMMON_SH=""
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/scripts/slurm/contradiff_eval_common.sh" ]; then
  COMMON_SH="${SLURM_SUBMIT_DIR}/scripts/slurm/contradiff_eval_common.sh"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  if [ -f "${SCRIPT_DIR}/contradiff_eval_common.sh" ]; then
    COMMON_SH="${SCRIPT_DIR}/contradiff_eval_common.sh"
  fi
fi

if [ -z "${COMMON_SH}" ]; then
  echo "[ERR] Cannot locate contradiff_eval_common.sh." >&2
  exit 2
fi

source "${COMMON_SH}"

#!/bin/bash
# shellcheck shell=bash
set -euo pipefail

# Submit two parallel tracks:
#   Track A: aarch64-compatible (just-d4rl + proxy eval)
#   Track B: standard online protocol (full d4rl+mujoco_py), optional
#
# Example:
#   bash scripts/slurm/submit_contradiff_dual_tracks.sh
#   ENABLE_TRACK_B=1 TRACK_B_CONDA_ENV=$HOME/.conda/envs/rlvr_full bash scripts/slurm/submit_contradiff_dual_tracks.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

SUBMIT_TRAIN_SCRIPT="${SUBMIT_TRAIN_SCRIPT:-${REPO_ROOT}/scripts/slurm/run_contradiff_ideaA_1gpu.sh}"
SUBMIT_EVAL_SCRIPT="${SUBMIT_EVAL_SCRIPT:-${REPO_ROOT}/scripts/slurm/run_contradiff_eval_1gpu.sh}"
if [ ! -f "${SUBMIT_TRAIN_SCRIPT}" ] || [ ! -f "${SUBMIT_EVAL_SCRIPT}" ]; then
  echo "[ERR] Missing submit scripts." >&2
  echo "train=${SUBMIT_TRAIN_SCRIPT}" >&2
  echo "eval=${SUBMIT_EVAL_SCRIPT}" >&2
  exit 2
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
DRY_RUN="${DRY_RUN:-0}"
dry_job_counter=0

# Common experiment params
DATASET="${DATASET:-hopper-random-v2}"
EXP_DATASET="${EXP_DATASET:-expert}"
EXPERT_RATIO="${EXPERT_RATIO:-0.2}"
MAX_STEPS="${MAX_STEPS:-200000}"
LOG_INTERVAL="${LOG_INTERVAL:-500}"
EVAL_INTERVAL="${EVAL_INTERVAL:-10000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
LEARNING_RATE="${LEARNING_RATE:-5e-6}"
BATCH_SIZE="${BATCH_SIZE:-128}"
COUNTERFACTUAL_K="${COUNTERFACTUAL_K:-8}"
BRANCH="${BRANCH:-plan2_hard}"
N_DIFFUSION_STEPS="${N_DIFFUSION_STEPS:-20}"
HORIZON="${HORIZON:-32}"
NUMS_EVAL="${NUMS_EVAL:-50}"

# Cluster/runtime params
CUDA_MODULE="${CUDA_MODULE:-compilers/cuda/11.8}"
GCC_MODULE="${GCC_MODULE:-compilers/gcc/11.3.0}"
CUDNN_MODULE="${CUDNN_MODULE:-cudnn/8.6.0.163_cuda11.x}"
NCCL_MODULE="${NCCL_MODULE:-nccl/2.11.4-1_cuda11.8}"
CONTRADIFF_DIR="${CONTRADIFF_DIR:-${REPO_ROOT}/contradiff}"
CONDA_ENV="${CONDA_ENV:-$HOME/.conda/envs/rlvr}"
PYTHON_BIN="${PYTHON_BIN:-${CONDA_ENV}/bin/python3}"

submit_and_get_id() {
  if [ "${DRY_RUN}" = "1" ]; then
    dry_job_counter=$((dry_job_counter + 1))
    echo "[DRY_RUN] $*" >&2
    SUBMITTED_JOB_ID="dry_${dry_job_counter}"
    return 0
  fi
  local out
  out="$("$@")"
  echo "${out}" >&2
  SUBMITTED_JOB_ID="$(echo "${out}" | awk '/Submitted batch job/ {print $4}')"
  if [ -z "${SUBMITTED_JOB_ID}" ]; then
    return 1
  fi
}

submit_track() {
  local track="$1"
  local use_counterfactual="$2"
  local use_just_d4rl="$3"
  local strict_real_eval="$4"
  local allow_value_fallback="$5"
  local conda_env="$6"
  local python_bin="$7"
  local name_prefix="$8"

  local exp_name="${name_prefix}_${track}_${DATASET//-/_}_${timestamp}_$RANDOM"
  local train_job eval_job

  submit_and_get_id env \
    CUDA_MODULE="${CUDA_MODULE}" GCC_MODULE="${GCC_MODULE}" NCCL_MODULE="${NCCL_MODULE}" CUDNN_MODULE="${CUDNN_MODULE}" \
    CONDA_ENV="${conda_env}" PYTHON_BIN="${python_bin}" CONTRADIFF_DIR="${CONTRADIFF_DIR}" \
    EXPERIMENT_NAME="${exp_name}" BRANCH="${BRANCH}" DATASET="${DATASET}" EXP_DATASET="${EXP_DATASET}" EXPERT_RATIO="${EXPERT_RATIO}" \
    USE_JUST_D4RL_BACKEND="${use_just_d4rl}" RENDERER="utils.NullRenderer" \
    USE_COUNTERFACTUAL_CREDIT="${use_counterfactual}" COUNTERFACTUAL_K="${COUNTERFACTUAL_K}" \
    MAX_STEPS="${MAX_STEPS}" LOG_INTERVAL="${LOG_INTERVAL}" EVAL_INTERVAL="${EVAL_INTERVAL}" SAVE_INTERVAL="${SAVE_INTERVAL}" \
    LEARNING_RATE="${LEARNING_RATE}" BATCH_SIZE="${BATCH_SIZE}" HORIZON="${HORIZON}" N_DIFFUSION_STEPS="${N_DIFFUSION_STEPS}" \
    sbatch "${SUBMIT_TRAIN_SCRIPT}"
  train_job="${SUBMITTED_JOB_ID:-}"
  if [ -z "${train_job}" ]; then
    echo "[ERR] failed to submit training job for ${exp_name}" >&2
    return 1
  fi

  submit_and_get_id env \
    CUDA_MODULE="${CUDA_MODULE}" GCC_MODULE="${GCC_MODULE}" NCCL_MODULE="${NCCL_MODULE}" CUDNN_MODULE="${CUDNN_MODULE}" \
    CONDA_ENV="${conda_env}" PYTHON_BIN="${python_bin}" CONTRADIFF_DIR="${CONTRADIFF_DIR}" \
    EXPERIMENT_NAME="${exp_name}" BRANCH="${BRANCH}" DATASET="${DATASET}" EXP_DATASET="${EXP_DATASET}" EXPERT_RATIO="${EXPERT_RATIO}" \
    NUMS_EVAL="${NUMS_EVAL}" LOAD_ITER="-1" STRICT_REAL_EVAL="${strict_real_eval}" ALLOW_VALUE_FALLBACK="${allow_value_fallback}" \
    sbatch --dependency=afterok:"${train_job}" "${SUBMIT_EVAL_SCRIPT}"
  eval_job="${SUBMITTED_JOB_ID:-}"
  if [ -z "${eval_job}" ]; then
    echo "[ERR] failed to submit eval job for ${exp_name}" >&2
    return 1
  fi

  echo "[${track}] exp=${exp_name} train_job=${train_job} eval_job=${eval_job}"
}

echo "=== Track A: aarch64-compatible (proxy eval) ==="
submit_track "base" 0 1 0 1 "${CONDA_ENV}" "${PYTHON_BIN}" "trackA"
submit_track "idea" 1 1 0 1 "${CONDA_ENV}" "${PYTHON_BIN}" "trackA"

ENABLE_TRACK_B="${ENABLE_TRACK_B:-1}"
if [ "${ENABLE_TRACK_B}" = "1" ]; then
  TRACK_B_CONDA_ENV="${TRACK_B_CONDA_ENV:-$HOME/.conda/envs/rlvr_full}"
  TRACK_B_PYTHON_BIN="${TRACK_B_PYTHON_BIN:-${TRACK_B_CONDA_ENV}/bin/python3}"
  TRACK_B_USE_JUST_D4RL="${TRACK_B_USE_JUST_D4RL:-0}"
  TRACK_B_STRICT_REAL_EVAL="${TRACK_B_STRICT_REAL_EVAL:-1}"
  TRACK_B_ALLOW_VALUE_FALLBACK="${TRACK_B_ALLOW_VALUE_FALLBACK:-0}"

  if [ ! -x "${TRACK_B_PYTHON_BIN}" ]; then
    echo "[WARN] Track B skipped: python not found at ${TRACK_B_PYTHON_BIN}"
    echo "       Set TRACK_B_CONDA_ENV/TRACK_B_PYTHON_BIN to a full d4rl+mujoco runtime env."
    exit 0
  fi

  echo "=== Track B: standard online protocol ==="
  submit_track "base" 0 "${TRACK_B_USE_JUST_D4RL}" "${TRACK_B_STRICT_REAL_EVAL}" "${TRACK_B_ALLOW_VALUE_FALLBACK}" "${TRACK_B_CONDA_ENV}" "${TRACK_B_PYTHON_BIN}" "trackB"
  submit_track "idea" 1 "${TRACK_B_USE_JUST_D4RL}" "${TRACK_B_STRICT_REAL_EVAL}" "${TRACK_B_ALLOW_VALUE_FALLBACK}" "${TRACK_B_CONDA_ENV}" "${TRACK_B_PYTHON_BIN}" "trackB"
else
  echo "[INFO] Track B disabled (ENABLE_TRACK_B=${ENABLE_TRACK_B})."
fi

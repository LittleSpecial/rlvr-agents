#!/bin/bash
# shellcheck shell=bash

set -eo pipefail
trap 'rc=$?; echo "[ERR] contradiff_ideaA_common.sh failed at line ${LINENO} (rc=${rc})" >&2' ERR

echo "=== Job started at $(date) ==="
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: $(hostname)"
echo "PWD: $(pwd)"
echo "SLURM_SUBMIT_DIR: ${SLURM_SUBMIT_DIR:-$PWD}"

mkdir -p logs experiments
cd "${SLURM_SUBMIT_DIR:-$PWD}"

export TMPDIR="${TMPDIR:-$HOME/tmp}"
mkdir -p "${TMPDIR}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

echo "=== Loading modules (cluster template) ==="
module purge
module load miniforge3/24.1

echo "=== Conda env / Python selection ==="
CONDA_ENV="${CONDA_ENV:-$HOME/.conda/envs/rlvr}"
if [ -d "${CONDA_ENV}" ]; then
  source activate "${CONDA_ENV}" 2>/dev/null || true
fi

if [ -z "${PYTHON_BIN:-}" ] && [ -x "${CONDA_ENV}/bin/python3" ]; then
  PYTHON_BIN="${CONDA_ENV}/bin/python3"
fi
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || true)}"
if [ -z "${PYTHON_BIN}" ]; then
  echo "[ERR] python3 not found in PATH after module load / conda activate." >&2
  exit 2
fi

echo "=== Torch wheel info (no import) ==="
TORCH_WHL_VER="$(${PYTHON_BIN} - <<'PY'
import importlib.metadata as md
try:
    print(md.version("torch"))
except Exception:
    print("")
PY
)"
echo "torch (wheel): ${TORCH_WHL_VER:-UNKNOWN}"

: "${CUDA_MODULE:=compilers/cuda/11.8}"
: "${GCC_MODULE:=compilers/gcc/11.3.0}"
: "${CUDNN_MODULE:=cudnn/8.6.0.163_cuda11.x}"
: "${NCCL_MODULE:=nccl/2.11.4-1_cuda11.8}"

echo "=== Loading compute modules ==="
echo "GCC_MODULE=${GCC_MODULE}"
echo "CUDA_MODULE=${CUDA_MODULE}"
echo "CUDNN_MODULE=${CUDNN_MODULE}"
echo "NCCL_MODULE=${NCCL_MODULE}"
module load "${GCC_MODULE}"
module load "${CUDA_MODULE}"
module load "${CUDNN_MODULE}"
if [ -n "${NCCL_MODULE}" ]; then
  module load "${NCCL_MODULE}" 2>/dev/null || true
fi

echo "=== Module list ==="
module list 2>&1 || true

echo "=== nvidia-smi ==="
nvidia-smi || true

"${PYTHON_BIN}" - <<'PY'
import sys
import torch
print("python:", sys.version.replace("\n", " "))
print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("cuda.is_available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit("\n[ERROR] torch.cuda.is_available() == False\n")
PY

echo "=== Python dependency check ==="
"${PYTHON_BIN}" - <<'PY'
import importlib.util
import sys

mods = ["gym", "d4rl", "numpy", "scipy", "sklearn", "h5py"]
missing = [m for m in mods if importlib.util.find_spec(m) is None]
if missing:
    py = sys.executable
    raise SystemExit(
        "[ERROR] Missing python deps in runtime env: "
        + ", ".join(missing)
        + "\nInstall with:\n"
        + f"  {py} -m pip install --no-user gym==0.26.2 numpy scipy scikit-learn h5py\n"
        + f"  {py} -m pip install --no-user 'd4rl @ git+https://github.com/Farama-Foundation/D4RL.git'\n"
        + "\nOr (if you intentionally installed to user-site): set PYTHONNOUSERSITE=0 when submitting."
    )
print("python deps: OK")
PY

CONTRADIFF_DIR="${CONTRADIFF_DIR:-${SLURM_SUBMIT_DIR:-$PWD}/contradiff}"
if [ ! -d "${CONTRADIFF_DIR}/main" ]; then
  echo "[ERR] CONTRADIFF_DIR is invalid: ${CONTRADIFF_DIR}" >&2
  exit 2
fi

BRANCH="${BRANCH:-plan2_hard}"
DATASET="${DATASET:-hopper-random-v2}"
EXP_DATASET="${EXP_DATASET:-expert}"
EXPERT_RATIO="${EXPERT_RATIO:-0.2}"
LOWERBOUND="${LOWERBOUND:-0.2}"
UPPERBOUND="${UPPERBOUND:-0.0}"
HORIZON="${HORIZON:-32}"
N_DIFFUSION_STEPS="${N_DIFFUSION_STEPS:-20}"
METRICS="${METRICS:-canberra}"
SEED="${SEED:-1000}"
DEVICE="${DEVICE:-cuda:0}"
MAX_STEPS="${MAX_STEPS:-400}"
LOG_INTERVAL="${LOG_INTERVAL:-20}"
EVAL_INTERVAL="${EVAL_INTERVAL:-100}"
SAVE_INTERVAL="${SAVE_INTERVAL:-100}"
LEARNING_RATE="${LEARNING_RATE:-5e-6}"
BATCH_SIZE="${BATCH_SIZE:-5}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-contradiff_${SLURM_JOB_ID:-local}}"
DATASET_INFOS_PATH="${DATASET_INFOS_PATH:-${CONTRADIFF_DIR}/main/dataset_infos}"
VALUE_OF_STATES_PATH="${VALUE_OF_STATES_PATH:-${CONTRADIFF_DIR}/main/value_of_states}"

USE_COUNTERFACTUAL_CREDIT="${USE_COUNTERFACTUAL_CREDIT:-1}"
COUNTERFACTUAL_K="${COUNTERFACTUAL_K:-2}"
CREDIT_WEIGHT_MODE="${CREDIT_WEIGHT_MODE:-anchor_minus_negative}"
CREDIT_WEIGHT_NORM="${CREDIT_WEIGHT_NORM:-signed}"
CREDIT_WEIGHT_ALPHA="${CREDIT_WEIGHT_ALPHA:-1.0}"
CREDIT_WEIGHT_MIN="${CREDIT_WEIGHT_MIN:-0.2}"
CREDIT_WEIGHT_MAX="${CREDIT_WEIGHT_MAX:-2.0}"
CREDIT_WEIGHT_ON_DIFFUSION="${CREDIT_WEIGHT_ON_DIFFUSION:-1}"
CREDIT_WEIGHT_ON_CONTRAST="${CREDIT_WEIGHT_ON_CONTRAST:-1}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

NUM_GPUS="${NUM_GPUS:-${SLURM_GPUS_ON_NODE:-${DEFAULT_NUM_GPUS:-1}}}"
if ! [[ "${NUM_GPUS}" =~ ^[0-9]+$ ]]; then
  NUM_GPUS="$(echo "${NUM_GPUS}" | grep -oE '[0-9]+' | head -n1)"
fi
if ! [[ "${NUM_GPUS}" =~ ^[0-9]+$ ]] || [ "${NUM_GPUS}" -lt 1 ]; then
  echo "[ERR] Invalid NUM_GPUS=${NUM_GPUS}" >&2
  exit 2
fi
if [ "${NUM_GPUS}" -gt 1 ]; then
  echo "[WARN] ContraDiff train_diffuser.py is single-process; using DEVICE=${DEVICE}."
fi

printf -v EXPERT_RATIO_FMT "%.2f" "${EXPERT_RATIO}"
DATASET_PREFIX="${DATASET%-v2}"
CLUSTER_INFO_FILE="${DATASET_INFOS_PATH}/cluster_infos_${DATASET_PREFIX}-${EXPERT_RATIO_FMT}-v2.pkl"
if [ ! -f "${CLUSTER_INFO_FILE}" ]; then
  echo "[ERR] Missing cluster info file: ${CLUSTER_INFO_FILE}" >&2
  echo "Generate dataset infos first (dataset_info + cluster_infos) under ${DATASET_INFOS_PATH}." >&2
  exit 2
fi

echo "=== ContraDiff run config ==="
echo "CONTRADIFF_DIR=${CONTRADIFF_DIR}"
echo "BRANCH=${BRANCH} DATASET=${DATASET}"
echo "EXP_DATASET=${EXP_DATASET} EXPERT_RATIO=${EXPERT_RATIO}"
echo "LOWERBOUND=${LOWERBOUND} UPPERBOUND=${UPPERBOUND} METRICS=${METRICS}"
echo "HORIZON=${HORIZON} N_DIFFUSION_STEPS=${N_DIFFUSION_STEPS}"
echo "MAX_STEPS=${MAX_STEPS} LOG_INTERVAL=${LOG_INTERVAL} EVAL_INTERVAL=${EVAL_INTERVAL} SAVE_INTERVAL=${SAVE_INTERVAL}"
echo "LEARNING_RATE=${LEARNING_RATE} BATCH_SIZE=${BATCH_SIZE}"
echo "USE_COUNTERFACTUAL_CREDIT=${USE_COUNTERFACTUAL_CREDIT} COUNTERFACTUAL_K=${COUNTERFACTUAL_K}"
echo "CREDIT_WEIGHT_MODE=${CREDIT_WEIGHT_MODE} NORM=${CREDIT_WEIGHT_NORM}"
echo "CREDIT_WEIGHT_ALPHA=${CREDIT_WEIGHT_ALPHA} MIN=${CREDIT_WEIGHT_MIN} MAX=${CREDIT_WEIGHT_MAX}"
echo "CREDIT_WEIGHT_ON_DIFFUSION=${CREDIT_WEIGHT_ON_DIFFUSION} CREDIT_WEIGHT_ON_CONTRAST=${CREDIT_WEIGHT_ON_CONTRAST}"
echo "DATASET_INFOS_PATH=${DATASET_INFOS_PATH}"
echo "VALUE_OF_STATES_PATH=${VALUE_OF_STATES_PATH}"
echo "EXTRA_ARGS=${EXTRA_ARGS}"

cd "${CONTRADIFF_DIR}"

PY_ARGS=(
  main/train_diffuser.py
  --branch "${BRANCH}"
  --dataset "${DATASET}"
  --exp_dataset "${EXP_DATASET}"
  --expert_ratio "${EXPERT_RATIO}"
  --lowerbound "${LOWERBOUND}"
  --upperbound "${UPPERBOUND}"
  --metrics "${METRICS}"
  --horizon "${HORIZON}"
  --n_diffusion_steps "${N_DIFFUSION_STEPS}"
  --seed "${SEED}"
  --device "${DEVICE}"
  --n_train_steps "${MAX_STEPS}"
  --log_freq "${LOG_INTERVAL}"
  --sample_freq "${EVAL_INTERVAL}"
  --save_freq "${SAVE_INTERVAL}"
  --learning_rate "${LEARNING_RATE}"
  --batch_size "${BATCH_SIZE}"
  --dataset_infos_path "${DATASET_INFOS_PATH}"
  --value_of_states_path "${VALUE_OF_STATES_PATH}"
  --tag "${EXPERIMENT_NAME}"
  --use_counterfactual_credit "${USE_COUNTERFACTUAL_CREDIT}"
  --counterfactual_k "${COUNTERFACTUAL_K}"
  --credit_weight_mode "${CREDIT_WEIGHT_MODE}"
  --credit_weight_norm "${CREDIT_WEIGHT_NORM}"
  --credit_weight_alpha "${CREDIT_WEIGHT_ALPHA}"
  --credit_weight_min "${CREDIT_WEIGHT_MIN}"
  --credit_weight_max "${CREDIT_WEIGHT_MAX}"
  --credit_weight_on_diffusion "${CREDIT_WEIGHT_ON_DIFFUSION}"
  --credit_weight_on_contrast "${CREDIT_WEIGHT_ON_CONTRAST}"
)

if [ -n "${EXTRA_ARGS}" ]; then
  # shellcheck disable=SC2206
  PY_ARGS+=(${EXTRA_ARGS})
fi

"${PYTHON_BIN}" -u "${PY_ARGS[@]}"

echo "=== ContraDiff ideaA done ==="

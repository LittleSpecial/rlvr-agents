#!/bin/bash
# shellcheck shell=bash

set -eo pipefail
trap 'rc=$?; echo "[ERR] contradiff_value_common.sh failed at line ${LINENO} (rc=${rc})" >&2' ERR

echo "=== Value job started at $(date) ==="
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

echo "=== Python dependency check (value training) ==="
"${PYTHON_BIN}" - <<'PY'
import importlib.util
import os
import sys

use_just = os.environ.get("USE_JUST_D4RL_BACKEND", "1") == "1"
mods = [
    "gym",
    "numpy",
    "scipy",
    "sklearn",
    "h5py",
    "einops",
    "tap",
    "wandb",
    "matplotlib",
    "git",
]
if use_just:
    mods.append("just_d4rl")
else:
    mods += ["d4rl", "mujoco_py"]
missing = [m for m in mods if importlib.util.find_spec(m) is None]
if missing:
    py = sys.executable
    raise SystemExit(
        "[ERROR] Missing python deps in runtime env: "
        + ", ".join(missing)
        + "\nInstall with:\n"
        + f"  {py} -m pip install --no-user gym==0.26.2 numpy scipy scikit-learn h5py just-d4rl "
        + "einops typed-argument-parser wandb matplotlib gitpython"
    )
print("python deps: OK")
PY

CONTRADIFF_DIR="${CONTRADIFF_DIR:-${SLURM_SUBMIT_DIR:-$PWD}/contradiff}"
if [ ! -d "${CONTRADIFF_DIR}/main" ]; then
  echo "[ERR] CONTRADIFF_DIR is invalid: ${CONTRADIFF_DIR}" >&2
  exit 2
fi

VALUE_BRANCH="${VALUE_BRANCH:-plan1_diffuser}"
DATASET="${DATASET:-hopper-random-v2}"
EXP_DATASET="${EXP_DATASET:-expert}"
EXPERT_RATIO="${EXPERT_RATIO:-0.2}"
HORIZON="${HORIZON:-32}"
N_DIFFUSION_STEPS="${N_DIFFUSION_STEPS:-20}"
SEED="${SEED:-1000}"
DEVICE="${DEVICE:-cuda:0}"
VALUE_STEPS="${VALUE_STEPS:-200000}"
LOG_INTERVAL="${LOG_INTERVAL:-1000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
VALUE_BATCH_SIZE="${VALUE_BATCH_SIZE:-32}"
VALUE_RENDERER="${VALUE_RENDERER:-utils.NullRenderer}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-}"

LOGBASE_ROOT="${LOGBASE_ROOT:-${CONTRADIFF_DIR}/main/logs_runs}"
if [ -z "${RUN_LOGBASE:-}" ]; then
  if [ -z "${EXPERIMENT_NAME}" ]; then
    echo "[ERR] Please set either RUN_LOGBASE (absolute path) or EXPERIMENT_NAME." >&2
    exit 2
  fi
  RUN_LOGBASE="${LOGBASE_ROOT}/${EXPERIMENT_NAME}"
fi
if [ -d "$(dirname "${RUN_LOGBASE}")" ]; then
  RUN_LOGBASE="$(cd "$(dirname "${RUN_LOGBASE}")" && pwd)/$(basename "${RUN_LOGBASE}")"
fi
mkdir -p "${RUN_LOGBASE}"

echo "=== ContraDiff value run config ==="
echo "CONTRADIFF_DIR=${CONTRADIFF_DIR}"
echo "RUN_LOGBASE=${RUN_LOGBASE}"
echo "VALUE_BRANCH=${VALUE_BRANCH} DATASET=${DATASET}"
echo "EXP_DATASET=${EXP_DATASET} EXPERT_RATIO=${EXPERT_RATIO}"
echo "HORIZON=${HORIZON} N_DIFFUSION_STEPS=${N_DIFFUSION_STEPS}"
echo "SEED=${SEED} DEVICE=${DEVICE}"
echo "VALUE_STEPS=${VALUE_STEPS} LOG_INTERVAL=${LOG_INTERVAL} SAVE_INTERVAL=${SAVE_INTERVAL}"
echo "LEARNING_RATE=${LEARNING_RATE} VALUE_BATCH_SIZE=${VALUE_BATCH_SIZE}"
echo "VALUE_RENDERER=${VALUE_RENDERER}"
echo "EXTRA_ARGS=${EXTRA_ARGS}"

export VALUE_RENDERER
export N_DIFFUSION_STEPS
export VALUE_STEPS
export SAVE_INTERVAL
export LEARNING_RATE
export VALUE_BATCH_SIZE
export LOG_INTERVAL

cd "${CONTRADIFF_DIR}"

PY_ARGS=(
  main/train_values.py
  --branch "${VALUE_BRANCH}"
  --valuebranch "${VALUE_BRANCH}"
  --dataset "${DATASET}"
  --exp_dataset "${EXP_DATASET}"
  --expert_ratio "${EXPERT_RATIO}"
  --horizon "${HORIZON}"
  --seed "${SEED}"
  --device "${DEVICE}"
  --log_freq "${LOG_INTERVAL}"
  --batch_size "${VALUE_BATCH_SIZE}"
  --logbase "${RUN_LOGBASE}"
)

if [ -n "${EXTRA_ARGS}" ]; then
  # shellcheck disable=SC2206
  PY_ARGS+=(${EXTRA_ARGS})
fi

"${PYTHON_BIN}" -u "${PY_ARGS[@]}"

echo "=== ContraDiff value training done ==="

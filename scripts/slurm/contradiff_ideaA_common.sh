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

if [ -n "${MUJOCO_PY_MUJOCO_PATH:-}" ]; then
  export LD_LIBRARY_PATH="${MUJOCO_PY_MUJOCO_PATH}/bin:${LD_LIBRARY_PATH:-}"
  echo "MUJOCO_PY_MUJOCO_PATH=${MUJOCO_PY_MUJOCO_PATH}"
fi

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
    if use_just:
        fix_hint = (
            f"  {py} -m pip install --no-user gym==0.26.2 numpy scipy scikit-learn h5py just-d4rl "
            "einops typed-argument-parser wandb matplotlib gitpython\n"
            "Or (if you intentionally installed to user-site): set PYTHONNOUSERSITE=0 when submitting."
        )
    else:
        fix_hint = (
            f"  {py} -m pip install --no-user gym==0.23.1 numpy scipy scikit-learn h5py mujoco-py==2.1.2.14 "
            "einops typed-argument-parser wandb matplotlib gitpython\n"
            f"  {py} -m pip install --no-user --no-deps 'd4rl @ git+https://github.com/Farama-Foundation/D4RL.git'\n"
            "And set MUJOCO_PY_MUJOCO_PATH (e.g., $HOME/.mujoco/mujoco210)."
        )
    raise SystemExit(
        "[ERROR] Missing python deps in runtime env: "
        + ", ".join(missing)
        + "\nInstall with:\n"
        + fix_hint
    )
print("python deps: OK")
PY

if [ "${USE_JUST_D4RL_BACKEND:-1}" != "1" ]; then
  echo "=== D4RL env registration check ==="
  "${PYTHON_BIN}" - <<'PY'
import gym
import d4rl  # noqa: F401

name = "hopper-random-v2"
try:
    env = gym.make(name)
except Exception as e:
    raise SystemExit(
        f"[ERROR] gym.make('{name}') failed: {type(e).__name__}: {e}\n"
        "Likely missing MuJoCo runtime / mujoco_py compatibility.\n"
        "Set MUJOCO_PY_MUJOCO_PATH (e.g., $HOME/.mujoco/mujoco210) and ensure "
        "mujoco_py is installed in the same Python env."
    )
print("d4rl env check: OK", name, "max_episode_steps=", getattr(env, "_max_episode_steps", "NA"))
PY
else
  echo "=== just-d4rl mode: skip gym.make(d4rl-env) check ==="
fi

CONTRADIFF_DIR="${CONTRADIFF_DIR:-${SLURM_SUBMIT_DIR:-$PWD}/contradiff}"
if [ ! -d "${CONTRADIFF_DIR}/main" ]; then
  echo "[ERR] CONTRADIFF_DIR is invalid: ${CONTRADIFF_DIR}" >&2
  exit 2
fi

echo "=== ContraDiff dataset loader check ==="
CONTRADIFF_DIR="${CONTRADIFF_DIR}" DATASET="${DATASET:-hopper-random-v2}" USE_JUST_D4RL_BACKEND="${USE_JUST_D4RL_BACKEND:-1}" \
"${PYTHON_BIN}" - <<'PY'
import importlib.util
import os

repo = os.environ["CONTRADIFF_DIR"]
dataset_file = os.path.join(repo, "main", "diffuser", "datasets", "d4rl.py")
spec = importlib.util.spec_from_file_location("contradiff_dataset_d4rl", dataset_file)
if spec is None or spec.loader is None:
    raise SystemExit(f"[ERROR] Cannot load dataset loader file: {dataset_file}")
d4rl_dataset = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d4rl_dataset)

if not hasattr(d4rl_dataset, "load_environment"):
    raise SystemExit(
        "[ERROR] Invalid ContraDiff dataset loader module (missing load_environment). "
        "Please sync contradiff/main/diffuser/datasets/d4rl.py."
    )

if os.environ.get("USE_JUST_D4RL_BACKEND", "1") == "1":
    name = os.environ.get("DATASET", "hopper-random-v2")
    env = d4rl_dataset.load_environment(name)
    print(
        "dataset loader probe: OK",
        name,
        "env_type=",
        type(env).__name__,
        "max_steps=",
        getattr(env, "_max_episode_steps", "NA"),
    )
else:
    print("dataset loader probe: OK (full d4rl mode)")
PY

BRANCH="${BRANCH:-plan2_hard}"
DATASET="${DATASET:-hopper-random-v2}"
RENDERER="${RENDERER:-utils.NullRenderer}"
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
LOGBASE_ROOT="${LOGBASE_ROOT:-${CONTRADIFF_DIR}/main/logs_runs}"
RUN_LOGBASE="${RUN_LOGBASE:-${LOGBASE_ROOT}/${EXPERIMENT_NAME}}"
mkdir -p "${RUN_LOGBASE}"

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
USE_DATAPARALLEL="${USE_DATAPARALLEL:-1}"
export NUM_GPUS
export USE_DATAPARALLEL
if [ "${NUM_GPUS}" -gt 1 ]; then
  if [ "${USE_DATAPARALLEL}" = "1" ]; then
    echo "[INFO] Multi-GPU enabled via torch.nn.DataParallel (single-process), NUM_GPUS=${NUM_GPUS}."
    if [ "${BATCH_SIZE}" -lt "${NUM_GPUS}" ]; then
      echo "[WARN] BATCH_SIZE=${BATCH_SIZE} < NUM_GPUS=${NUM_GPUS}; some GPUs may be under-utilized."
    fi
  else
    echo "[WARN] USE_DATAPARALLEL=${USE_DATAPARALLEL}; job will still run single-process on DEVICE=${DEVICE}."
  fi
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
echo "RENDERER=${RENDERER} USE_JUST_D4RL_BACKEND=${USE_JUST_D4RL_BACKEND:-1}"
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
echo "RUN_LOGBASE=${RUN_LOGBASE}"
echo "EXTRA_ARGS=${EXTRA_ARGS}"

export RENDERER
export N_DIFFUSION_STEPS
export MAX_STEPS
export EVAL_INTERVAL
export SAVE_INTERVAL
export LEARNING_RATE
export DATASET_INFOS_PATH
export VALUE_OF_STATES_PATH
export USE_COUNTERFACTUAL_CREDIT
export COUNTERFACTUAL_K
export CREDIT_WEIGHT_MODE
export CREDIT_WEIGHT_NORM
export CREDIT_WEIGHT_ALPHA
export CREDIT_WEIGHT_MIN
export CREDIT_WEIGHT_MAX
export CREDIT_WEIGHT_ON_DIFFUSION
export CREDIT_WEIGHT_ON_CONTRAST

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
  --seed "${SEED}"
  --device "${DEVICE}"
  --log_freq "${LOG_INTERVAL}"
  --batch_size "${BATCH_SIZE}"
  --logbase "${RUN_LOGBASE}"
  --tag "${EXPERIMENT_NAME}"
)

if [ -n "${EXTRA_ARGS}" ]; then
  # shellcheck disable=SC2206
  PY_ARGS+=(${EXTRA_ARGS})
fi

"${PYTHON_BIN}" -u "${PY_ARGS[@]}"

echo "=== ContraDiff ideaA done ==="

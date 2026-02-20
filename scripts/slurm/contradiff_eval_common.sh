#!/bin/bash
# shellcheck shell=bash

set -eo pipefail
trap 'rc=$?; echo "[ERR] contradiff_eval_common.sh failed at line ${LINENO} (rc=${rc})" >&2' ERR

echo "=== Eval job started at $(date) ==="
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

STRICT_REAL_EVAL="${STRICT_REAL_EVAL:-0}"
EVAL_BACKEND="${EVAL_BACKEND:-auto}"
export STRICT_REAL_EVAL
export EVAL_BACKEND
export CONTRADIFF_EVAL_BACKEND="${EVAL_BACKEND}"

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

echo "=== Python dependency check (eval) ==="
"${PYTHON_BIN}" - <<'PY'
import importlib.util
import os
import sys

strict_real_eval = os.environ.get("STRICT_REAL_EVAL", "0") == "1"
eval_backend = os.environ.get("EVAL_BACKEND", "auto").strip().lower()
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
if strict_real_eval:
    if eval_backend in ("gymnasium", "gymnasium_mujoco", "mujoco"):
        mods += ["gymnasium", "mujoco"]
    else:
        mods += ["d4rl", "mujoco_py"]
missing = [m for m in mods if importlib.util.find_spec(m) is None]
if missing:
    py = sys.executable
    if strict_real_eval and eval_backend in ("gymnasium", "gymnasium_mujoco", "mujoco"):
        mode = "gymnasium+mujoco online evaluation"
    elif strict_real_eval:
        mode = "legacy d4rl+mujoco_py online evaluation"
    else:
        mode = "proxy evaluation"
    raise SystemExit(
        f"[ERROR] Missing python deps for {mode}: "
        + ", ".join(missing)
        + "\nInstall (example):\n"
        + f"  {py} -m pip install --no-user gym==0.23.1 numpy scipy scikit-learn h5py "
        + "einops typed-argument-parser wandb matplotlib gitpython\n"
        + (
            (
                f"  {py} -m pip install --no-user 'gymnasium[mujoco]' mujoco"
                if eval_backend in ("gymnasium", "gymnasium_mujoco", "mujoco")
                else (
                    f"  {py} -m pip install --no-user mujoco-py==2.1.2.14\n"
                    f"  {py} -m pip install --no-user --no-deps 'd4rl @ git+https://github.com/Farama-Foundation/D4RL.git'"
                )
            )
            if strict_real_eval
            else
            "  # d4rl+mujoco_py are optional in proxy mode"
        )
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
EVAL_SEED="${EVAL_SEED:-1000}"
DEVICE="${DEVICE:-cuda:0}"
LOAD_ITER="${LOAD_ITER:--1}"
NUMS_EVAL="${NUMS_EVAL:-50}"
GUIDE_SCALE="${GUIDE_SCALE:--1}"
BATCH_SIZE="${BATCH_SIZE:-64}"
VALUEBRANCH="${VALUEBRANCH:-plan1_diffuser}"
VALUESEED="${VALUESEED:-1000}"
SAVE_PLANNED="${SAVE_PLANNED:-0}"
SAVE_DIFFUSION="${SAVE_DIFFUSION:-0}"
STRICT_REAL_EVAL="${STRICT_REAL_EVAL:-0}"
EVAL_BACKEND="${EVAL_BACKEND:-auto}"
ALLOW_VALUE_FALLBACK="${ALLOW_VALUE_FALLBACK:-0}"
export ALLOW_VALUE_FALLBACK
export CONTRADIFF_EVAL_BACKEND="${EVAL_BACKEND}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-}"
LOGBASE_ROOT="${LOGBASE_ROOT:-${CONTRADIFF_DIR}/main/logs_runs}"
if [ -z "${RUN_LOGBASE:-}" ]; then
  if [ -z "${EXPERIMENT_NAME}" ]; then
    echo "[ERR] Please set either RUN_LOGBASE (absolute path) or EXPERIMENT_NAME." >&2
    echo "Example: EXPERIMENT_NAME=cdiff_base_hopper_rand02_user_12345" >&2
    exit 2
  fi
  RUN_LOGBASE="${LOGBASE_ROOT}/${EXPERIMENT_NAME}"
fi
if [ -d "$(dirname "${RUN_LOGBASE}")" ]; then
  RUN_LOGBASE="$(cd "$(dirname "${RUN_LOGBASE}")" && pwd)/$(basename "${RUN_LOGBASE}")"
fi

VALBASE="${VALBASE:-${RUN_LOGBASE}}"
export VALBASE

if [ ! -d "${RUN_LOGBASE}/${BRANCH}/${DATASET}" ]; then
  echo "[ERR] Run logbase not found: ${RUN_LOGBASE}/${BRANCH}/${DATASET}" >&2
  exit 2
fi

CKPT_COUNT="$(find "${RUN_LOGBASE}/${BRANCH}/${DATASET}" -type f -name 'state_*.pt' | wc -l | tr -d ' ')"
if [ "${CKPT_COUNT}" = "0" ]; then
  echo "[ERR] No diffusion checkpoints found under ${RUN_LOGBASE}/${BRANCH}/${DATASET}" >&2
  exit 2
fi

if [ ! -d "${VALBASE}" ]; then
  if [ "${ALLOW_VALUE_FALLBACK}" = "1" ]; then
    echo "[WARN] VALBASE does not exist (${VALBASE}); value-model fallback will be used."
  else
    echo "[ERR] VALBASE does not exist: ${VALBASE}" >&2
    echo "Set VALBASE to your value-function checkpoint root, or ALLOW_VALUE_FALLBACK=1." >&2
    exit 2
  fi
fi

if [ "${STRICT_REAL_EVAL}" = "1" ]; then
  if [ "${EVAL_BACKEND}" = "gymnasium" ] || [ "${EVAL_BACKEND}" = "gymnasium_mujoco" ] || [ "${EVAL_BACKEND}" = "mujoco" ]; then
    echo "=== Real env check (gymnasium+mujoco backend) ==="
    CONTRADIFF_DIR="${CONTRADIFF_DIR}" DATASET="${DATASET}" CONTRADIFF_EVAL_BACKEND="${EVAL_BACKEND}" "${PYTHON_BIN}" - <<'PY'
import os
import sys

repo = os.environ["CONTRADIFF_DIR"]
name = os.environ["DATASET"]
sys.path.insert(0, os.path.join(repo, "main"))

from diffuser.datasets.d4rl import load_environment  # noqa: E402

env = load_environment(name)
missing = [m for m in ("reset", "step", "get_normalized_score") if not hasattr(env, m)]
if missing:
    raise SystemExit(
        f"[ERROR] Environment missing required API: {missing}. "
        "This is not a valid online rollout env."
    )
print("real env check: OK", name, "backend=gymnasium", "max_episode_steps=", getattr(env, "_max_episode_steps", "NA"))
PY
  else
    echo "=== Real env check (legacy d4rl+mujoco_py backend) ==="
    DATASET="${DATASET}" "${PYTHON_BIN}" - <<'PY'
import os
import gym
name = os.environ["DATASET"]

try:
    import d4rl  # noqa: F401
except Exception as e:
    raise SystemExit(f"[ERROR] import d4rl failed: {type(e).__name__}: {e}")

try:
    env = gym.make(name)
except Exception as e:
    raise SystemExit(
        f"[ERROR] gym.make({name}) failed: {type(e).__name__}: {e}\n"
        "Legacy paper-style evaluation requires a D4RL MuJoCo env."
    )

missing = [m for m in ("reset", "step", "get_normalized_score") if not hasattr(env, m)]
if missing:
    raise SystemExit(
        f"[ERROR] Environment missing required API: {missing}. "
        "This is not a valid D4RL rollout env."
    )

print("real env check: OK", name, "backend=d4rl", "max_episode_steps=", getattr(env, "_max_episode_steps", "NA"))
PY
  fi
fi

echo "=== ContraDiff eval config ==="
echo "CONTRADIFF_DIR=${CONTRADIFF_DIR}"
echo "RUN_LOGBASE=${RUN_LOGBASE}"
echo "VALBASE=${VALBASE}"
echo "BRANCH=${BRANCH} DATASET=${DATASET}"
echo "LOAD_ITER=${LOAD_ITER} NUMS_EVAL=${NUMS_EVAL}"
echo "SEED=${SEED} EVAL_SEED=${EVAL_SEED} DEVICE=${DEVICE}"
echo "EXP_DATASET=${EXP_DATASET} EXPERT_RATIO=${EXPERT_RATIO}"
echo "LOWERBOUND=${LOWERBOUND} UPPERBOUND=${UPPERBOUND} METRICS=${METRICS}"
echo "HORIZON=${HORIZON} N_DIFFUSION_STEPS=${N_DIFFUSION_STEPS}"
echo "VALUEBRANCH=${VALUEBRANCH} VALUESEED=${VALUESEED}"
echo "GUIDE_SCALE=${GUIDE_SCALE} BATCH_SIZE=${BATCH_SIZE}"
echo "SAVE_PLANNED=${SAVE_PLANNED} SAVE_DIFFUSION=${SAVE_DIFFUSION}"
echo "STRICT_REAL_EVAL=${STRICT_REAL_EVAL} EVAL_BACKEND=${EVAL_BACKEND} ALLOW_VALUE_FALLBACK=${ALLOW_VALUE_FALLBACK}"
echo "EXTRA_ARGS=${EXTRA_ARGS}"

cd "${CONTRADIFF_DIR}"

PY_ARGS=(
  main/plan_guided.py
  --branch "${BRANCH}"
  --dataset "${DATASET}"
  --seed "${SEED}"
  --evalseed "${EVAL_SEED}"
  --device "${DEVICE}"
  --load_iter "${LOAD_ITER}"
  --nums_eval "${NUMS_EVAL}"
  --logbase "${RUN_LOGBASE}"
  --exp_dataset "${EXP_DATASET}"
  --expert_ratio "${EXPERT_RATIO}"
  --lowerbound "${LOWERBOUND}"
  --upperbound "${UPPERBOUND}"
  --horizon "${HORIZON}"
  --n_diffusion_steps "${N_DIFFUSION_STEPS}"
  --metrics "${METRICS}"
  --valuebranch "${VALUEBRANCH}"
  --valueseed "${VALUESEED}"
  --guide_scale "${GUIDE_SCALE}"
  --batch_size "${BATCH_SIZE}"
  --save_planned "${SAVE_PLANNED}"
  --save_diffusion "${SAVE_DIFFUSION}"
)

if [ -n "${EXTRA_ARGS}" ]; then
  # shellcheck disable=SC2206
  PY_ARGS+=(${EXTRA_ARGS})
fi

"${PYTHON_BIN}" -u "${PY_ARGS[@]}"

echo "=== ContraDiff eval done ==="

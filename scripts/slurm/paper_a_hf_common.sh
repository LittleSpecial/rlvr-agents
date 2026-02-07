#!/bin/bash
# shellcheck shell=bash

set -eo pipefail
trap 'rc=$?; echo "[ERR] paper_a_hf_common.sh failed at line ${LINENO} (rc=${rc})" >&2' ERR

echo "=== Job started at $(date) ==="
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: $(hostname)"
echo "PWD: $(pwd)"
echo "SLURM_SUBMIT_DIR: ${SLURM_SUBMIT_DIR:-$PWD}"

mkdir -p logs experiments
cd "${SLURM_SUBMIT_DIR:-$PWD}"

export TMPDIR="${TMPDIR:-$HOME/tmp}"
mkdir -p "${TMPDIR}"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_DDP_TIMEOUT_SECONDS="${TORCH_DDP_TIMEOUT_SECONDS:-3600}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

# Keep all ranks in lock-step around rank0 eval/checkpoint by default.
export SYNC_EVAL_AND_SAVE="${SYNC_EVAL_AND_SAVE:-1}"
# Truncate per-rank samples to global min to reduce step-time straggler.
export TRUNCATE_TO_GLOBAL_MIN_SAMPLES="${TRUNCATE_TO_GLOBAL_MIN_SAMPLES:-1}"

# Threads
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

# Offline mode on compute nodes
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: ${SLURMD_NODENAME:-$(hostname)}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES:-N/A}"
echo "PYTHON_BIN: ${PYTHON_BIN}"
echo "=========================================="

echo "=== nvidia-smi ==="
nvidia-smi || true

"${PYTHON_BIN}" - <<'PY'
import sys
try:
    import torch
except ImportError as e:
    raise SystemExit(
        f"\n[ERROR] torch import failed: {e}\n"
        "Common cause: torch wheel is +cu118 but job loaded mismatched cuda module.\n"
    )
print("python:", sys.version.replace("\n", " "))
print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("cuda.is_available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit("\n[ERROR] torch.cuda.is_available() == False\n")
PY

echo "=== HF stack versions (best-effort) ==="
"${PYTHON_BIN}" - <<'PY'
import importlib

def v(name):
    try:
        m = importlib.import_module(name)
        return getattr(m, "__version__", "unknown")
    except Exception as e:
        return f"ERR({type(e).__name__})"

for pkg in ["numpy", "transformers", "peft", "datasets"]:
    print(f"{pkg}:", v(pkg))
PY

MODEL_PATH="${MODEL_PATH:-/path/to/local/Qwen2.5-7B-Instruct}"
TRAIN_DATA="${TRAIN_DATA:-datasets/code/mbpp_train.jsonl}"
EVAL_DATA="${EVAL_DATA:-datasets/code/humaneval_test.jsonl}"

NUM_GPUS="${NUM_GPUS:-${SLURM_GPUS_ON_NODE:-${DEFAULT_NUM_GPUS:-1}}}"
if ! [[ "${NUM_GPUS}" =~ ^[0-9]+$ ]]; then
  NUM_GPUS="$(echo "${NUM_GPUS}" | grep -oE '[0-9]+' | head -n1)"
fi
if ! [[ "${NUM_GPUS}" =~ ^[0-9]+$ ]] || [ "${NUM_GPUS}" -lt 1 ]; then
  echo "[ERR] Invalid NUM_GPUS=${NUM_GPUS}" >&2
  exit 2
fi

EXPERIMENT_NAME="${EXPERIMENT_NAME:-paper_a_hf_${SLURM_JOB_ID:-local}}"
ENV_TYPE="${ENV_TYPE:-code}"
SHOW_TESTS="${SHOW_TESTS:-1}"
DTYPE="${DTYPE:-bf16}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_ROLLOUTS_PER_PROMPT="${NUM_ROLLOUTS_PER_PROMPT:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-1024}"
MAX_TRAJECTORY_LENGTH="${MAX_TRAJECTORY_LENGTH:-20}"
MAX_POLICY_TURNS="${MAX_POLICY_TURNS:-1}"
TASK_TIMEOUT_SECONDS="${TASK_TIMEOUT_SECONDS:-8}"
MAX_STEPS="${MAX_STEPS:-50}"
LOG_INTERVAL="${LOG_INTERVAL:-5}"
EVAL_INTERVAL="${EVAL_INTERVAL:-25}"
SAVE_INTERVAL="${SAVE_INTERVAL:-0}"
EVAL_SAMPLE_K="${EVAL_SAMPLE_K:-${NUM_ROLLOUTS_PER_PROMPT}}"
EVAL_GEN_BATCH_SIZE="${EVAL_GEN_BATCH_SIZE:-8}"
HEARTBEAT_INTERVAL="${HEARTBEAT_INTERVAL:-1}"
USE_COUNTERFACTUAL_CREDIT="${USE_COUNTERFACTUAL_CREDIT:-1}"
PRIORITIZE_HIGH_VALUE_CF="${PRIORITIZE_HIGH_VALUE_CF:-0}"
COUNTERFACTUAL_K="${COUNTERFACTUAL_K:-2}"
REWARD_MODE="${REWARD_MODE:-mixed}"
REWARD_BLEND_ALPHA="${REWARD_BLEND_ALPHA:-0.5}"
FLAT_GROUP_FALLBACK="${FLAT_GROUP_FALLBACK:-raw}"
FAILURE_REWARD_FLOOR="${FAILURE_REWARD_FLOOR:--0.01}"
CREDIT_FALLBACK_WHEN_ZERO_ADV="${CREDIT_FALLBACK_WHEN_ZERO_ADV:-1}"
FALLBACK_TO_ADV_WHEN_ZERO_CREDIT="${FALLBACK_TO_ADV_WHEN_ZERO_CREDIT:-1}"
ZERO_CREDIT_THRESHOLD="${ZERO_CREDIT_THRESHOLD:-1e-8}"
FAILURE_REPLAY_RATIO="${FAILURE_REPLAY_RATIO:-0.25}"
FAILURE_BUFFER_SIZE="${FAILURE_BUFFER_SIZE:-2048}"
FAILURE_REPLAY_WARMUP_STEPS="${FAILURE_REPLAY_WARMUP_STEPS:-20}"
FAILURE_BUFFER_UNIQUE="${FAILURE_BUFFER_UNIQUE:-1}"
REPLAY_MIN_SUCCESS_EMA="${REPLAY_MIN_SUCCESS_EMA:-0.2}"
REPLAY_EMA_ALPHA="${REPLAY_EMA_ALPHA:-0.1}"
GUARD_ALL_NEGATIVE_BATCH="${GUARD_ALL_NEGATIVE_BATCH:-1}"
ALL_NEGATIVE_REWARD_SPAN_THRESHOLD="${ALL_NEGATIVE_REWARD_SPAN_THRESHOLD:-1e-8}"
INTERVENTION_TYPES="${INTERVENTION_TYPES:-delete truncate}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

if [ "${SHOW_TESTS}" = "1" ]; then
  SHOW_TESTS_FLAG="--show_tests"
else
  SHOW_TESTS_FLAG="--no-show_tests"
fi

if [ "${USE_COUNTERFACTUAL_CREDIT}" = "1" ]; then
  CF_FLAG="--use_counterfactual_credit"
else
  CF_FLAG="--no-use_counterfactual_credit"
fi

if [ "${PRIORITIZE_HIGH_VALUE_CF}" = "1" ]; then
  CF_PRIOR_FLAG="--prioritize_high_value_cf"
else
  CF_PRIOR_FLAG="--no-prioritize_high_value_cf"
fi

if [ "${CREDIT_FALLBACK_WHEN_ZERO_ADV}" = "1" ]; then
  CF_ZERO_ADV_FLAG="--credit_fallback_when_zero_adv"
else
  CF_ZERO_ADV_FLAG="--no-credit_fallback_when_zero_adv"
fi

if [ "${FALLBACK_TO_ADV_WHEN_ZERO_CREDIT}" = "1" ]; then
  ZERO_CREDIT_FALLBACK_FLAG="--fallback_to_adv_when_zero_credit"
else
  ZERO_CREDIT_FALLBACK_FLAG="--no-fallback_to_adv_when_zero_credit"
fi

if [ "${FAILURE_BUFFER_UNIQUE}" = "1" ]; then
  FAILURE_BUFFER_UNIQUE_FLAG="--failure_buffer_unique"
else
  FAILURE_BUFFER_UNIQUE_FLAG="--no-failure_buffer_unique"
fi

if [ "${GUARD_ALL_NEGATIVE_BATCH}" = "1" ]; then
  ALL_NEGATIVE_GUARD_FLAG="--guard_all_negative_batch"
else
  ALL_NEGATIVE_GUARD_FLAG="--no-guard_all_negative_batch"
fi

if [ "${SYNC_EVAL_AND_SAVE}" = "1" ]; then
  SYNC_EVAL_FLAG="--sync_eval_and_save"
else
  SYNC_EVAL_FLAG="--no-sync_eval_and_save"
fi

if [ "${TRUNCATE_TO_GLOBAL_MIN_SAMPLES}" = "1" ]; then
  TRUNCATE_FLAG="--truncate_to_global_min_samples"
else
  TRUNCATE_FLAG="--no-truncate_to_global_min_samples"
fi

echo "=== Paper A run config ==="
echo "NUM_GPUS=${NUM_GPUS}"
echo "EXPERIMENT_NAME=${EXPERIMENT_NAME}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "TRAIN_DATA=${TRAIN_DATA}"
echo "EVAL_DATA=${EVAL_DATA}"
echo "DTYPE=${DTYPE} LR=${LEARNING_RATE}"
echo "BATCH_SIZE=${BATCH_SIZE} NUM_ROLLOUTS_PER_PROMPT=${NUM_ROLLOUTS_PER_PROMPT}"
echo "MAX_PROMPT_TOKENS=${MAX_PROMPT_TOKENS} MAX_NEW_TOKENS=${MAX_NEW_TOKENS}"
echo "MAX_TRAJECTORY_LENGTH=${MAX_TRAJECTORY_LENGTH} MAX_POLICY_TURNS=${MAX_POLICY_TURNS}"
echo "TASK_TIMEOUT_SECONDS=${TASK_TIMEOUT_SECONDS}"
echo "MAX_STEPS=${MAX_STEPS} LOG_INTERVAL=${LOG_INTERVAL} EVAL_INTERVAL=${EVAL_INTERVAL} SAVE_INTERVAL=${SAVE_INTERVAL}"
echo "EVAL_SAMPLE_K=${EVAL_SAMPLE_K}"
echo "EVAL_GEN_BATCH_SIZE=${EVAL_GEN_BATCH_SIZE}"
echo "HEARTBEAT_INTERVAL=${HEARTBEAT_INTERVAL}"
echo "USE_COUNTERFACTUAL_CREDIT=${USE_COUNTERFACTUAL_CREDIT} PRIORITIZE_HIGH_VALUE_CF=${PRIORITIZE_HIGH_VALUE_CF}"
echo "COUNTERFACTUAL_K=${COUNTERFACTUAL_K} INTERVENTION_TYPES=${INTERVENTION_TYPES}"
echo "REWARD_MODE=${REWARD_MODE} BLEND_ALPHA=${REWARD_BLEND_ALPHA} FLAT_GROUP_FALLBACK=${FLAT_GROUP_FALLBACK}"
echo "FAILURE_REWARD_FLOOR=${FAILURE_REWARD_FLOOR}"
echo "CREDIT_FALLBACK_WHEN_ZERO_ADV=${CREDIT_FALLBACK_WHEN_ZERO_ADV}"
echo "FALLBACK_TO_ADV_WHEN_ZERO_CREDIT=${FALLBACK_TO_ADV_WHEN_ZERO_CREDIT} ZERO_CREDIT_THRESHOLD=${ZERO_CREDIT_THRESHOLD}"
echo "FAILURE_REPLAY_RATIO=${FAILURE_REPLAY_RATIO} FAILURE_BUFFER_SIZE=${FAILURE_BUFFER_SIZE} WARMUP=${FAILURE_REPLAY_WARMUP_STEPS}"
echo "FAILURE_BUFFER_UNIQUE=${FAILURE_BUFFER_UNIQUE}"
echo "REPLAY_MIN_SUCCESS_EMA=${REPLAY_MIN_SUCCESS_EMA} REPLAY_EMA_ALPHA=${REPLAY_EMA_ALPHA}"
echo "GUARD_ALL_NEGATIVE_BATCH=${GUARD_ALL_NEGATIVE_BATCH} ALL_NEGATIVE_REWARD_SPAN_THRESHOLD=${ALL_NEGATIVE_REWARD_SPAN_THRESHOLD}"
echo "SYNC_EVAL_AND_SAVE=${SYNC_EVAL_AND_SAVE} TRUNCATE_TO_GLOBAL_MIN_SAMPLES=${TRUNCATE_TO_GLOBAL_MIN_SAMPLES}"
echo "EXTRA_ARGS=${EXTRA_ARGS}"

PY_ARGS=(
  paper_a_credit_assignment/train.py
  --experiment_name "${EXPERIMENT_NAME}"
  --backend hf
  --model_path "${MODEL_PATH}"
  --env_type "${ENV_TYPE}"
  ${SHOW_TESTS_FLAG}
  --require_cuda
  --train_dataset "${TRAIN_DATA}"
  --eval_dataset "${EVAL_DATA}"
  --dtype "${DTYPE}"
  --learning_rate "${LEARNING_RATE}"
  --batch_size "${BATCH_SIZE}"
  --num_rollouts_per_prompt "${NUM_ROLLOUTS_PER_PROMPT}"
  --max_new_tokens "${MAX_NEW_TOKENS}"
  --max_prompt_tokens "${MAX_PROMPT_TOKENS}"
  --max_trajectory_length "${MAX_TRAJECTORY_LENGTH}"
  --max_policy_turns "${MAX_POLICY_TURNS}"
  --task_timeout_seconds "${TASK_TIMEOUT_SECONDS}"
  --max_steps "${MAX_STEPS}"
  --log_interval "${LOG_INTERVAL}"
  --eval_interval "${EVAL_INTERVAL}"
  --save_interval "${SAVE_INTERVAL}"
  --eval_sample_k "${EVAL_SAMPLE_K}"
  --eval_gen_batch_size "${EVAL_GEN_BATCH_SIZE}"
  --heartbeat_interval "${HEARTBEAT_INTERVAL}"
  ${SYNC_EVAL_FLAG}
  ${TRUNCATE_FLAG}
  ${CF_FLAG}
  ${CF_PRIOR_FLAG}
  ${CF_ZERO_ADV_FLAG}
  ${ZERO_CREDIT_FALLBACK_FLAG}
  --zero_credit_threshold "${ZERO_CREDIT_THRESHOLD}"
  --counterfactual_k "${COUNTERFACTUAL_K}"
  --reward_mode "${REWARD_MODE}"
  --reward_blend_alpha "${REWARD_BLEND_ALPHA}"
  --failure_reward_floor "${FAILURE_REWARD_FLOOR}"
  --flat_group_fallback "${FLAT_GROUP_FALLBACK}"
  --failure_replay_ratio "${FAILURE_REPLAY_RATIO}"
  --failure_buffer_size "${FAILURE_BUFFER_SIZE}"
  --failure_replay_warmup_steps "${FAILURE_REPLAY_WARMUP_STEPS}"
  ${FAILURE_BUFFER_UNIQUE_FLAG}
  --replay_min_success_ema "${REPLAY_MIN_SUCCESS_EMA}"
  --replay_ema_alpha "${REPLAY_EMA_ALPHA}"
  ${ALL_NEGATIVE_GUARD_FLAG}
  --all_negative_reward_span_threshold "${ALL_NEGATIVE_REWARD_SPAN_THRESHOLD}"
  --intervention_types ${INTERVENTION_TYPES}
  --output_dir ./experiments
)

if [ -n "${EXTRA_ARGS}" ]; then
  # shellcheck disable=SC2206
  PY_ARGS+=(${EXTRA_ARGS})
fi

if [ "${NUM_GPUS}" -gt 1 ]; then
  "${PYTHON_BIN}" -u -m torch.distributed.run --standalone --nproc_per_node="${NUM_GPUS}" "${PY_ARGS[@]}"
else
  "${PYTHON_BIN}" -u "${PY_ARGS[@]}"
fi

echo "=== Paper A HF done ==="

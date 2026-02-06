#!/bin/bash
#SBATCH --job-name=paper_b_hf
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --output=logs/paper_b_hf_%j.out
#SBATCH --error=logs/paper_b_hf_%j.err
#SBATCH --export=ALL

set -eo pipefail
trap 'rc=$?; echo "[ERR] run_paper_b_hf_1gpu.sh failed at line ${LINENO} (rc=${rc})" >&2' ERR

mkdir -p logs experiments
cd "${SLURM_SUBMIT_DIR:-$PWD}"

export TMPDIR="${TMPDIR:-$HOME/tmp}"
mkdir -p "${TMPDIR}"
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

module purge
module load miniforge3/24.1

CONDA_ENV="${CONDA_ENV:-$HOME/.conda/envs/rlvr}"
if [ -d "${CONDA_ENV}" ]; then
  source activate "${CONDA_ENV}" 2>/dev/null || true
fi

if [ -z "${PYTHON_BIN:-}" ] && [ -x "${CONDA_ENV}/bin/python3" ]; then
  PYTHON_BIN="${CONDA_ENV}/bin/python3"
fi
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || true)}"
if [ -z "${PYTHON_BIN}" ]; then
  echo "[ERR] python3 not found." >&2
  exit 2
fi

module load "${GCC_MODULE:-compilers/gcc/11.3.0}"
module load "${CUDA_MODULE:-compilers/cuda/11.8}"
module load "${CUDNN_MODULE:-cudnn/8.6.0.163_cuda11.x}"
if [ -n "${NCCL_MODULE:-}" ]; then
  module load "${NCCL_MODULE}" 2>/dev/null || true
fi

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

MODEL_PATH="${MODEL_PATH:-/path/to/local/Qwen2.5-7B-Instruct}"
TRAIN_DATA="${TRAIN_DATA:-datasets/code/mbpp_train.jsonl}"
EVAL_DATA="${EVAL_DATA:-datasets/code/humaneval_test.jsonl}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-paper_b_hf_${SLURM_JOB_ID}}"
ENV_TYPE="${ENV_TYPE:-code}"
SHOW_TESTS="${SHOW_TESTS:-1}"
DTYPE="${DTYPE:-bf16}"
LEARNING_RATE="${LEARNING_RATE:-5e-6}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_ROLLOUTS_PER_PROMPT="${NUM_ROLLOUTS_PER_PROMPT:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-320}"
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-1024}"
MAX_STEPS="${MAX_STEPS:-300}"
LOG_INTERVAL="${LOG_INTERVAL:-20}"
EVAL_INTERVAL="${EVAL_INTERVAL:-100}"
SAVE_INTERVAL="${SAVE_INTERVAL:-0}"
PROTOCOL="${PROTOCOL:-sequential}"
SKILL_KEY="${SKILL_KEY:-task_type}"
SKILL_SEQUENCE="${SKILL_SEQUENCE:-}"
STAGE_STEPS="${STAGE_STEPS:-200}"
PROJECTION="${PROJECTION:-sequential_margin}"
EPSILON="${EPSILON:-0.0}"
MEMORY_PER_PROTECTED="${MEMORY_PER_PROTECTED:-2}"
MAX_PROTECTED_SKILLS="${MAX_PROTECTED_SKILLS:-2}"
REWARD_MODE="${REWARD_MODE:-mixed}"
REWARD_BLEND_ALPHA="${REWARD_BLEND_ALPHA:-0.7}"
FAILURE_REWARD_FLOOR="${FAILURE_REWARD_FLOOR:--0.01}"
FLAT_GROUP_FALLBACK="${FLAT_GROUP_FALLBACK:-raw}"
GRAD_PARAM_INCLUDE="${GRAD_PARAM_INCLUDE:-lora_ adapter}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

if [ "${SHOW_TESTS}" = "1" ]; then
  SHOW_TESTS_FLAG="--show_tests"
else
  SHOW_TESTS_FLAG="--no-show_tests"
fi

PY_ARGS=(
  paper_b_conflict_aware/train.py
  --experiment_name "${EXPERIMENT_NAME}"
  --backend hf
  --model_path "${MODEL_PATH}"
  --env_type "${ENV_TYPE}"
  ${SHOW_TESTS_FLAG}
  --dtype "${DTYPE}"
  --learning_rate "${LEARNING_RATE}"
  --batch_size "${BATCH_SIZE}"
  --num_rollouts_per_prompt "${NUM_ROLLOUTS_PER_PROMPT}"
  --max_new_tokens "${MAX_NEW_TOKENS}"
  --max_prompt_tokens "${MAX_PROMPT_TOKENS}"
  --max_steps "${MAX_STEPS}"
  --log_interval "${LOG_INTERVAL}"
  --eval_interval "${EVAL_INTERVAL}"
  --save_interval "${SAVE_INTERVAL}"
  --train_dataset "${TRAIN_DATA}"
  --eval_dataset "${EVAL_DATA}"
  --protocol "${PROTOCOL}"
  --skill_key "${SKILL_KEY}"
  --stage_steps "${STAGE_STEPS}"
  --projection "${PROJECTION}"
  --epsilon "${EPSILON}"
  --memory_per_protected "${MEMORY_PER_PROTECTED}"
  --max_protected_skills "${MAX_PROTECTED_SKILLS}"
  --reward_mode "${REWARD_MODE}"
  --reward_blend_alpha "${REWARD_BLEND_ALPHA}"
  --failure_reward_floor "${FAILURE_REWARD_FLOOR}"
  --flat_group_fallback "${FLAT_GROUP_FALLBACK}"
  --grad_param_include ${GRAD_PARAM_INCLUDE}
  --output_dir ./experiments
)

if [ -n "${SKILL_SEQUENCE}" ]; then
  # shellcheck disable=SC2206
  _SEQ=(${SKILL_SEQUENCE})
  PY_ARGS+=(--skill_sequence "${_SEQ[@]}")
fi

if [ -n "${EXTRA_ARGS}" ]; then
  # shellcheck disable=SC2206
  PY_ARGS+=(${EXTRA_ARGS})
fi

"${PYTHON_BIN}" "${PY_ARGS[@]}"

echo "=== Paper B HF (1GPU) done ==="

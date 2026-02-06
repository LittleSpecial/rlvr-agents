#!/bin/bash
#SBATCH --job-name=paper_a_hf
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=128
#SBATCH --time=24:00:00
#SBATCH --output=logs/paper_a_hf_%j.out
#SBATCH --error=logs/paper_a_hf_%j.err
#SBATCH --export=ALL

# ============================================================
# Paper A: Counterfactual Credit Assignment (HF backend, 4×A100)
#
# 1) 必须先在“登录节点”完成：
#    - conda 环境 + pip install -r requirements.txt
#    - 下载模型到本地路径（计算节点无网）
#    - 导出数据集 JSONL（见 datasets/code/README.md）
#
# 2) 用法（登录节点）：
#    mkdir -p logs experiments
#    sbatch scripts/slurm/run_paper_a_hf_4gpu.sh
#
# 3) 重要：本脚本默认用 torchrun 做单机多卡 DDP。
#    这会“复制模型到每张卡”，提升吞吐；不会自动做模型切分（不等价于更大模型能放下）。
# ============================================================

# 注意：不用 -u，否则 conda 的 deactivate/activate 脚本可能触发 unbound variable 导致作业秒退
set -eo pipefail
trap 'rc=$?; echo "[ERR] run_paper_a_hf_4gpu.sh failed at line ${LINENO} (rc=${rc})" >&2' ERR

mkdir -p logs experiments
cd "${SLURM_SUBMIT_DIR:-$PWD}"

export TMPDIR="${TMPDIR:-$HOME/tmp}"
mkdir -p "${TMPDIR}"
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_DDP_TIMEOUT_SECONDS="${TORCH_DDP_TIMEOUT_SECONDS:-3600}"

# NCCL settings for PCIe-connected multi-GPU (no InfiniBand)
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

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
TORCH_WHL_VER="$("${PYTHON_BIN}" - <<'PY'
import importlib.metadata as md
try:
    print(md.version("torch"))
except Exception:
    print("")
PY
)"
echo "torch (wheel): ${TORCH_WHL_VER:-UNKNOWN}"

# Fixed cluster defaults (can still be overridden by sbatch --export).
: "${CUDA_MODULE:=compilers/cuda/11.8}"
: "${GCC_MODULE:=compilers/gcc/11.3.0}"
: "${CUDNN_MODULE:=cudnn/8.6.0.163_cuda11.x}"
: "${NCCL_MODULE:=nccl/2.11.4-1_cuda11.8}"

echo "=== Loading compute modules ==="
echo "GCC_MODULE=${GCC_MODULE:-}"
echo "CUDA_MODULE=${CUDA_MODULE:-}"
echo "CUDNN_MODULE=${CUDNN_MODULE:-cudnn/8.6.0.163_cuda11.x}"
echo "NCCL_MODULE=${NCCL_MODULE:-}"
module load "${GCC_MODULE}"
module load "${CUDA_MODULE}"
module load "${CUDNN_MODULE:-cudnn/8.6.0.163_cuda11.x}"
if [ -n "${NCCL_MODULE:-}" ]; then
  module load "${NCCL_MODULE}" 2>/dev/null || true
fi

echo "=== Module list ==="
module list 2>&1 || true

# 离线模式（计算节点通常无互联网）
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

# 线程数（可按需调）
export OMP_NUM_THREADS=8

echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME:-$(hostname)}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES:-N/A}"
echo "PYTHON_BIN: ${PYTHON_BIN}"
echo "=========================================="

echo "=== nvidia-smi ==="
nvidia-smi || true

# fail-fast：避免“申请了 4 卡但 torch 只能 CPU 跑”的浪费
"${PYTHON_BIN}" - <<'PY'
import sys
try:
    import torch
except ImportError as e:
    raise SystemExit(
        f"\n[ERROR] torch import failed: {e}\n"
        "常见原因：torch wheel 是 +cu118 但作业里加载了 cuda/11.6，导致找不到 libcupti.so.11.8。\n"
        "修复：提交作业时设置 CUDA_MODULE=compilers/cuda/11.8，并匹配 GCC/NCCL 模块。\n"
    )
print("python:", sys.version.replace("\n", " "))
print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("cuda.is_available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit(
        "\n[ERROR] torch.cuda.is_available() == False\n"
        "大概率是装了 CPU-only PyTorch 或者缺少 cudnn module。\n"
    )
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

# 你需要把模型与数据放在计算节点可见的路径上
MODEL_PATH="${MODEL_PATH:-/path/to/local/Qwen2.5-7B-Instruct}"
TRAIN_DATA="${TRAIN_DATA:-datasets/code/mbpp_train.jsonl}"
EVAL_DATA="${EVAL_DATA:-datasets/code/humaneval_test.jsonl}"

echo "MODEL_PATH=${MODEL_PATH}"
echo "TRAIN_DATA=${TRAIN_DATA}"
echo "EVAL_DATA=${EVAL_DATA}"

NUM_GPUS="${SLURM_GPUS_ON_NODE:-4}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-paper_a_hf_${SLURM_JOB_ID}}"
ENV_TYPE="${ENV_TYPE:-code}"
SHOW_TESTS="${SHOW_TESTS:-1}"  # 1=show tests, 0=hide tests
DTYPE="${DTYPE:-bf16}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_ROLLOUTS_PER_PROMPT="${NUM_ROLLOUTS_PER_PROMPT:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-1024}"
MAX_TRAJECTORY_LENGTH="${MAX_TRAJECTORY_LENGTH:-20}"
MAX_POLICY_TURNS="${MAX_POLICY_TURNS:-2}"
TASK_TIMEOUT_SECONDS="${TASK_TIMEOUT_SECONDS:-4}"  # shorter timeout for multi-GPU to reduce straggler
MAX_STEPS="${MAX_STEPS:-200}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"
EVAL_INTERVAL="${EVAL_INTERVAL:-50}"
SAVE_INTERVAL="${SAVE_INTERVAL:-200}"
USE_COUNTERFACTUAL_CREDIT="${USE_COUNTERFACTUAL_CREDIT:-1}"  # 1=on, 0=off
PRIORITIZE_HIGH_VALUE_CF="${PRIORITIZE_HIGH_VALUE_CF:-0}"    # 1=only high-value trajectories, 0=all trajectories
COUNTERFACTUAL_K="${COUNTERFACTUAL_K:-2}"
REWARD_MODE="${REWARD_MODE:-mixed}"                          # binary|score|mixed
REWARD_BLEND_ALPHA="${REWARD_BLEND_ALPHA:-0.5}"
FLAT_GROUP_FALLBACK="${FLAT_GROUP_FALLBACK:-raw}" # zero|batch_centered|raw
FAILURE_REWARD_FLOOR="${FAILURE_REWARD_FLOOR:--0.01}"
CREDIT_FALLBACK_WHEN_ZERO_ADV="${CREDIT_FALLBACK_WHEN_ZERO_ADV:-1}"  # 1|0
FAILURE_REPLAY_RATIO="${FAILURE_REPLAY_RATIO:-0.5}"
FAILURE_BUFFER_SIZE="${FAILURE_BUFFER_SIZE:-4096}"
FAILURE_REPLAY_WARMUP_STEPS="${FAILURE_REPLAY_WARMUP_STEPS:-50}"
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

echo "=== Paper A run config ==="
echo "EXPERIMENT_NAME=${EXPERIMENT_NAME}"
echo "ENV_TYPE=${ENV_TYPE} SHOW_TESTS=${SHOW_TESTS}"
echo "DTYPE=${DTYPE} LR=${LEARNING_RATE}"
echo "BATCH_SIZE=${BATCH_SIZE} NUM_ROLLOUTS_PER_PROMPT=${NUM_ROLLOUTS_PER_PROMPT}"
echo "MAX_PROMPT_TOKENS=${MAX_PROMPT_TOKENS} MAX_NEW_TOKENS=${MAX_NEW_TOKENS}"
echo "MAX_TRAJECTORY_LENGTH=${MAX_TRAJECTORY_LENGTH} MAX_POLICY_TURNS=${MAX_POLICY_TURNS}"
echo "TASK_TIMEOUT_SECONDS=${TASK_TIMEOUT_SECONDS}"
echo "MAX_STEPS=${MAX_STEPS} LOG_INTERVAL=${LOG_INTERVAL} EVAL_INTERVAL=${EVAL_INTERVAL} SAVE_INTERVAL=${SAVE_INTERVAL}"
echo "USE_COUNTERFACTUAL_CREDIT=${USE_COUNTERFACTUAL_CREDIT} PRIORITIZE_HIGH_VALUE_CF=${PRIORITIZE_HIGH_VALUE_CF}"
echo "COUNTERFACTUAL_K=${COUNTERFACTUAL_K} INTERVENTION_TYPES=${INTERVENTION_TYPES}"
echo "REWARD_MODE=${REWARD_MODE} BLEND_ALPHA=${REWARD_BLEND_ALPHA} FLAT_GROUP_FALLBACK=${FLAT_GROUP_FALLBACK}"
echo "FAILURE_REWARD_FLOOR=${FAILURE_REWARD_FLOOR}"
echo "CREDIT_FALLBACK_WHEN_ZERO_ADV=${CREDIT_FALLBACK_WHEN_ZERO_ADV}"
echo "FAILURE_REPLAY_RATIO=${FAILURE_REPLAY_RATIO} FAILURE_BUFFER_SIZE=${FAILURE_BUFFER_SIZE} WARMUP=${FAILURE_REPLAY_WARMUP_STEPS}"
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
  ${CF_FLAG}
  ${CF_PRIOR_FLAG}
  ${CF_ZERO_ADV_FLAG}
  --counterfactual_k "${COUNTERFACTUAL_K}"
  --reward_mode "${REWARD_MODE}"
  --reward_blend_alpha "${REWARD_BLEND_ALPHA}"
  --failure_reward_floor "${FAILURE_REWARD_FLOOR}"
  --flat_group_fallback "${FLAT_GROUP_FALLBACK}"
  --failure_replay_ratio "${FAILURE_REPLAY_RATIO}"
  --failure_buffer_size "${FAILURE_BUFFER_SIZE}"
  --failure_replay_warmup_steps "${FAILURE_REPLAY_WARMUP_STEPS}"
  --intervention_types ${INTERVENTION_TYPES}
  --output_dir ./experiments
)

if [ -n "${EXTRA_ARGS}" ]; then
  # shellcheck disable=SC2206
  PY_ARGS+=(${EXTRA_ARGS})
fi

"${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node="${NUM_GPUS}" "${PY_ARGS[@]}"

echo "=== Paper A HF done ==="

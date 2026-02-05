#!/bin/bash
#SBATCH --job-name=paper_a_hf
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --output=logs/paper_a_hf_%j.out
#SBATCH --error=logs/paper_a_hf_%j.err

# ============================================================
# Paper A: HF backend (单卡 smoke / debug)
# 用法：
#   MODEL_PATH=... TRAIN_DATA=... EVAL_DATA=... sbatch scripts/slurm/run_paper_a_hf_1gpu.sh
# ============================================================

echo "=== Job started at $(date) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "PWD: $(pwd)"
echo "SLURM_SUBMIT_DIR: ${SLURM_SUBMIT_DIR}"

mkdir -p logs experiments
cd "${SLURM_SUBMIT_DIR}"

echo "=== Loading modules ==="
module purge
module load miniforge3/24.1 || echo "Warning: miniforge3 module load failed"

# N32-H 手册：CUDA/GCC 模块一般在 compilers/* 下（不要用 cuda/11.x 这种名字）
# 如果你安装了超算提供的 PyTorch wheel，可以直接 source 它的 env.sh（里面会 module load）
# 推荐：选一个与你 Python 版本 (cp310) + 驱动支持的 CUDA 版本匹配的 env.sh。
# 你集群当前驱动显示 CUDA 11.6，因此优先 cu116；避免 cu118/cu121 以免报 driver/runtime 不兼容。
PYTORCH_ENV_SH="${PYTORCH_ENV_SH:-/home/bingxing2/apps/package/pytorch/1.13.1+cu116_cp310/env.sh}"
if [ -f "${PYTORCH_ENV_SH}" ]; then
  echo "Sourcing PYTORCH_ENV_SH=${PYTORCH_ENV_SH} (non-fatal)"
  # shellcheck disable=SC1090
  source "${PYTORCH_ENV_SH}" || echo "Warning: source env.sh failed; falling back to manual modules"
fi

module load compilers/gcc/9.3.0 2>/dev/null || true
module load compilers/cuda/11.6 2>/dev/null || true
# cuDNN is required by CUDA-enabled PyTorch wheels; auto-pick a cuda11.* variant if available.
if command -v module >/dev/null 2>&1; then
  if [ -n "${CUDNN_MODULE:-}" ]; then
    module load "${CUDNN_MODULE}" 2>/dev/null || true
  else
    _CUDNN_CAND="$(module avail cudnn 2>&1 | grep -oE 'cudnn/[^[:space:]]+' | grep -E 'cuda11\\.x|cu11' | head -n 1)"
    if [ -z "${_CUDNN_CAND}" ]; then
      _CUDNN_CAND="$(module avail cudnn 2>&1 | grep -oE 'cudnn/[^[:space:]]+' | grep -E 'cuda11|cu11' | head -n 1)"
    fi
    if [ -n "${_CUDNN_CAND}" ]; then
      echo "Auto-loading ${_CUDNN_CAND}"
      module load "${_CUDNN_CAND}" 2>/dev/null || true
    else
      echo "Warning: could not auto-detect a cuda11 cuDNN module (module avail cudnn)."
    fi
  fi
fi

echo "=== CUDA check ==="
nvidia-smi || echo "Warning: nvidia-smi failed"

echo "=== Activating conda ==="
eval "$(conda shell.bash hook)" 2>/dev/null || true
conda activate rlvr || source activate rlvr || echo "Warning: conda activate failed"

echo "=== Python info ==="
which python3
python3 --version

echo "=== Torch CUDA check (fail-fast) ==="
python3 - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("cuda.is_available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit(
        "\n[ERROR] torch.cuda.is_available() == False\n"
        "This usually means you installed a CPU-only PyTorch build on N32-H (aarch64).\n"
        "Fix: install the cluster-provided CUDA PyTorch wheel and source its env.sh, e.g. (per manual):\n"
        "  pip install /home/bingxing2/apps/package/pytorch/<...>/*.whl\n"
        "  source /home/bingxing2/apps/package/pytorch/<...>/env.sh\n"
        "Then re-submit the job.\n"
    )
PY

echo "=== Environment variables ==="
echo "MODEL_PATH=${MODEL_PATH:-NOT SET}"
echo "TRAIN_DATA=${TRAIN_DATA:-NOT SET}"
echo "EVAL_DATA=${EVAL_DATA:-NOT SET}"

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

MODEL_PATH="${MODEL_PATH:-/path/to/local/Qwen2.5-7B-Instruct}"
TRAIN_DATA="${TRAIN_DATA:-datasets/code/mbpp_train.jsonl}"
EVAL_DATA="${EVAL_DATA:-datasets/code/humaneval_test.jsonl}"

python3 paper_a_credit_assignment/train.py \
  --experiment_name "paper_a_hf_${SLURM_JOB_ID}" \
  --backend hf \
  --model_path "${MODEL_PATH}" \
  --env_type code \
  --no-show_tests \
  --require_cuda \
  --train_dataset "${TRAIN_DATA}" \
  --eval_dataset "${EVAL_DATA}" \
  --dtype bf16 \
  --learning_rate 1e-5 \
  --batch_size 2 \
  --num_rollouts_per_prompt 2 \
  --max_new_tokens 256 \
  --max_prompt_tokens 1024 \
  --max_steps 50 \
  --log_interval 5 \
  --eval_interval 25 \
  --save_interval 0 \
  --use_counterfactual_credit \
  --counterfactual_k 2 \
  --intervention_types delete truncate \
  --output_dir ./experiments

echo "=== Paper A HF (1GPU) done ==="

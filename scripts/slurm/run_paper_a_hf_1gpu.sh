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
#SBATCH --export=ALL

# ============================================================
# Paper A: HF backend (单卡 smoke / debug)
# 用法：
#   MODEL_PATH=... TRAIN_DATA=... EVAL_DATA=... sbatch scripts/slurm/run_paper_a_hf_1gpu.sh
# ============================================================

set -euo pipefail
trap 'rc=$?; echo "[ERR] run_paper_a_hf_1gpu.sh failed at line ${LINENO} (rc=${rc})" >&2' ERR

echo "=== Job started at $(date) ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "PWD: $(pwd)"
echo "SLURM_SUBMIT_DIR: ${SLURM_SUBMIT_DIR}"

mkdir -p logs experiments
cd "${SLURM_SUBMIT_DIR}"

# tmp 目录：某些集群 /tmp 配额或权限会导致 pip/transformers 临时文件失败
export TMPDIR="${TMPDIR:-$HOME/tmp}"
mkdir -p "${TMPDIR}"

echo "=== Loading modules (cluster template) ==="
module purge
module load miniforge3/24.1
module load compilers/gcc/9.3.0
module load compilers/cuda/11.6
module load "${CUDNN_MODULE:-cudnn/8.6.0.163_cuda11.x}"

echo "=== Module list ==="
module list 2>&1 || true

echo "=== CUDA check ==="
nvidia-smi || echo "Warning: nvidia-smi failed"

echo "=== Conda env / Python selection ==="
CONDA_ENV="${CONDA_ENV:-$HOME/.conda/envs/rlvr}"
if [ -d "${CONDA_ENV}" ]; then
  # 工程师模板：source activate /path/to/env
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

echo "=== Python info ==="
echo "PYTHON_BIN=${PYTHON_BIN}"
"${PYTHON_BIN}" --version

echo "=== Torch CUDA check (fail-fast) ==="
"${PYTHON_BIN}" - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("cuda.is_available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit(
        "\n[ERROR] torch.cuda.is_available() == False\n"
        "大概率是装了 CPU-only PyTorch 或者缺少 cudnn module。\n"
        "按 N32-H 手册：module load cudnn/... 并安装集群提供的 CUDA PyTorch wheel。\n"
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

"${PYTHON_BIN}" paper_a_credit_assignment/train.py \
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

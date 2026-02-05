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

set -euo pipefail

mkdir -p logs experiments
cd "${SLURM_SUBMIT_DIR}"

module purge
module load miniforge3/24.1
# N32-H 手册：CUDA/GCC 模块一般在 compilers/* 下；或直接 source 超算提供的 PyTorch env.sh
PYTORCH_ENV_SH="${PYTORCH_ENV_SH:-/home/bingxing2/apps/package/pytorch/1.11.0+cu113_cp38/env.sh}"
if [ -f "${PYTORCH_ENV_SH}" ]; then
  echo "Sourcing PYTORCH_ENV_SH=${PYTORCH_ENV_SH}"
  # shellcheck disable=SC1090
  source "${PYTORCH_ENV_SH}"
else
  module load compilers/gcc/9.3.0 2>/dev/null || true
  module load compilers/cuda/11.3 2>/dev/null || true
fi
eval "$(conda shell.bash hook)" 2>/dev/null || true
conda activate rlvr || source activate rlvr

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
echo "Python: $(which python3)"
echo "=========================================="

# fail-fast：避免“申请了 4 卡但 torch 只能 CPU 跑”的浪费
python3 - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("cuda.is_available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit(
        "\n[ERROR] torch.cuda.is_available() == False\n"
        "You are likely using a CPU-only PyTorch build. Install the cluster-provided CUDA wheel + source env.sh.\n"
    )
PY

# 你需要把模型与数据放在计算节点可见的路径上
MODEL_PATH="${MODEL_PATH:-/path/to/local/Qwen2.5-7B-Instruct}"
TRAIN_DATA="${TRAIN_DATA:-datasets/code/mbpp_train.jsonl}"
EVAL_DATA="${EVAL_DATA:-datasets/code/humaneval_test.jsonl}"

echo "MODEL_PATH=${MODEL_PATH}"
echo "TRAIN_DATA=${TRAIN_DATA}"
echo "EVAL_DATA=${EVAL_DATA}"

NUM_GPUS="${SLURM_GPUS_ON_NODE:-4}"

torchrun --standalone --nproc_per_node="${NUM_GPUS}" paper_a_credit_assignment/train.py \
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
  --max_steps 200 \
  --log_interval 10 \
  --eval_interval 50 \
  --save_interval 200 \
  --use_counterfactual_credit \
  --counterfactual_k 2 \
  --intervention_types delete truncate \
  --output_dir ./experiments

echo "=== Paper A HF done ==="

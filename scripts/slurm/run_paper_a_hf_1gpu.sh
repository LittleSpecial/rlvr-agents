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

# Debug mode: show errors instead of silent fail
set -x

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

echo "=== Activating conda ==="
eval "$(conda shell.bash hook)" 2>/dev/null || true
conda activate rlvr || source activate rlvr || echo "Warning: conda activate failed"

echo "=== Python info ==="
which python3
python3 --version

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


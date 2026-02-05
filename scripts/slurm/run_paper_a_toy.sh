#!/bin/bash
#SBATCH --job-name=paper_a_toy
#SBATCH --partition=N32-H           # GPU分区，根据手册调整
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1                # 申请1块GPU
#SBATCH --time=02:00:00
#SBATCH --output=logs/paper_a_%j.out
#SBATCH --error=logs/paper_a_%j.err

# ============================================================
# Paper A: Counterfactual Credit Assignment (Toy Backend)
# 用法: sbatch scripts/slurm/run_paper_a_toy.sh
# ============================================================

set -e
echo "=== Paper A Training (Toy Backend) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"

mkdir -p logs experiments

# 加载环境
module purge
module load anaconda3 2>/dev/null || module load miniconda3 2>/dev/null || true
module load cuda/12.1 2>/dev/null || module load cuda/11.8 2>/dev/null || true
conda activate rlvr

cd $SLURM_SUBMIT_DIR

# 运行训练
python3 paper_a_credit_assignment/train.py \
    --experiment_name paper_a_toy_${SLURM_JOB_ID} \
    --backend toy \
    --env_type code \
    --no-show_tests \
    --use_counterfactual_credit \
    --counterfactual_k 4 \
    --intervention_types delete truncate \
    --max_steps 500 \
    --log_interval 20 \
    --eval_interval 100 \
    --batch_size 32 \
    --num_rollouts_per_prompt 4 \
    --seed 42

echo "=== Training Complete ==="

#!/bin/bash
#SBATCH --job-name=paper_b_toy
#SBATCH --partition=gpu             # 按超算手册：分区通常为 gpu（如有不同用 sinfo 确认）
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1                    # 申请1块GPU（如不支持该字段可改用 --gres=gpu:1）
#SBATCH --cpus-per-task=32          # 按手册：1 GPU 绑定 32 CPU
#SBATCH --time=02:00:00
#SBATCH --output=logs/paper_b_%j.out
#SBATCH --error=logs/paper_b_%j.err

# ============================================================
# Paper B: Interference-aware RLVR (Toy Backend)
# 用法: sbatch scripts/slurm/run_paper_b_toy.sh
# ============================================================

set -e
echo "=== Paper B Training (Toy Backend) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"

mkdir -p logs experiments

# 加载环境
module purge
module load miniforge3/24.1
eval "$(conda shell.bash hook)" 2>/dev/null || true
conda activate rlvr || source activate rlvr

cd $SLURM_SUBMIT_DIR

# 运行训练
python3 paper_b_conflict_aware/train.py \
    --experiment_name paper_b_toy_${SLURM_JOB_ID} \
    --backend toy \
    --env_type code \
    --no-show_tests \
    --protocol sequential \
    --skill_sequence type0 type1 \
    --stage_steps 100 \
    --projection sequential_margin \
    --epsilon 0.0 \
    --memory_per_protected 4 \
    --max_steps 500 \
    --log_interval 20 \
    --eval_interval 100 \
    --batch_size 32 \
    --num_rollouts_per_prompt 4 \
    --seed 42

echo "=== Training Complete ==="

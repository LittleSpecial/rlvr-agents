#!/bin/bash
#SBATCH -J rlvr_exp            # 作业名称
#SBATCH -p gpu                 # 分区名称
#SBATCH -N 1                   # 申请1个节点
#SBATCH --gpus=1               # 申请1块GPU卡
#SBATCH --cpus-per-task=32     # 1块卡默认配32核CPU
#SBATCH --output=logs/job_%j.out    # 标准输出日志
#SBATCH --error=logs/job_%j.err     # 错误日志

# === 环境加载 ===
module load miniforge3/24.1
# 如果需要特定CUDA版本: module load compilers/cuda/11.6
source activate rlvr

# 调试信息
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "节点: $(hostname)"
echo "显卡: $(nvidia-smi -L)"
echo "Python: $(which python3)"
echo "=========================================="

cd $SLURM_SUBMIT_DIR

# === 实验 A: Credit Assignment ===
echo ""
echo "[$(date '+%H:%M:%S')] Start Experiment A (Credit Assignment)..."
python3 paper_a_credit_assignment/train.py \
    --experiment_name paper_a_${SLURM_JOB_ID} \
    --backend toy \
    --env_type code \
    --no-show_tests \
    --use_counterfactual_credit \
    --counterfactual_k 4 \
    --intervention_types delete truncate \
    --max_steps 50 \
    --log_interval 5 \
    --eval_interval 25 \
    --batch_size 32 \
    --output_dir ./experiments

echo ""
echo "[$(date '+%H:%M:%S')] Experiment A finished."

# === 实验 B: Interference-aware RLVR ===
echo ""
echo "[$(date '+%H:%M:%S')] Start Experiment B (Interference-aware RLVR)..."
python3 paper_b_conflict_aware/train.py \
    --experiment_name paper_b_${SLURM_JOB_ID} \
    --backend toy \
    --env_type code \
    --no-show_tests \
    --protocol sequential \
    --skill_sequence type0 type1 \
    --stage_steps 50 \
    --projection sequential_margin \
    --epsilon 0.0 \
    --memory_per_protected 4 \
    --max_steps 200 \
    --log_interval 10 \
    --eval_interval 50 \
    --batch_size 32 \
    --output_dir ./experiments

echo ""
echo "[$(date '+%H:%M:%S')] Experiment B finished."
echo ""
echo "=========================================="
echo "All experiments completed!"
echo "Results saved to: ./experiments/"
echo "=========================================="

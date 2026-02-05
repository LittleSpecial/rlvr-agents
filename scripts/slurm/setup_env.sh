#!/bin/bash
#SBATCH --job-name=rlvr_setup
#SBATCH --partition=N32-H           # 根据手册调整分区名
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=logs/setup_%j.out
#SBATCH --error=logs/setup_%j.err

# ============================================================
# 环境配置脚本 - 只需运行一次
# 用法: sbatch scripts/slurm/setup_env.sh
# ============================================================

set -e
echo "=== RLVR Environment Setup ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"

# 创建日志目录
mkdir -p logs

# 加载基础模块 (根据集群调整)
module purge
module load anaconda3 2>/dev/null || module load miniconda3 2>/dev/null || true
module load cuda/12.1 2>/dev/null || module load cuda/11.8 2>/dev/null || true

# 创建conda环境
ENV_NAME="rlvr"
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Environment $ENV_NAME already exists, updating..."
    conda activate $ENV_NAME
else
    echo "Creating new environment: $ENV_NAME"
    conda create -n $ENV_NAME python=3.10 -y
    conda activate $ENV_NAME
fi

# 安装PyTorch (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装其他依赖
pip install -r requirements.txt

# 验证安装
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"

echo "=== Setup Complete ==="

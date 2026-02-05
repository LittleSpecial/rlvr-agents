#!/bin/bash
# 首次部署脚本 - 在登录节点运行（有网络）
# 用法: bash setup_cluster.sh

set -e
echo "=== RLVR 环境配置 (北京超算 N32-H) ==="

# 加载 miniforge
module load miniforge3/24.1

# 创建 conda 环境
if conda env list | grep -q "^rlvr "; then
    echo "环境 rlvr 已存在，跳过创建"
else
    echo "创建新环境: rlvr"
    conda create -n rlvr python=3.10 -y
fi

conda activate rlvr

# 安装依赖 (登录节点有网络)
echo "安装 PyTorch..."
pip install torch torchvision torchaudio

echo "安装其他依赖..."
pip install numpy pyyaml transformers accelerate peft datasets

# 验证
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# 创建日志目录
mkdir -p logs experiments

echo ""
echo "=== 配置完成 ==="
echo "提交作业: sbatch run_job.sh"
echo "查看状态: parajobs"

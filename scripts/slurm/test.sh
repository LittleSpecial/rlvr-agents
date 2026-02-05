#!/bin/bash
#SBATCH --job-name=rlvr_setup
#SBATCH --partition=gpu             # 用 sinfo 确认分区名
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
export  PYTHONUNBUFFERED=1
# 加载基础模块 (根据集群调整)
module purge
module load miniforge3/24.1
module load compilers/gcc/9.3.0
module load compilers/cuda/11.6
module load cudnn/8.6.0.163_cuda11.x
source activate /home/bingxing2/home/scx9krq/.conda/envs/rlvr


echo "=== nvidia-smi ==="
nvidia-smi || true
echo `which python`
python 1.py
echo "=== Setup Complete ==="

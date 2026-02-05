#!/bin/bash
#SBATCH --job-name=rlvr_setup
#SBATCH --partition=gpu             # 用 sinfo 确认分区名
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=logs/setup_%j.out
#SBATCH --error=logs/setup_%j.err

# ============================================================
# 环境检查/辅助脚本（建议先在登录节点跑 setup_cluster.sh）
# 用法: sbatch scripts/slurm/setup_env.sh
#
# 作用：
# - 在“有 GPU 的计算节点”上确认 torch 是否能看到 CUDA（避免你现在遇到的 CUDA: False）
# - 若 CUDA 不可用，打印手册对应的修复建议（安装超算提供的 PyTorch CUDA wheel + source env.sh）
# ============================================================

set -euo pipefail
echo "=== RLVR Environment Setup ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"

# 创建日志目录
mkdir -p logs

# 加载基础模块 (根据集群调整)
module purge
module load miniforge3/24.1 2>/dev/null || true

# N32-H 手册：优先 source 超算提供的 PyTorch env.sh（里面会 module load compilers/cuda/gcc）
PYTORCH_ENV_SH="${PYTORCH_ENV_SH:-/home/bingxing2/apps/package/pytorch/1.13.1+cu116_cp310/env.sh}"
if [ -f "${PYTORCH_ENV_SH}" ]; then
  echo "Sourcing PYTORCH_ENV_SH=${PYTORCH_ENV_SH} (non-fatal)"
  # 某些 env.sh 会尝试加载不存在的 anaconda 模块；失败则回退到手动 module load
  set +e
  # shellcheck disable=SC1090
  source "${PYTORCH_ENV_SH}"
  _SRC_RC=$?
  set -e
  if [ ${_SRC_RC} -ne 0 ]; then
    echo "Warning: source env.sh failed (rc=${_SRC_RC}); falling back to manual modules"
  fi
fi

module load compilers/gcc/9.3.0 2>/dev/null || true
module load compilers/cuda/11.6 2>/dev/null || true

eval "$(conda shell.bash hook)" 2>/dev/null || true
conda activate rlvr 2>/dev/null || source activate rlvr 2>/dev/null || true

echo "=== nvidia-smi ==="
nvidia-smi || true

echo "=== Python / Torch check ==="
python3 - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("cuda.is_available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
if not torch.cuda.is_available():
    print(
        "\n[ERROR] torch.cuda.is_available() == False\n"
        "你现在的环境很可能是 CPU-only PyTorch（aarch64 上最常见）。\n"
        "按手册修复（示例）：\n"
        "  pip install /home/bingxing2/apps/package/pytorch/1.11.0+cu113_cp38/*.whl\n"
        "  source /home/bingxing2/apps/package/pytorch/1.11.0+cu113_cp38/env.sh\n"
        "然后重新提交作业。\n"
    )
    raise SystemExit(2)
PY

echo "=== Setup Complete ==="

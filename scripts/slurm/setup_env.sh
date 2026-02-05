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
#SBATCH --export=ALL

# ============================================================
# 环境检查/辅助脚本（建议先在登录节点跑 setup_cluster.sh）
# 用法: sbatch scripts/slurm/setup_env.sh
#
# 作用：
# - 在“有 GPU 的计算节点”上确认 torch 是否能看到 CUDA（避免你现在遇到的 CUDA: False）
# - 若 CUDA 不可用，打印手册对应的修复建议（安装超算提供的 PyTorch CUDA wheel + source env.sh）
# ============================================================

set -eo pipefail  # 注意：不用 -u，否则 conda 的 deactivate 脚本会报 unbound variable
trap 'rc=$?; echo "[ERR] setup_env.sh failed at line ${LINENO} (rc=${rc})" >&2' ERR
echo "=== RLVR Environment Setup ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"

# 创建日志目录
mkdir -p logs

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

echo "=== Python selection ==="
CONDA_ENV="${CONDA_ENV:-$HOME/.conda/envs/rlvr}"
if [ -d "${CONDA_ENV}" ]; then
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

echo "PYTHON_BIN=${PYTHON_BIN}"
"${PYTHON_BIN}" -V

echo "=== nvidia-smi ==="
nvidia-smi || true

echo "=== Python / Torch check ==="
"${PYTHON_BIN}" - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("cuda.is_available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
if not torch.cuda.is_available():
    print(
        "\n[ERROR] torch.cuda.is_available() == False\n"
        "大概率是装了 CPU-only PyTorch 或者缺少 cudnn module。\n"
        "请确认：module load cudnn/... 并安装集群提供的 CUDA PyTorch wheel。\n"
    )
    raise SystemExit(2)
PY

echo "=== Setup Complete ==="

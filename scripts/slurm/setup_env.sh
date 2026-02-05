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
export PYTHONNOUSERSITE=1

echo "=== Loading modules (cluster template) ==="
module purge
module load miniforge3/24.1
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

echo "=== Torch wheel info (no import) ==="
TORCH_WHL_VER="$("${PYTHON_BIN}" - <<'PY'
import importlib.metadata as md
try:
    print(md.version("torch"))
except Exception:
    print("")
PY
)"
echo "torch (wheel): ${TORCH_WHL_VER:-UNKNOWN}"

# Auto-align CUDA/GCC/NCCL module versions with the installed torch wheel tag (cu116/cu118/...)
if [ -z "${CUDA_MODULE:-}" ] || [ -z "${GCC_MODULE:-}" ]; then
  TORCH_CU_TAG="$(echo "${TORCH_WHL_VER}" | sed -n 's/.*+cu\\([0-9][0-9][0-9]\\).*/cu\\1/p')"
  case "${TORCH_CU_TAG}" in
    cu118)
      : "${CUDA_MODULE:=compilers/cuda/11.8}"
      : "${GCC_MODULE:=compilers/gcc/11.3.0}"
      : "${NCCL_MODULE:=nccl/2.11.4-1_cuda11.8}"
      ;;
    cu116)
      : "${CUDA_MODULE:=compilers/cuda/11.6}"
      : "${GCC_MODULE:=compilers/gcc/9.3.0}"
      ;;
    *)
      : "${CUDA_MODULE:=compilers/cuda/11.6}"
      : "${GCC_MODULE:=compilers/gcc/9.3.0}"
      ;;
  esac
fi

echo "=== Loading compute modules ==="
echo "GCC_MODULE=${GCC_MODULE:-}"
echo "CUDA_MODULE=${CUDA_MODULE:-}"
echo "CUDNN_MODULE=${CUDNN_MODULE:-cudnn/8.6.0.163_cuda11.x}"
echo "NCCL_MODULE=${NCCL_MODULE:-}"
module load "${GCC_MODULE}"
module load "${CUDA_MODULE}"
module load "${CUDNN_MODULE:-cudnn/8.6.0.163_cuda11.x}"
if [ -n "${NCCL_MODULE:-}" ]; then
  module load "${NCCL_MODULE}" 2>/dev/null || true
fi

echo "=== Module list ==="
module list 2>&1 || true

echo "=== nvidia-smi ==="
nvidia-smi || true

echo "=== Python / Torch check ==="
"${PYTHON_BIN}" - <<'PY'
import sys
try:
    import torch
except ImportError as e:
    raise SystemExit(
        f"\n[ERROR] torch import failed: {e}\n"
        "常见原因：torch wheel 是 +cu118 但作业里加载了 cuda/11.6，导致找不到 libcupti.so.11.8。\n"
        "修复：提交作业时设置 CUDA_MODULE=compilers/cuda/11.8，并匹配 GCC/NCCL 模块。\n"
    )
print("python:", sys.version.replace("\n", " "))
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

echo "=== HF stack versions (best-effort) ==="
"${PYTHON_BIN}" - <<'PY'
import importlib
def v(name):
    try:
        m = importlib.import_module(name)
        return getattr(m, "__version__", "unknown")
    except Exception as e:
        return f"ERR({type(e).__name__})"
for pkg in ["numpy", "transformers", "peft", "datasets"]:
    print(f"{pkg}:", v(pkg))
PY

echo "=== Setup Complete ==="

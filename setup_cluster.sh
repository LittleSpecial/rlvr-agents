#!/bin/bash
# 首次部署脚本 - 建议在登录节点运行
# 用法: bash setup_cluster.sh

set -euo pipefail

echo "=== RLVR 环境配置 (北京超算 N32-H / aarch64) ==="

# 加载 miniforge
module purge
module load miniforge3/24.1

ENV_NAME="${ENV_NAME:-rlvr}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"

# 创建 conda 环境
if conda env list | grep -q "^${ENV_NAME} "; then
  echo "环境 ${ENV_NAME} 已存在，跳过创建"
else
  echo "创建新环境: ${ENV_NAME} (python=${PYTHON_VERSION})"
  conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y
fi

eval "$(conda shell.bash hook)" 2>/dev/null || true
conda activate "${ENV_NAME}" || source activate "${ENV_NAME}"

python3 -V

echo ""
echo "=== 1) 安装 PyTorch (关键) ==="
echo "N32-H 是 aarch64，官方 PyTorch CUDA wheel 往往不匹配；优先用超算提供的 wheel（见用户手册）。"

if [ -d "/home/bingxing2/apps/package/pytorch" ]; then
  shopt -s nullglob
  PYTAG="$(python3 - <<'PY'\nimport sys\nprint(f\"cp{sys.version_info.major}{sys.version_info.minor}\")\nPY\n)"
  echo "Detected python wheel tag: ${PYTAG}"

  echo ""
  echo "建议：先用与你驱动兼容的 CUDA wheel（例如 nvidia-smi 显示 CUDA 11.6 就优先 cu116）。"

  # Prefer a conservative, widely compatible build: cu116 + cp310.
  # (Avoid cu121/cu124 unless your driver is new enough.)
  CAND_DIRS=(
    "/home/bingxing2/apps/package/pytorch/1.13.1+cu116_${PYTAG}"
    "/home/bingxing2/apps/package/pytorch/1.12.1+cu116_${PYTAG}"
    "/home/bingxing2/apps/package/pytorch/2.0.0+cu117_${PYTAG}"
    "/home/bingxing2/apps/package/pytorch/2.0.1+cu118_${PYTAG}"
  )

  PICKED=""
  for d in "${CAND_DIRS[@]}"; do
    if [ -d "$d" ] && ls "$d"/*.whl >/dev/null 2>&1; then
      PICKED="$d"
      break
    fi
  done

  if [ -n "${PICKED}" ]; then
    echo "Using wheels from: ${PICKED}"
    python3 -m pip install --no-cache-dir --force-reinstall "${PICKED}"/*.whl
    if [ -f "${PICKED}/env.sh" ]; then
      echo ""
      echo "提示：作业脚本里建议 source："
      echo "  source ${PICKED}/env.sh"
    fi
  else
    echo "[WARN] 没找到推荐的 wheel 目录（${PYTAG}）。你可以："
    echo "  1) 列出可用项：ls /home/bingxing2/apps/package/pytorch"
    echo "  2) 选择与你 Python (${PYTAG}) 匹配的目录后安装："
    echo "       python3 -m pip install /home/bingxing2/apps/package/pytorch/<DIR>/*.whl"
    echo "  3) 若只有 cp38 wheel：创建 python=3.8 环境："
    echo "       ENV_NAME=rlvr_py38 PYTHON_VERSION=3.8 bash setup_cluster.sh"
  fi

  # 提示 env.sh（用于 module load cuda/gcc）
  ENV_SH_CAND=(/home/bingxing2/apps/package/pytorch/*/env.sh)
  if [ ${#ENV_SH_CAND[@]} -gt 0 ]; then
    echo ""
    echo "提示：作业脚本里建议 source PyTorch 的 env.sh（会自动 module load），例如："
    echo "  source /home/bingxing2/apps/package/pytorch/1.13.1+cu116_cp310/env.sh"
  fi
else
  echo "[WARN] 未检测到 /home/bingxing2/apps/package/pytorch；请联系管理员确认 PyTorch CUDA wheel 位置。"
fi

echo ""
echo "=== 2) 安装项目依赖 ==="
echo "如果登录节点能访问 pip 源："
python3 -m pip install -r requirements.txt || echo "[WARN] pip install -r requirements.txt 失败（可能无网或源不可达），需要离线 wheelhouse。"

echo ""
echo "=== 3) 创建目录 ==="
mkdir -p logs experiments

echo ""
echo "=== 配置完成 ==="
echo "提交 toy 跑通：   sbatch scripts/slurm/run_paper_a_toy.sh"
echo "提交 HF (1GPU)：  MODEL_PATH=... TRAIN_DATA=... EVAL_DATA=... sbatch scripts/slurm/run_paper_a_hf_1gpu.sh"
echo "查看任务：        squeue -u $USER   (或集群自带 parajobs)"

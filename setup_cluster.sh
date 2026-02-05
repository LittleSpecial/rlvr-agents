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

  # 找到匹配当前 Python 的 wheel（torch/torchvision/torchaudio 等）
  WHEELS=(/home/bingxing2/apps/package/pytorch/*/*"${PYTAG}"*.whl)
  if [ ${#WHEELS[@]} -gt 0 ]; then
    echo "Found ${#WHEELS[@]} wheels under /home/bingxing2/apps/package/pytorch (matching ${PYTAG})"
    python3 -m pip install --no-cache-dir --force-reinstall "${WHEELS[@]}"
  else
    echo "[WARN] 没找到匹配 ${PYTAG} 的 wheel。你可以："
    echo "  1) 先 ls /home/bingxing2/apps/package/pytorch 看有哪些版本"
    echo "  2) 或者创建 python=3.8 环境以匹配 cp38 wheel："
    echo "       ENV_NAME=rlvr_py38 PYTHON_VERSION=3.8 bash setup_cluster.sh"
    echo "  3) 安装示例（手册给的例子）："
    echo "       pip install /home/bingxing2/apps/package/pytorch/1.11.0+cu113_cp38/*.whl"
  fi

  # 提示 env.sh（用于 module load cuda/gcc）
  ENV_SH_CAND=(/home/bingxing2/apps/package/pytorch/*/env.sh)
  if [ ${#ENV_SH_CAND[@]} -gt 0 ]; then
    echo ""
    echo "提示：作业脚本里建议 source PyTorch 的 env.sh（会自动 module load），例如："
    echo "  source ${ENV_SH_CAND[0]}"
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

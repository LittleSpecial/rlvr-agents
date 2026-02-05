# 集群部署指南

## 文件传输

```bash
# 本地 -> 服务器 (rsync更快，支持断点续传)
rsync -avz --progress . username@cluster:/path/to/rlvr-agents/

# 或用scp
scp -r . username@cluster:/path/to/rlvr-agents/
```

## 首次配置环境

```bash
# 登录集群
ssh username@cluster

# 进入项目目录
cd /path/to/rlvr-agents

# 首次配置环境：必须在“登录节点”完成（计算节点通常无互联网）
bash setup_cluster.sh

# 验证环境
python3 -c "import sys; print(sys.version)"
```

## N32-H (aarch64) 的 PyTorch 注意事项（你现在卡住的点）

你在计算节点里看到：

- `nvidia-smi` 有 GPU
- 但 `torch.cuda.is_available()` 是 `False`

这几乎总是因为 **装成了 CPU-only 的 PyTorch**（aarch64 上从官方源装 CUDA wheel 经常不匹配）。

按手册的做法：安装超算提供的 CUDA PyTorch wheel，并在作业里 `source env.sh`。

在集群上常见目录：

```bash
ls /home/bingxing2/apps/package/pytorch
ls /home/bingxing2/apps/package/pytorch/*/*.whl
cat /home/bingxing2/apps/package/pytorch/*/env.sh
```

手册示例（你截图里的那段）：

```bash
pip install /home/bingxing2/apps/package/pytorch/1.11.0+cu113_cp38/*.whl
source /home/bingxing2/apps/package/pytorch/1.11.0+cu113_cp38/env.sh
```

在你这台机器上（你贴的列表里），更推荐直接用 **cp310 + cu116**（因为 `nvidia-smi` 显示驱动 CUDA 11.6）：

```bash
pip uninstall -y torch torchvision torchaudio
pip install /home/bingxing2/apps/package/pytorch/1.13.1+cu116_cp310/*.whl
source /home/bingxing2/apps/package/pytorch/1.13.1+cu116_cp310/env.sh
python3 -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())"
```

如果 `source .../env.sh` 报类似 `Unable to locate a modulefile for 'anaconda/2021.11'`，说明该 `env.sh` 里加载了集群上不存在的模块。
这不影响你使用 wheel：可以 **跳过 env.sh**，改为手动加载（手册 2.1.4 的写法）：

```bash
module load compilers/gcc/9.3.0
module load compilers/cuda/11.6
```

或者在提交作业时临时禁用 `env.sh`（让脚本走 fallback）：

```bash
PYTORCH_ENV_SH=/nonexistent sbatch scripts/slurm/setup_env.sh
```

然后用一个最小 Slurm 作业在“有 GPU 的计算节点”上验证：

```bash
sbatch scripts/slurm/setup_env.sh
```

## transformers/peft 版本不匹配（你现在遇到的报错）

如果你看到类似报错：

- `Disabling PyTorch because PyTorch >= 2.2 is required but found 1.13.1+cu116`
- `AutoModelForCausalLM requires the PyTorch library but it was not found`
- `A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x`

原因通常是：

- 你的 `~/.local`（用户目录）里装了“过新”的 `transformers/peft`，它们会要求 `torch>=2.2`；
- 你当前环境里是 `torch 1.13.*`，或混装导致 `transformers` 找不到 torch；
- 你的 `numpy==2.x` 和部分二进制包（torch/扩展）不兼容。

解决原则：

- **不要混用 `~/.local`**：跑作业时设置 `PYTHONNOUSERSITE=1`（我们已在 slurm 脚本里加了）。
- 在 **conda 环境里**把 `numpy/transformers/peft` 统一到兼容组合。

### 方案 A（更稳）：torch 1.13.1+cu116 + 旧版 transformers/peft

在登录节点执行：

```bash
module purge
module load miniforge3/24.1
source activate $HOME/.conda/envs/rlvr

export PYTHONNOUSERSITE=1
export TMPDIR=$HOME/tmp; mkdir -p $TMPDIR

python -m pip install --no-cache-dir --force-reinstall -r requirements.txt
python scripts/slurm/1.py
```

### 方案 B（更省心贴合新版 HF）：torch 2.4.0+cu118（推荐你现在用）

在登录节点执行：

```bash
module purge
module load miniforge3/24.1
source activate $HOME/.conda/envs/rlvr

export PYTHONNOUSERSITE=1
export TMPDIR=$HOME/tmp; mkdir -p $TMPDIR

python -m pip uninstall -y torch torchvision torchaudio transformers peft datasets numpy
python -m pip install --no-cache-dir --force-reinstall /home/bingxing2/apps/package/pytorch/2.4.0+cu118_cp310/*.whl
python -m pip install --no-cache-dir --force-reinstall -r requirements_torch2.txt

python scripts/slurm/1.py
```

然后提交作业时显式指定模块版本（cu118 对齐 cuda11.8 + nccl11.8）：

```bash
CUDA_MODULE=compilers/cuda/11.8 \
GCC_MODULE=compilers/gcc/11.3.0 \
NCCL_MODULE=nccl/2.11.4-1_cuda11.8 \
CUDNN_MODULE=cudnn/8.6.0.163_cuda11.x \
sbatch scripts/slurm/run_paper_a_hf_1gpu.sh
```

## 运行训练

```bash
# 首次提交前创建目录（Slurm 不会自动创建 --output/--error 的父目录）
mkdir -p logs experiments

# Paper A (toy backend)
sbatch scripts/slurm/run_paper_a_toy.sh

# Paper A (HF backend, 单机 4 卡)
# 需要提前在登录节点下载好模型/数据，并设置路径：
MODEL_PATH=/path/to/local/Qwen2.5-7B-Instruct \
TRAIN_DATA=datasets/code/mbpp_train.jsonl \
EVAL_DATA=datasets/code/humaneval_test.jsonl \
sbatch scripts/slurm/run_paper_a_hf_4gpu.sh

# Paper A (HF backend, 单卡 debug)
MODEL_PATH=/path/to/local/Qwen2.5-7B-Instruct \
TRAIN_DATA=datasets/code/mbpp_train.jsonl \
EVAL_DATA=datasets/code/humaneval_test.jsonl \
sbatch scripts/slurm/run_paper_a_hf_1gpu.sh

# Paper B (toy backend)
sbatch scripts/slurm/run_paper_b_toy.sh

# 或者一次性跑 A+B
sbatch run_job.sh

# 查看运行中的任务
squeue -u $USER

# 取消任务
scancel <JOB_ID>
```

## 常用命令

| 命令                     | 说明           |
| ------------------------ | -------------- |
| `sbatch script.sh`       | 提交批处理作业 |
| `squeue -u $USER`        | 查看自己的任务 |
| `scancel <ID>`           | 取消任务       |
| `sinfo`                  | 查看分区状态   |
| `scontrol show job <ID>` | 查看任务详情   |

## 查看结果

```bash
# 训练日志
ls logs/

# 实验结果
ls experiments/
cat experiments/*/EXPERIMENT.md
```

## 注意事项

1. **严格区分节点**：登录节点只做下载/安装/编辑；训练必须 `sbatch` 到计算节点（见 `超算操作规范手册.md`）。
2. **离线运行**：计算节点通常无互联网；模型/数据要提前在登录节点下载到本地磁盘。
3. **分区/资源配比**：用 `sinfo` 确认分区名；按手册 1 GPU 绑定 32 CPU（`--gpus=1` + `--cpus-per-task=32`）。
4. **Slurm 版本差异**：若 `--gpus=1` 不被支持，改用 `--gres=gpu:1`。

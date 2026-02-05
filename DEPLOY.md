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

## 运行训练

```bash
# 首次提交前创建目录（Slurm 不会自动创建 --output/--error 的父目录）
mkdir -p logs experiments

# Paper A (toy backend)
sbatch scripts/slurm/run_paper_a_toy.sh

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

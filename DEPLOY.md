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

# 提交环境配置任务
sbatch scripts/slurm/setup_env.sh

# 查看任务状态
squeue -u $USER

# 查看日志
tail -f logs/setup_*.out
```

## 运行训练

```bash
# Paper A (toy backend)
sbatch scripts/slurm/run_paper_a_toy.sh

# Paper B (toy backend)
sbatch scripts/slurm/run_paper_b_toy.sh

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

1. **分区名**：脚本中的 `--partition=N32-H` 需根据实际分区名调整
2. **GPU申请**：`--gres=gpu:1` 申请1块GPU，按需调整
3. **时间限制**：`--time=02:00:00` 为2小时，超时会被杀掉
4. **模块加载**：`module load` 命令需根据集群实际模块调整

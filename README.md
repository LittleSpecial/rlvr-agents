# NeurIPS 2026 RLVR 研究项目

两篇论文的代码实现，共享同一套基础设施。

## 项目结构

```
neurips2026_plans/
├── shared/                          # 共享代码库
│   ├── envs/                        # 可验证奖励环境
│   │   ├── base.py                  # 环境基类和数据结构
│   │   ├── code_env.py              # 代码环境（单元测试验证）
│   │   └── sql_env.py               # SQL环境（结果匹配验证）
│   ├── rl/                          # RL算法
│   │   ├── base_trainer.py          # 训练器基类
│   │   ├── grpo_trainer.py          # GRPO训练器
│   │   └── advantage.py             # 优势函数计算
│   └── experiment_tracking/         # 实验记录系统
│       ├── tracker.py               # 实验追踪器（自动生成文档）
│       ├── run_logger.py            # 运行日志
│       └── config_manager.py        # 配置管理
│
├── paper_a_credit_assignment/       # Paper A: 反事实信用分配
│   ├── counterfactual.py            # 反事实干预生成与执行
│   ├── credit_estimator.py          # Credit估计器
│   ├── advantage_mapper.py          # Advantage映射
│   ├── train_args.py                # 训练参数定义
│   ├── train_utils.py               # 共享训练工具函数
│   ├── toy_runner.py                # toy backend runner
│   ├── hf_runner.py                 # HF backend runner (DDP)
│   └── train.py                     # 精简训练入口
│
├── paper_b_conflict_aware/          # Paper B: 冲突感知RLVR
│   ├── group_assignment.py          # 分组策略
│   ├── conflict_detector.py         # 冲突检测
│   ├── gradient_surgery.py          # 梯度手术（PCGrad等）
│   └── train.py                     # 训练入口
│
└── experiments/                     # 实验结果目录（自动生成）
```

## 安装

```bash
# Toy backend（可本地跑通闭环，无重依赖）
# 仅需 Python 3.8+

# Real LLM 训练（HF Transformers + LoRA）
python3 -m pip install -r requirements.txt
```

## 数据集（CodeEnv）

- 格式：JSONL，一行一个任务；字段规范见 `datasets/code/README.md`
- 示例：`datasets/code/example.jsonl`
- 从 HuggingFace 导出（不建议把大文件提交 GitHub）：

```bash
python3 -m pip install datasets

python3 scripts/prepare_code_dataset_hf.py --dataset mbpp --split train --out datasets/code/mbpp_train.jsonl
python3 scripts/prepare_code_dataset_hf.py --dataset openai_humaneval --split test --out datasets/code/humaneval_test.jsonl
python3 scripts/validate_code_jsonl.py datasets/code/mbpp_train.jsonl
```

## Paper A: Verifiable Agent RL with Counterfactual Credit Assignment

**核心思想**: 用反事实重放估计每个step对最终成功的边际贡献，将稀疏的最终reward转化为step-level credit。

### 运行

```bash
python3 paper_a_credit_assignment/train.py \
    --experiment_name test_run \
    --backend toy \
    --model_name Qwen2.5-7B \
    --env_type code \
    --no-show_tests \
    --counterfactual_k 4 \
    --intervention_types delete truncate \
    --max_steps 200 \
    --log_interval 10
```

### HF 后端（真实模型）

先准备 `CodeEnv` JSONL 数据集（见 `datasets/code/README.md`），然后：

```bash
python3 paper_a_credit_assignment/train.py \
  --experiment_name hf_smoke \
  --backend hf \
  --model_path /path/to/local/Qwen2.5-7B-Instruct \
  --env_type code \
  --no-show_tests \
  --train_dataset datasets/code/mbpp_train.jsonl \
  --eval_dataset datasets/code/humaneval_test.jsonl \
  --dtype bf16 \
  --batch_size 2 \
  --num_rollouts_per_prompt 2 \
  --max_new_tokens 256 \
  --max_steps 50 \
  --log_interval 5 \
  --eval_interval 25
```

单机多卡（DDP）建议用 `torchrun`（见 `scripts/slurm/run_paper_a_hf_4gpu.sh`）。

多卡默认启用两项稳定性保护（可通过环境变量关闭）：

- `SYNC_EVAL_AND_SAVE=1`：在 eval/checkpoint 前后加分布式栅栏，避免 rank0 慢评估导致其余 rank 在下一步 DDP 卡住。
- `TRUNCATE_TO_GLOBAL_MIN_SAMPLES=1`：各 rank 每步按全局最小 sample 数对齐，降低步间时长抖动和长尾阻塞。
- `FALLBACK_TO_ADV_WHEN_ZERO_CREDIT=1`：当 counterfactual 映射出的 step-credit 近似全零时，回退到 trajectory advantage，避免整步无更新。
- `FAILURE_BUFFER_UNIQUE=1`：failure replay buffer 按 task 去重，避免被少数困难样本过度采样。

### 关键参数

| 参数                     | 默认值          | 说明                 |
| ------------------------ | --------------- | -------------------- |
| `--counterfactual_k`     | 4               | 每条轨迹的反事实数量 |
| `--intervention_types`   | delete truncate | 干预类型             |
| `--credit_normalization` | signed          | Credit归一化方式     |
| `--log_agop`             | true            | 记录 toy 梯度的 AGOP 统计 |

## Paper B: Interference-aware RLVR (Multi-Skill Code)

**核心思想**: 在 multi-skill RLVR 中显式控制跨 skill 的干扰（interference）与遗忘（retention），用“skill-safe”约束投影在提升目标 skill 的同时尽量不伤害受保护 skills。

### 运行

```bash
python3 paper_b_conflict_aware/train.py \
    --experiment_name test_run \
    --backend toy \
    --model_name Qwen2.5-7B \
    --env_type code \
    --no-show_tests \
    --protocol sequential \
    --skill_sequence type0 type1 \
    --stage_steps 50 \
    --projection sequential_margin \
    --epsilon 0.0 \
    --memory_per_protected 4 \
    --max_steps 200 \
    --log_interval 10
```

### 关键参数

| 参数                   | 默认值     | 说明         |
| ---------------------- | ---------- | ------------ |
| `--protocol`           | mixed      | mixed / sequential |
| `--skill_sequence`     | (auto)     | sequential 的 skill 顺序 |
| `--stage_steps`        | 2000       | 每个 stage 训练步数 |
| `--projection`         | sequential_margin | 约束投影方法（pcgrad 是 epsilon=0 特例） |
| `--epsilon`            | 0.0        | 约束松弛：dot(g_s, Δθ) >= -epsilon |
| `--memory_per_protected` | 8        | 每步对受保护 skill 采样多少 prompts 来估计约束梯度 |

## 实验记录

每次实验自动生成：

```
experiments/
└── paper_a_test_run_20260205_120000_abc12345/
    ├── EXPERIMENT.md          # 实验文档（自动更新）
    ├── config.json            # 配置
    ├── checkpoints/           # 模型检查点
    ├── logs/
    │   ├── metrics.jsonl      # 指标日志
    │   └── events.jsonl       # 事件日志
    ├── artifacts/             # 中间产物
    └── analysis/              # 分析图表
```

## 里程碑

- W1 (02-04~02-10): 统一环境 + baseline RLVR
- W2-W3: Paper A 反事实credit实现
- W4-W5: Paper B 冲突检测和梯度手术
- W6-W7: 扩展实验 + 机制分析
- W8: 论文写作

## 引用相关工作

**Paper A**:
- Mesnard et al., *Counterfactual Credit Assignment* (ICML 2021)

**Paper B**:
- Yu et al., *PCGrad* (NeurIPS 2020)
- DeepSeekMath GRPO (2024)

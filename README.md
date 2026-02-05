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
│   └── train.py                     # 训练入口
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
# 仅需 Python 3.10+

# Real LLM 训练（后续接 HF Transformers / vLLM 时需要）
pip install torch transformers pyyaml
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
    --counterfactual_k 4 \
    --intervention_types delete truncate \
    --max_steps 200 \
    --log_interval 10
```

### 关键参数

| 参数                     | 默认值          | 说明                 |
| ------------------------ | --------------- | -------------------- |
| `--counterfactual_k`     | 4               | 每条轨迹的反事实数量 |
| `--intervention_types`   | delete truncate | 干预类型             |
| `--credit_normalization` | signed          | Credit归一化方式     |

## Paper B: Conflict-aware RLVR

**核心思想**: 检测不同任务簇的梯度冲突，通过梯度手术缓解模式塌缩，保持pass@k多样性。

### 运行

```bash
python3 paper_b_conflict_aware/train.py \
    --experiment_name test_run \
    --backend toy \
    --model_name Qwen2.5-7B \
    --env_type code \
    --group_strategy task_type \
    --num_groups 4 \
    --surgery_method pcgrad \
    --max_steps 200 \
    --log_interval 10
```

### 关键参数

| 参数                   | 默认值     | 说明         |
| ---------------------- | ---------- | ------------ |
| `--group_strategy`     | difficulty | 分组策略     |
| `--num_groups`         | 4          | 组数量       |
| `--surgery_method`     | pcgrad     | 梯度手术算法 |
| `--conflict_threshold` | 0.0        | 冲突阈值     |

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

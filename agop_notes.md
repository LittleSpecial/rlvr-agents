# AGOP 在 Paper A（反事实 credit assignment）里的用法笔记

## 1) AGOP 是什么（我们在这里用到的版本）

给定一组梯度向量 `{g_i}`（可以是每条轨迹/每个 step 的 policy-gradient，或某层/LoRA 参数梯度的扁平化向量），定义：

`A = (1/n) Σ_i g_i g_i^T`

这就是 **Average Gradient Outer Product**（AGOP）的一般形式：梯度外积的均值，等价于“梯度方向的二阶统计/未中心化协方差”。

## 2) 为什么它能服务你的 A（credit map）叙事

Paper A 的核心是：用反事实重放得到 step-level credit，让更新更多由“关键 step”驱动。

直觉上，这会改变梯度集合 `{g_i}` 的结构：

- 如果 credit 更“尖锐”（少数 step 权重大），梯度能量可能更集中到少数方向（低有效秩）。
- 如果 credit 帮你避免把奖励均摊到无关 step，更新方向会更一致（top eigen / trace 比例上升），学习更稳定。

这些都可以用 AGOP 的谱/有效秩来做机制图（不是证明式理论，但很像“可量化的机制分析”）。

## 3) 我们在代码里记录了哪些 AGOP 标量

`paper_a_credit_assignment/train.py`（toy backend）会在 `logs/metrics.jsonl` 的 `extra` 字段里记录：

- `agop_trace`：`tr(A)`，等价于 `(1/n) Σ ||g_i||^2`
- `agop_effective_rank`：`(tr(A))^2 / tr(A^2)`（participation ratio），越小越“低秩/集中”
- `agop_top_trace_ratio`：`λ_max(A) / tr(A)`（用 power iteration 估 top eig），越大越“主方向占比高”
- `agop_baseline_*`：同一批样本，用“未做反事实重权”的 baseline 权重（advantages）算一遍，便于做对照

实现见 `shared/analysis/agop.py`（无 numpy/torch 依赖）。

## 4) 下一步怎么把它变成论文里的图

建议最少做 2 条曲线（同一个 seed 集合）：

1. `--use_counterfactual_credit` vs `--no-use_counterfactual_credit`
2. 画 `success/pass@k` 学习曲线 + `agop_effective_rank` 随 step 变化

如果 A 的叙事成立，你应该能看到：

- credit 打开后：更快的上升 or 更稳定的曲线
- 同时 AGOP 指标出现可解释变化（例如 effective rank 下降、top/trace 上升）

## 5) 真正上 LLM（HF backend）时怎么做

toy backend 的 AGOP 只是在 logit 空间（K 维）做示例。

上真实 LLM 后，建议先把 AGOP 计算限制在：

- LoRA/adapter 参数子集（开销可控）
- 或者最后几层/LM head（方便做谱分析）

然后用 “per-sample / per-group 梯度” 构造 `{g_i}`，复用同一套指标（trace / effective rank / top ratio）。


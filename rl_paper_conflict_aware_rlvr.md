# NeurIPS 2026 Paper B（更 RL / 更“有内容”）— Conflict-aware RLVR for LLM Reasoning

> 目标：做一篇偏 **机制 + 新方法 + 实证** 的 RL 论文，围绕 RLVR（对 LLM 进行可验证奖励的 RL，如 pass@1 / unit tests / exact match）中常见的 **多样性下降/高 k 退化/模式塌缩** 等现象，给出一个“可解释 + 可修复”的方法学贡献。

## 1. 背景与动机（Motivation）

### 1.1 观察到的问题
在“可验证奖励 + policy optimization”的训练中，常见现象包括：
- pass@1 变好，但 pass@k/多解多样性变差；
- 策略从“探索多种解法”切到“单一模板/捷径”；
- 不同题型（难题/易题、不同主题/不同推理模式）更新相互干扰，导致部分能力被压掉。

### 1.2 核心假设（Mechanism Hypothesis）
把 RLVR 看成在一个混合分布上的优化：每个 batch 里不同子任务/不同推理模式对应不同梯度方向。  
当这些方向 **冲突（梯度夹角为负）** 时，标准的 PPO/GRPO 更新会产生 **负迁移**，表现为：
- 某些模式被系统性削弱；
- 最终在采样空间里“边界收缩”（高 k 不再多样）。

研究问题：
**能否显式检测并缓解这种“梯度冲突”，从而在提升 pass@1 的同时保留 pass@k、多样性与稳健性？**

## 2. 核心 idea（One-liner）

在 RLVR 更新里加入一个轻量的 **conflict-aware 梯度处理**（类似 gradient surgery / 投影），让来自不同“任务簇/难度簇/推理簇”的更新尽量不互相伤害，从优化层面抑制模式塌缩。

## 3. 方法框架（Method）

### 3.1 任务分簇（Task/Mode Partition）
把每个样本分到一个簇 `g`，用于计算组梯度：
- 按 **难度**：用 base model 的 pass@k / logprob / self-consistency 估计难度
- 按 **题型**：math / code / logic / reading（如果数据带标签）
- 按 **解法模式**：用简单的特征（是否生成代码、是否调用工具、是否产生长 CoT）

### 3.2 组梯度与冲突度量
在一次更新里，对每个组 `g` 计算 `∇_g`（对 RL objective 的梯度估计）。  
定义冲突：`cos(∇_g, ∇_h) < 0` 或者 dot product 为负。

### 3.3 冲突缓解策略（核心算法）
给出一个“最小改动”的优化规则（便于审稿人理解与复现）：
- **投影/手术**：当 `∇_g · ∇_h < 0` 时，把 `∇_g` 在 `∇_h` 的反方向分量去掉
- 最终更新方向 `∇* = aggregate({∇_g'})`

工程上要点：
- 组数要少（2–8 组），否则开销大；
- 可以只在冲突严重时启用（阈值触发）。

### 3.4 与 RLVR recipe 的兼容
把它做成“插在 optimizer 前的一层”，兼容：
- GRPO / PPO / REINFORCE（只要能拿到 per-group gradient）
- 以及 KL/entropy 正则（依旧保留，但不是主修补手段）

## 4. 实验设计（Eval）

### 4.1 环境与任务（偏大模型、非机器人控制）
优先选择 **可验证奖励** 的推理任务（方便大规模实验）：
- 代码：unit-test / compile / execute（最干净）
- 数学：exact answer check（注意格式化/等价性）
- 结构化输出：parser/validator

### 4.2 指标（必须覆盖“边界收缩”）
- pass@1、pass@k（k=4/8/16）
- 多样性度量：unique solutions、n-gram/AST 多样性（针对 code）
- 可靠性：失败模式、错误类型分布
- 训练稳定性：不同 seed 方差

### 4.3 对比基线（强对比才像 NeurIPS）
- 原始 GRPO/PPO（+常用 KL/entropy）
- data-centric：难度过滤/重采样/ curriculum（如果你们愿意做）
- reward shaping：过程奖励/自一致性奖励（可选）

### 4.4 机制验证（让“机制假设”站得住）
你需要把“梯度冲突 → 模式塌缩”这条链条证出来：
- 冲突统计：训练过程中冲突比例如何变化
- “被压掉的簇”：哪一类任务/哪一种解法模式贡献下降
- 手术前后：冲突下降 ↔ pass@k/多样性回升

## 5. 资源与实现（Implementation Notes）

### 5.1 训练规模建议（8×A800）
- 先用 `Qwen2.5-7B/14B` 复现实验全流程（快速迭代）
- 再用更大模型做最终主表（如 `32B`，视吞吐而定）
- 优先 LoRA / QLoRA（减少迭代成本），最后再考虑全参

### 5.2 工程拆解（最小可行）
- dataset loader + verifier
- rollout sampler（支持 n-sample 用于 pass@k 估计）
- RL trainer（GRPO/PPO）
- group assignment（按难度/题型/模式）
- per-group grad 计算 + conflict surgery
- logging（冲突矩阵、簇内/簇间指标）

## 6. 风险点与规避（Risks）
- “别人已经做过类似梯度手术”：需要系统性 related-work 检索与清晰区分（例如他们若是多目标 RL 或多任务 SFT 的冲突处理，你强调 **RLVR 的边界收缩现象 + pass@k 指标 + 可验证奖励场景**）。
- 组划分不合理：做一个最稳的（按难度二分/三分），并做 ablation。
- 开销过大：组数小 + 触发式启用 + 只对部分层/部分梯度做（如 adapter 层）。

## 7. 里程碑（粗略）
- W1：复现 RLVR baseline + 复现“边界收缩”现象（pass@1 vs pass@k）
- W2–W3：实现 group + 冲突统计 + 初版 surgery
- W4–W5：扩到多个任务/多个模型规模 + 强消融
- W6–W7：机制分析图表 + 论文写作
- W8：收敛结果 + 打磨叙事 + 最终成稿


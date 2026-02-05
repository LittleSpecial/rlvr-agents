# NeurIPS 2026 Paper B（更 RL / 代码向）— Interference-aware RLVR for Multi-Skill Code Models

> 目标：做一篇偏 **机制 + 新方法 + 实证** 的 RL 论文，但刻意避开“修 pass@k / 防多样性塌缩”这条拥挤主线。  
> 我们把关注点改成更 RL 的问题：**multi-skill / continual RLVR** 下的 *interference*（负迁移）与 *retention*（不遗忘），面向 **代码任务**（unit tests / compile / execute 等可验证奖励）。

## 1. 背景与动机（Motivation）

### 1.1 观察到的问题
在“可验证奖励 + policy optimization（RLVR）”的代码训练中，经常会遇到：
- 针对某一类代码任务做 RL 后，另一类任务性能下降（负迁移 / 遗忘）。
- 先训 A 再训 B：B 上升，但 A 掉得明显（catastrophic forgetting）。
- 多任务混训：总体指标上升，但最差子任务（worst-skill）很差，且对 seed 很敏感。

现实痛点：我们通常希望“在不伤害已有能力”的情况下，用 RLVR 增加新技能或提升某一技能（例如 bug fixing），而不是拿一个更偏科的模型。

### 1.2 核心假设（Mechanism Hypothesis）
把 RLVR 看成一个多目标/多技能的优化：每个技能（skill/task）对应一个可验证 reward。  
标准 PPO/GRPO 往往在“平均意义上”最大化总体回报，但不保证每个 skill 都不受伤。  
在参数共享（尤其是 LoRA/adapter 低秩参数）下，更新会把某些 skill 的决策边界推到不利方向，导致可测的遗忘与负迁移。

研究问题：
**能否在 RLVR 中显式控制“跨技能干扰”，实现：训练新 skill 的同时，旧 skill 的性能不下降（或下降受控）？**

## 2. 核心 idea（One-liner）

把 RLVR 训练改写为 **skill-safe 的约束优化 / 多目标优化**：在提升目标 skill 的同时，对其它 skill 加 “do-no-harm / bounded forgetting” 约束；用低开销（仅在 LoRA/adapter 参数上）的近似投影/修正来实现。

## 3. 方法框架（Method）

### 3.1 skill 定义与训练协议（Protocols）
定义 skills：一组可验证的代码子任务（每个 skill 都有 verifier）：
- 代码生成（unit tests）
- 代码修复/补丁（unit tests）
- 代码推理/重构（compile/execute/format checks）

训练协议（至少 2 种都要做）：
- **Sequential**：先训 skill A，再训 B、C…（典型遗忘场景）
- **Mixed**：多 skill 混训（典型 worst-skill 场景）

### 3.2 约束式 RLVR（Skill-safe Update）
设每个 skill 为 `s∈{1..S}`，对应一个 RLVR surrogate objective `J_s(θ)`（如 PPO/GRPO 的期望优势）。  
我们希望最大化目标 skill（或平均）同时满足“其它 skill 不退化”：

- 目标：`max_θ  J_target(θ)`（或 `Σ_s w_s J_s(θ)`）
- 约束：对每个受保护 skill `s`，要求 `J_s(θ_new) ≥ J_s(θ_old) - ε`

一阶近似后可转成对更新方向 `Δθ` 的线性约束：
- `∇J_s(θ_old) · Δθ ≥ -ε`

然后做一个 **最小改动投影**：从候选更新 `g`（比如目标 skill 的梯度或加权平均梯度）出发，求一个满足约束的 `Δθ*`。

工程落地要点：
- 只在 **LoRA/adapter 参数** 上做（维度小、开销可控）
- 约束 skill 数 `S` 不要太大（2–8 个足够先出论文）

### 3.3 低开销的实现近似（Practical Approximation）
你不需要一开始就上复杂 QP 求解器，可以按“从简到繁”做：
- **Sequential projection**：依次对每个约束 skill 做投影修正（一次只处理一个约束）
- **Trigger-based**：只在检测到遗忘趋势（protected skill 的在线评测下降）时启用
- **Memory-based constraints（小回放）**：每个 protected skill 保留少量 prompt（几十到几百），用它估计 `∇J_s`，避免每步都大规模采样

工程上要点：
- 受保护 skills 越多越难；建议先从 “保护 1–2 个关键 skill” 做起。
- 约束松弛 `ε` 要做 sweep：`ε=0` 太硬可能学不动，`ε>0` 更现实。

### 3.4 与 RLVR recipe 的兼容
把它做成“插在 optimizer 前的一层”，兼容：
- GRPO / PPO（主推）
- REINFORCE（作为 sanity baseline）
- KL/entropy 正则：可以保留，但论文主卖点是 **retention/skill-safety**，不是“更强正则”

## 4. 实验设计（Eval）

### 4.1 任务与数据（代码向）
优先选择可复现、可并行的大规模 verifier：
- unit tests（最核心）
- compile/execute（补充）

数据来源可以先从“小而干净”的集合做起，再扩：
- 生成：HumanEval/MBPP 风格
- 修复：小型 bugfix 数据（有测试即可）

### 4.2 指标（必须覆盖 retention）
除每个 skill 的成功率外，重点指标是：
- **Avg**：平均成功率
- **Worst-skill**：最差 skill 成功率（公平性/鲁棒性）
- **Forgetting**：训练后相对训练前/峰值的下降幅度
- **Forward transfer**：训新 skill 是否顺便提升旧 skill（可选）
- 训练稳定性：seed 方差、学习曲线震荡

（可选）成本指标：token/tool cost（如果你们想给“现实可用性”加分）

### 4.3 对比基线（强对比才像 NeurIPS）
- 原始 GRPO/PPO（单 skill、混训、顺序训三种协议）
- 仅加 KL/entropy
- 经验回放（简单 replay）：为旧 skill 混入少量样本
- 常见 continual learning 正则（如 EWC 类、L2 到旧参数；可选）

### 4.4 机制验证（让“干扰”站得住）
你需要把“interference → forgetting”这条链条证出来：
- 哪些 skill 对哪些 skill 造成最大伤害（干扰矩阵）
- 约束启用前后：forgetting 曲线是否显著改善
- 保护强度（`ε`）与目标提升的 trade-off（Pareto 曲线）

## 5. 资源与实现（Implementation Notes）

### 5.1 训练规模建议（8×A800）
- 先用 `Qwen2.5-7B/14B` 复现实验全流程（快速迭代）
- 再用更大模型做最终主表（如 `32B`，视吞吐而定）
- 优先 LoRA / QLoRA（减少迭代成本），最后再考虑全参

### 5.2 工程拆解（最小可行）
- dataset loader + verifier
- rollout sampler（每个 prompt 采样 n 次用于稳定估计）
- RL trainer（GRPO/PPO）
- skill assignment（按任务簇/数据源/工具类型）
- per-skill grad 估计 + constraint projection（只做 LoRA/adapter）
- logging（干扰矩阵、forgetting 曲线、worst-skill）

## 6. 风险点与规避（Risks）
- 和“梯度冲突/多样性”主线混淆：论文叙事要明确：我们优化的是 **retention/worst-skill/forgetting**，不是 pass@k。
- 受保护 skills 太多导致目标 skill 学不动：先从 2–4 skills 做起，逐步扩展。
- 开销过大：只在 LoRA/adapter 上算；memory-based 估计；触发式启用。

## 7. 里程碑（粗略）
- W1：搭建 2–3 个 code skills + 跑通单 skill RLVR baseline
- W2：跑通 sequential vs mixed 协议 + 复现 forgetting/worst-skill 问题
- W3–W4：实现 skill-safe update（投影/触发式）+ 初版 ablation
- W5–W6：扩到 4–6 skills + 更大模型/更多 seeds + 机制分析图表
- W7–W8：写作与打磨（trade-off/Pareto、干扰矩阵、案例分析）

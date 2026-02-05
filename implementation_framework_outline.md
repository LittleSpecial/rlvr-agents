# NeurIPS 2026 两篇论文：实现框架大纲（基于当前两份草稿）

本文档目标：把你现有两份想法，落到**可复现的工程模块 + 实验闭环**，并尽量让两篇共享同一套 RLVR/Agent 基础设施，降低总工程量。

---

## 0. 统一假设与边界（两篇共享）

- **奖励形态**：主要是 *verifiable final reward*（`r_final∈{0,1}` 或离散/可检验标量）；必要时可扩展到带诊断信息的 `verifier_info`（例如 unit-test 失败用例、SQL mismatch diff）。
- **训练范式**：优先复现/实现一条稳定 pipeline（GRPO/PPO/REINFORCE 任选其一先跑通），再在其上做方法插拔。
- **环境优先级**：先做 **本地可验证、低工程风险、高吞吐** 的 tool/code 环境；web/GUI 作为可选故事线补充。

---

## 1. 共享代码架构（建议一套 repo 同时服务 Paper A/B）

### 1.1 目录与组件边界（建议）

- `envs/`：可验证环境（code/SQL/JSON…），统一 step API
- `verifiers/`：程序化 verifier（可复用，含诊断输出）
- `models/`：LLM 推理封装（采样、logprob、KV-cache、tool-call 格式化）
- `rollout/`：并行 rollout、轨迹收集、缓存与回放
- `rl/`：RL 算法实现（GRPO/PPO/REINFORCE）与 advantage 接口
- `methods/`：
  - `credit/`：Paper A 的 counterfactual credit 估计
  - `conflict/`：Paper B 的 conflict-aware 梯度处理
- `eval/`：pass@1/pass@k、多样性、成本等统一评测脚本
- `configs/`：训练/评测配置（模型、采样、batch、K、分组等）
- `logs/`（或外部如 wandb）：曲线、credit map、冲突矩阵、样例追踪

### 1.2 统一数据结构（关键：让 A/B 都能“插模块”）

**(1) 轨迹 Trajectory**
- `prompt_id / task_id`
- `steps[t]`：`(obs_t, action_t, logprob_t, tool_meta_t, token_meta_t, step_hash_t)`
- `r_final`
- `verifier_info`（diff/失败用例/最早成功时间等）
- `cost`（tokens、tool 调用次数、walltime）

**(2) 反事实 Counterfactual（Paper A 用）**
- `base_traj_id`
- `intervention_spec`（delete/swap/truncate/perturb + 位置/替换内容）
- `r_final_cf`
- `validity`（是否可执行、是否触发非法状态）

**(3) 分组 Group（Paper B 用）**
- `group_id`（难度簇/题型簇/解法模式簇）
- `group_features`（可选：base logprob、长度、是否写代码、是否调用工具等）

### 1.3 训练闭环（共享主流程）

1) `SFT/RFT warmstart`（可选，但强烈建议先稳住可用策略）
2) 并行 rollout → 得到 `(τ, r_final, verifier_info)`
3) 选择训练样本（成功/近成功/困难样本重采样）
4) 计算 advantage（A/B 插拔点）
5) RL update（GRPO/PPO…）+ 常规正则（KL/entropy）
6) 定期 eval：`pass@1/pass@k + cost + 多样性 + 稳定性`
7) 日志与可解释分析产物（论文图表）

---

## 2. Paper A：Verifiable Agent RL 的 Credit Assignment（Counterfactual Replay）

### 2.1 目标落点（实现视角）

把 `r_final` 变成 **step-level credit map**，再转为训练可用的 `A_t`，从而：
- 学得更快（样本效率）
- 更稳（seed 方差小）
- 更可解释（credit map 与关键步骤一致）

### 2.2 方法模块拆解

**A1) Intervention Sampler（反事实干预生成）**
- 输入：`τ`（含 steps、tool_meta、verifier_info）
- 输出：`{intervention_spec_k}`，每条轨迹抽 `K=2~8` 个干预
- 支持类型（与你草稿一致）：
  - `delete-step`：删 1 步或 1 段（block）
  - `swap-step`：同工具不同参数 / 同类 action 替换
  - `truncate`：截断到 t（识别“最早成功点”）
  - （可选）`perturb`：对工具输出/环境做轻噪声（鲁棒性 credit）
- 关键工程约束：
  - **合法性**：删步可能导致轨迹不可执行；优先在“可跳过动作”或 block 上操作
  - **成本**：只对高价值轨迹做（成功 + near-miss），并且 K 很小

**A2) Counterfactual Executor（回放 + verifier）**
- 输入：`(τ, intervention_spec)`
- 输出：`(τ_cf, r_final_cf, verifier_info_cf)`
- 需要：
  - 复现同一环境（deterministic seeds / tool 版本固定）
  - 缓存：相同 `(state_hash, action)` 的 tool 输出可复用（大幅省时）

**A3) Credit Estimator（从反事实到 credit）**
- 核心输出：`c_t` 或 `c_{t:t+k}`（block credit）
- 典型定义（按你草稿）：
  - `c_t = E[r(τ) - r(τ \\ a_t)]`（delete-step 的成功率下降）
  - `c_t = E[r(τ) - r(τ’)]`（swap 的边际影响）
  - `t* = min{t: truncate(τ,t) succeeds}` → 把 credit 集中到 `t*` 前后窗口
- 归一化与稳健：
  - `clip/normalize` 生成 `A_t`
  - 对不同长度轨迹做长度校正（避免长轨迹 credit 被稀释/放大）

**A4) Advantage Mapper（对接 GRPO/PPO）**
- step-level `A_t` → token-level（若 action 是文本 token）
- tool action：可将一个 tool-call 当作一个 action step（更干净）
- 对话式：把 credit 分配到对应 span（function-call JSON / code block）

### 2.3 训练与评测闭环（Paper A）

**环境（先稳后扩）**
- Code 修复：unit tests pass（推荐）
- SQL/DB：执行结果 exact match
- JSON/结构化：schema validator + exact match
- （可选）web benchmark：作为迁移与故事性补充

**主对比基线（必须干净）**
- Final-reward RL（credit 均匀摊到 token/step）
- 成功轨迹 RFT（筛选成功样本做 RFT）
- 仅 curriculum / 仅重采样（不做 credit）
- 仅加 KL/entropy（常规稳定化）

**关键消融（写进大纲就等于论文骨架）**
- 只用 delete / 只用 truncate / 只用 swap
- step vs block credit（block=2~4）
- K=2/4/8 的性价比曲线
- 只对成功轨迹 vs 成功+near-miss
- credit map 的“人类一致性”（抽样人工标注关键步骤做相关性）

**论文图表清单（实现需要提前埋点）**
- credit map 可视化（按 step/工具调用）
- sample-efficiency（同预算下 success 提升）
- 成本（tokens/tool calls）与 success 的 Pareto
- failure taxonomy（剩余失败类型）

---

## 3. Paper B：Conflict-aware RLVR（梯度冲突 → 模式塌缩 → 可修复）

### 3.1 目标落点（实现视角）

在提升 `pass@1` 的同时，**保住/提升**：
- `pass@k`（k=4/8/16）
- 多样性（unique solutions / AST 多样性 / n-gram 多样性）
- 稳定性（seed 方差、训练曲线波动）

### 3.2 方法模块拆解

**B1) Group Assignment（把样本分簇）**
- 方案 1（最稳）：按难度分组（2~4 组）
  - 用 base model 的 `logprob`、self-consistency、历史成功率估计难度
- 方案 2：按题型标签（math/code/logic…）
- 方案 3：按“解法模式”特征（是否写代码、是否长 CoT、是否工具调用）
- 输出：`group_id` 写进样本 metadata

**B2) Per-group Gradient（组梯度获取）**
- 对每组 `g` 计算 `∇_g`（对 RL objective 的梯度估计）
- 工程要点：
  - 组数小（2~8），否则显存/时间炸
  - 优先 LoRA/adapter 层算组梯度（成本可控）

**B3) Conflict Metric（冲突统计）**
- `dot(∇_g, ∇_h)` 或 `cos(∇_g, ∇_h)`
- 日志：冲突比例、冲突矩阵随训练演化

**B4) Conflict Surgery（核心算法层）**
- 当 `∇_g · ∇_h < 0` 时做投影/手术（PCGrad 风格）
- 聚合得到最终更新 `∇*`
- 触发式启用：冲突超过阈值才启用（省成本/更稳定）
- 可选优化：
  - 只对 adapter 层手术
  - 只对 top-N 最冲突组对做修正

### 3.3 训练与评测闭环（Paper B）

**复现“边界收缩”现象（先把问题做实）**
- 固定训练预算下：`pass@1 ↑`、但 `pass@k/多样性 ↓`
- 记录：冲突矩阵、各组贡献变化、样本分布变化

**对比基线（建议包含“最新 RLVR”口径的强基线）**
- 原始 GRPO/PPO（+KL/entropy）
- 数据侧：难度重采样/过滤（curriculum）
- 目标侧：改 divergence / advantage shaping / pass@k 直接优化 / uniqueness-aware

**关键消融**
- 分组策略（难度/题型/模式）+ 组数（2/4/8）
- surgery 触发阈值/频率
- surgery 作用范围（全参 vs LoRA 层）
- “只统计冲突不修正” vs “统计+修正”

**论文图表清单**
- 冲突矩阵热力图（随训练迭代）
- `pass@1/pass@k` 与冲突强度的关联曲线
- 多样性指标与冲突修正前后对比
- “被压掉的簇”案例分析（定性展示）

---

## 4. 相关工作与可引用基线（与你两篇的定位强相关）

### 4.1 Credit / Counterfactual（Paper A 相关）
- Mesnard et al., *Counterfactual Credit Assignment in Model-Free RL*（ICML 2021）
- Shapley / counterfactual credit（多智能体 RL，2023 arXiv）
- AGILE（NeurIPS 2024）这类 LLM agent RL 框架：可在 related work 里对齐“agent RL 训练范式”

### 4.2 RLVR / pass@k / collapse / divergence（Paper B 相关）
- DeepSeekMath 引入 GRPO（2024 arXiv）作为 GRPO 的常见引用入口
- PCGrad（NeurIPS 2020）作为“梯度冲突/手术”的最标准引用
- 近一年围绕 RLVR 的新口径（用于 baselines/讨论）：
  - Pass@K Policy Optimization（2025 arXiv）
  - PPO divergence choice 与奖励结构（2025 arXiv）
  - Advantage shaping for RLVR（2025 arXiv）
  - RLVR incentivizes correct reasoning（2025 arXiv）
  - Uniqueness-aware RL / Rewarding the Rare（2026 arXiv）

> 注：最后一组多为 arXiv，新鲜但未必都已顶会接收；用作 baseline/讨论时建议标清出处与时间。

---

## 5. 从今天开始的 8 周交付建议（可按实际资源调整）

以 **2026-02-04** 为 Week0 参考：

- W1（02-04~02-10）：统一环境 + baseline RLVR 跑通；打通日志与评测（pass@1/k）
- W2（02-11~02-17）：Paper A：实现 delete/truncate 两类 counterfactual + credit map
- W3（02-18~02-24）：Paper A：swap/block/缓存优化 + 初版消融
- W4（02-25~03-03）：Paper B：复现边界收缩 + 分组与冲突统计
- W5（03-04~03-10）：Paper B：surgery（PCGrad 风格）+ 初版消融
- W6（03-11~03-17）：两篇扩任务/扩模型规模（7B→14B/32B）、补强表格
- W7（03-18~03-24）：机制分析图（credit map / 冲突矩阵 / failure taxonomy）
- W8（03-25~03-31）：写作收敛：方法、实验、图表与 ablation 定稿


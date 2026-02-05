# Related Work 对照表（用于“撞车风险”评估）

> 备注：本表主要基于各论文 arXiv 页面摘要（abstract）与题面信息做“快速定位”。写论文前建议精读 PDF，补齐更细的差异点与实验设置。

## Paper A：Counterfactual / Intervention-based Credit Assignment（Agent + Verifiable Reward）

### 我们当前 A 的“最稳差异化”写法（建议钉死）

- **场景**：长轨迹 tool-using agent（code / sql / structured tools），而不是单轮推理题（math）。
- **监督信号**：仅依赖 **程序化 verifier** 的最终成功（可重放、可缓存），不依赖 PRM / LLM judge。
- **方法核心**：利用环境可重放做 **反事实重放**（delete / truncate / swap）估计 action/step 对最终成功的边际贡献 → credit map → step-level advantage → RL update。
- **论文贡献表达**：从“我也做 step credit assignment”改成“把 credit assignment 系统性推到 **agent long-horizon + pure verifiable reward** 的 setting，并给出可重放、可大规模的 counterfactual evaluator + 机制分析”。

### 可能最像的近年工作（用于对照）

- **InT: Self-Proposed Interventions Enable Credit Assignment in LLM Reasoning**（arXiv:2601.14209, 2026）  
  - 摘要要点：用参考解（reference solutions）定位 first error，并提出单步 intervention；用 SFT 作为更好 RL 初始化。  
  - 与我们的关键差异点（建议强调）：  
    - 我们不假设 reference solution 可用；我们只依赖 verifier。  
    - 我们是环境重放的 counterfactual（删/换/截断），而不是“对推理链做编辑并继续采样”的训练范式。  
    - 我们目标是 tool agent/多 action type 的 long-horizon credit map（可用于 cost-aware）。

- **Pinpointing crucial steps: Attribution-based Credit Assignment for Verifiable RL (ACPO)**（arXiv:2510.08899, 2025）  
  - 摘要要点：difficulty-aware curriculum + 语义分段 + attribution 表示；同时调控熵避免 collapse，并做分解式 reward/贡献度。  
  - 与我们的关键差异点：  
    - 我们把 “credit 估计”建立在 **counterfactual outcome 差值** 上（可重放、可解释）；而 ACPO 偏向 attribution/分段 + 熵调控框架。  
    - 我们更偏 agent 可执行的干预集合（delete/truncate/swap），并给出干预有效性/开销的系统分析。

- **Stop Summation: Min-Form Credit Assignment Is All PRM Needs (PURE)**（arXiv:2504.15275, NeurIPS 2025）  
  - 摘要要点：讨论 PRM 场景下 reward hacking；提出 min-form credit assignment 稳定训练。  
  - 与我们的关键差异点：  
    - 我们不训练/依赖 PRM；我们直接用 verifier + counterfactual 重放做 step credit（避免 PRM reward hacking 依赖）。  

## Paper B：Conflict / Diversity Collapse / pass@k（RLVR）

### 当前 B（“梯度冲突 + PCGrad 手术”）的风险点

你原始叙事「冲突梯度 → 多样性/Pass@k 塌缩 → gradient surgery 修复」在 2025-2026 的 RLVR 文献里 **很拥挤**，且已有工作直接点名 GRPO/熵塌缩/Pass@k 退化并给出修复。

### 与 B 叙事最直接冲突的代表性工作（用于对照）

- **GTPO: Stabilizing GRPO via Gradient and Entropy Control**（arXiv:2508.03772, 2025）  
  - 摘要要点：分析 GRPO 的 token-level penalization（冲突梯度）与 policy collapse；提出 conflict token 保护 + 熵阈值过滤。  
  - 对 B 的影响：你如果主打“GRPO 冲突梯度导致塌缩”，很容易被认为是同一问题空间的变体。

- **Pass@K Policy Optimization (PKPO)**（arXiv:2505.15201, 2025）  
  - 摘要要点：通过 reward 变换直接优化 pass@k，并给出无偏低方差估计；还支持训练中 anneal k。  
  - 对 B 的影响：你如果主打“保住 pass@k”，PKPO 是强相关 baseline/对手。

- **SimKO: Simple Pass@K Policy Optimization**（arXiv:2510.14807, 2025）  
  - 摘要要点：分析 token-level probability 过度集中导致 pass@k 下降；提出非对称更新缓解过度集中。  
  - 对 B 的影响：同样把“pass@1 ↑ / pass@k ↓”机制化，并给出更新规则。

- **DPH-RL: The Choice of Divergence… Mitigating Diversity Collapse**（arXiv:2509.07430, 2025）  
  - 摘要要点：把多样性塌缩与遗忘归因到 divergence 选择；用 forward-KL/JS 等 mass-covering divergence 做 rehearsal 机制，提升 pass@1 & pass@k。  
  - 对 B 的影响：直接占据了“多样性塌缩 ↔ divergence 设计”这条主线。

### B 还想保住的话：建议“换叙事/换切面”（两条更可能有空间的方向）

1) **从“单域 pass@k”换到“跨任务/跨技能的负迁移（interference）”**  
   - 场景：tool agent / multi-domain RLVR（code+sql+math 等）里的能力互相伤害与遗忘。  
   - 主张：不是“提升 pass@k”本身，而是“保持各簇/各技能的 retention + fairness”，pass@k 只是其中一个指标。  
   - 这样能与 PKPO/SimKO/DPH-RL/GTPO 拉开一些距离（他们多在单域推理或单任务分布上做）。

2) **把“冲突检测/手术”绑定到 agent setting 的结构性对象**  
   - 例如：按 **action type / tool type / subgoal stage** 分组，并只在 LoRA/adapter 层做低开销冲突统计与触发式 surgery；强调这是 agent long-horizon 的结构化冲突，而不是 token-level 冲突。


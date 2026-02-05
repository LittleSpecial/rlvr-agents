# NeurIPS 2026 Paper A（Agent / “抽奖”向）— Verifiable Agent RL 的信用分配

> 目标：做一篇 **agent 向**、但尽量避免“纯系统工程 + 跑榜讲故事”的路线；核心贡献放在 **长轨迹稀疏奖励下的信用分配**（credit assignment），从而与常见的 end-to-end web RL（如 multi-turn GRPO/端到端直接优化最终成功率）区分开。

## 1. 问题定义（Problem）

### 1.1 现象
- tool/web agent 大多只有 **最终二值成功奖励**（task success / exact match / unit test pass / URL match 等）。
- 轨迹长、探索难：一次成功往往依赖少数关键步骤，但训练时梯度/优势信号被“平均”到整条轨迹，导致：
  - 学得慢、样本效率低；
  - 容易过拟合少数模板策略；
  - reward hacking / shortcut 行为难定位。

### 1.2 研究问题
**在只给最终可验证 reward 的前提下，如何自动识别“关键步骤”，并把 credit 更合理地分配给这些步骤，从而提升学习稳定性与泛化？**

## 2. 核心 idea（One-liner）

用 **反事实重放 (counterfactual replay)** 在训练时对轨迹做轻量的“删步/替步/换工具参数”干预，估计每一步对最终成功的 **边际贡献**，把稀疏的最终 reward 变成更信息化的 step-level advantage / credit map，再用于 RL 更新。

直觉：把“这条轨迹成功了”改成“它成功主要因为第 t 步做对了；删掉第 t 步就失败”，从而更像监督信号。

## 3. 方法框架（Method）

### 3.1 轨迹表示
- 轨迹：`(s1, a1, ... , sT, aT, r_final)`，其中 `r_final ∈ {0,1}` 由程序化 verifier 给出。
- `ai` 可以是：
  - tool agent：`call(tool, args)` / `write(code)` / `run_tests()` 等；
  - web agent：`click/typing/scroll` 等。

### 3.2 反事实干预集合（Interventions）
对每条 rollout 轨迹抽样若干个干预，得到反事实轨迹 `τ'` 并重新用 verifier 计算 `r'(τ')`：
- **Delete-step**：删除某一步（或一个短片段），看是否仍成功；
- **Swap-step**：把某一步替换为“相似但不同”的动作（例如不同的工具参数、不同的检索 query）；
- **Truncate**：截断到某一步，看是否已经足够成功（用来识别“成功发生的最早时间点”）；
- （可选）**Replay-with-perturbation**：对环境轻微扰动（比如工具输出噪声/网页 DOM 轻微改动）来识别“脆弱步骤”。

### 3.3 从反事实到 credit（Credit Estimation）
- 定义每一步的贡献分数 `c_t`，例如：
  - `c_t = E[r(τ) - r(τ \ a_t)]`（删掉第 t 步后的成功率下降）
  - 或对短片段 `t:t+k` 做 block contribution。
- 把 `c_t` 映射为 step-level advantage：`A_t = normalize(c_t)`，用于策略梯度更新：
  - 简化版：把 `A_t` 作为 token-level reward/advantage（适配 GRPO/PPO 类实现）。

### 3.4 训练与基线
训练主线可以选一个你们更好复现的 RL recipe：
- GRPO / PPO（对话式 LM RL 常用的 pipeline）
- 或者“success-only”的 REINFORCE（配合 strong baseline 的 SFT / RFT）

对比基线（必须做干净）：
- End-to-end multi-turn RL（只用最终 reward，均匀分配到 token/step）
- 仅加 KL/entropy 正则
- 仅用轨迹筛选（只保留成功轨迹做 RFT）
- 仅用 curriculum（从易到难采样）

## 4. 环境与数据（Environment）

### 4.1 为了 4 月前出稿：优先选 tool/code agent
建议先做 **可本地验证** 的环境（低工程风险，高吞吐）：
- 代码修复/单测通过（reward=unit tests pass）
- SQL 生成（reward=query result exact match）
- JSON/结构化输出（reward=parser/validator）

这样反事实重放也很方便：删一步、改参数，重新运行程序即可得到准确 reward。

### 4.2 可选：再迁移到 web agent 做“外显故事”
如果你确实想要 web 味道：
- 先在 tool 环境把方法跑稳；
- 再挑一个 web benchmark 做补充实验（强调“同一个 credit 分配机制也适用 web”）。

## 5. 评测与分析（Eval）

### 5.1 主指标
- 成功率 / pass@1（多次 rollout 的 success@k 也可以）
- 平均 step 数 / token 数 / tool 调用成本（cost-aware）
- 训练稳定性：学习曲线方差、seed 稳定性

### 5.2 机制分析（把“像研究”讲清楚）
必须给出 “why it works” 的证据：
- credit map 可视化：哪些步骤拿到高 credit，与人工观察是否一致
- ablation：去掉 delete-step / truncate / block contribution 会怎样
- failure taxonomy：哪些任务仍失败（长程依赖？信息不足？工具错误？）

## 6. 资源与工程实现（Implementation Notes）

### 6.1 建议的模型与训练策略（适配 8×A800）
- 先用 `Qwen2.5-7B/14B` + LoRA 把 pipeline 跑通（吞吐/迭代快）
- 最后用 `Qwen2.5-32B` 做主表（如果吞吐允许）

### 6.2 系统组件（最小实现）
- 环境 runner（可并行 rollout）
- verifier（程序化）
- counterfactual generator（从一条轨迹生成 K 个反事实）
- trainer（GRPO/PPO/RFT，接收 step-level advantage）
- logging：每个 step 的 credit、反事实成功率、轨迹长度分布

## 7. 风险点与规避（Risks）
- 反事实重放开销大：只对“高价值轨迹”做（成功轨迹+接近成功轨迹），K 取很小（如 2–8）。
- 反事实不合理（删步导致轨迹非法）：限制在“可跳过的动作”或做 block-level。
- reviewer 觉得“像 feature engineering”：一定要做机制分析 + 多环境验证 + 证明可泛化。

## 8. 里程碑（粗略）
- W1–W2：选环境 + 跑通 end-to-end RL baseline
- W3–W4：实现 counterfactual credit + 初版 ablation
- W5–W6：扩任务/扩模型 + 写机制分析图表
- W7–W8：补 web/故事性实验（可选）+ 成稿


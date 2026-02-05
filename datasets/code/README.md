# Code 数据集格式（JSONL）

本项目的 `CodeEnv` 使用“单元测试/断言通过”作为可验证奖励（`reward=1` 表示测试全部通过，否则为 `0`）。

## 1) 文件格式

- **JSONL**：一行一个 JSON 对象（一个任务）。
- 建议把大文件（真实数据集 dump）放在本地/服务器，**不要提交到 GitHub**；仓库里只保留脚本与 schema。

## 2) 每条样本的字段

必需字段：

- `task_id`：`str`，全局唯一
- `prompt`：`str`，给模型/agent 的题面（可以包含函数签名、约束、示例等）

强烈推荐字段（用于 verifier）：

- `test_code`：`str`，Python 测试代码。要求：
  - 与模型生成的 `code` 拼接后可直接 `python file.py` 运行
  - 通过时 **进程退出码为 0**；失败时非 0（`assert` / 抛异常即可）
  - 尽量 deterministic（避免随机数、时间、网络等）

可选字段：

- `initial_code`：`str`，初始代码（做 code-repair/填空很有用；纯 code-gen 可为空）
- `expected_output`：`str`，无测试时可用“输出精确匹配”作为 verifier（不推荐，太弱）
- `language`：`str`，默认 `"python"`
- `timeout`：`float`，默认 `30.0`
- `metadata`：`dict`，建议包含：
  - `source`：`"mbpp" / "openai_humaneval" / "synthetic"` 等
  - `split`：`"train" / "val" / "test"`
  - `task_type`：用于 Paper B 分组（例如 `"easy" / "hard"` 或 `"type0" / "type1"`）

## 3) 示例

见 `datasets/code/example.jsonl`。

## 4) 推荐起步数据（真实 benchmark）

优先用“可复现/公开”的代码基准：

- **训练**：MBPP（规模更大，适合先把曲线跑出来）
- **评测**：HumanEval + MBPP test（报告 pass@1 / pass@k + 多样性）

可以用 HuggingFace `datasets` 下载后，导出成上面的 JSONL（仓库自带脚本）：

```bash
python3 -m pip install datasets

python3 scripts/prepare_code_dataset_hf.py \
  --dataset mbpp --split train \
  --out datasets/code/mbpp_train.jsonl

python3 scripts/prepare_code_dataset_hf.py \
  --dataset openai_humaneval --split test \
  --out datasets/code/humaneval_test.jsonl

python3 scripts/validate_code_jsonl.py datasets/code/mbpp_train.jsonl
```

## 5) 合成数据（你现在“只有卡没数据”时的方案）

合成数据可以用来扩训练集，但 **评测集务必用独立的公开 benchmark**。

一个比较稳的合成流程：

1. LLM 生成：`prompt + reference_solution + tests`
2. 运行 reference_solution，确保 tests 真的能区分对错
3. 用另一模型/另一提示再生成额外 tests（增强鲁棒性）
4. 去重（prompt 相似度、函数签名、AST hash）+ 过滤不稳定样本（随机/超时/依赖网络）


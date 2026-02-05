"""
Paper A 训练入口
Verifiable Agent RL with Counterfactual Credit Assignment
"""

import sys
import argparse
import statistics
from pathlib import Path
from typing import Dict, List, Tuple

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from shared.experiment_tracking import ExperimentTracker, ExperimentConfig, RunMetrics
from shared.envs import CodeEnv, EnvConfig, SQLEnv
from shared.envs.base import Action, ActionType, Trajectory
from shared.toy import ToyCategoricalPolicy, get_toy_tasks

from paper_a_credit_assignment import (
    CounterfactualGenerator,
    CounterfactualExecutor,
    CreditEstimator,
    AdvantageMapper
)


def parse_args():
    parser = argparse.ArgumentParser(description="Paper A: Counterfactual Credit Assignment")
    
    # 基本配置
    parser.add_argument("--experiment_name", type=str, default="credit_assignment")
    parser.add_argument("--description", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--backend", type=str, default="toy", choices=["toy", "hf"])
    
    # 模型配置
    parser.add_argument("--model_name", type=str, default="Qwen2.5-7B")
    parser.add_argument("--use_lora", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lora_rank", type=int, default=64)
    
    # 训练配置
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--toy_lr", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--num_rollouts_per_prompt", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    
    # 环境配置
    parser.add_argument("--env_type", type=str, default="code", choices=["code", "sql"])
    parser.add_argument("--max_trajectory_length", type=int, default=20)
    
    # Paper A 特定配置
    parser.add_argument("--use_counterfactual_credit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--counterfactual_k", type=int, default=4)
    parser.add_argument("--intervention_types", nargs="+", default=["delete", "truncate"])
    parser.add_argument("--credit_normalization", type=str, default="signed",
                        choices=["signed", "minmax", "zscore", "softmax", "none"])
    
    # 输出配置
    parser.add_argument("--output_dir", type=str, default="./experiments")
    
    return parser.parse_args()


def _compute_grpo_advantages(trajectories: List[Trajectory]) -> List[float]:
    rewards = [float(t.r_final) for t in trajectories]
    task_groups: Dict[str, List[int]] = {}
    for i, traj in enumerate(trajectories):
        task_groups.setdefault(traj.task_id, []).append(i)

    advantages = [0.0 for _ in trajectories]
    for indices in task_groups.values():
        group_rewards = [rewards[i] for i in indices]
        if len(group_rewards) > 1:
            mean = statistics.fmean(group_rewards)
            std = statistics.pstdev(group_rewards)
            if std > 1e-8:
                for i in indices:
                    advantages[i] = (rewards[i] - mean) / std
            else:
                for i in indices:
                    advantages[i] = rewards[i] - 0.5
        else:
            i = indices[0]
            advantages[i] = rewards[i] - 0.5
    return advantages


def _to_list(x) -> List[float]:
    if isinstance(x, list):
        return [float(v) for v in x]
    if hasattr(x, "tolist"):
        return [float(v) for v in x.tolist()]
    return [float(v) for v in x]


def _rollout_toy(
    env,
    env_type: str,
    toy_task,
    policy: ToyCategoricalPolicy,
    *,
    temperature: float,
) -> Tuple[Trajectory, int]:
    sample = policy.sample(temperature=temperature)
    candidate = toy_task.candidates[sample.action_index]

    env.reset(dict(toy_task.task))
    if env_type == "code":
        env.step(Action(ActionType.CODE_WRITE, candidate, metadata={"logprob": sample.logprob}))
        env.step(Action(ActionType.TOOL_CALL, "", tool_name="run_tests"))
    else:
        env.step(Action(ActionType.TOOL_CALL, candidate, tool_name="submit_query", metadata={"logprob": sample.logprob}))

    traj = env.get_trajectory()
    traj.metadata["toy_action_index"] = sample.action_index
    return traj, sample.action_index


def _pass_at_k_from_groups(trajectories: List[Trajectory], k: int) -> float:
    groups: Dict[str, List[Trajectory]] = {}
    for traj in trajectories:
        groups.setdefault(traj.task_id, []).append(traj)

    if not groups:
        return 0.0

    hits = 0
    for ts in groups.values():
        ts_k = ts[:k]
        if any(t.r_final > 0.5 for t in ts_k):
            hits += 1
    return hits / len(groups)


def main():
    args = parse_args()
    
    # 创建实验配置
    config = ExperimentConfig(
        experiment_name=args.experiment_name,
        project="paper_a",
        description=args.description,
        model_name=args.model_name,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        algorithm=("TOY_PG" if args.backend == "toy" else "GRPO"),
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        env_type=args.env_type,
        max_trajectory_length=args.max_trajectory_length,
        use_counterfactual_credit=args.use_counterfactual_credit,
        counterfactual_k=args.counterfactual_k,
        intervention_types=args.intervention_types,
        seed=args.seed
    )
    
    # 初始化实验追踪器
    tracker = ExperimentTracker(config, base_dir=args.output_dir)
    tracker.log_event("init", "Experiment initialized")
    
    # 创建环境
    env_config = EnvConfig(
        name=args.env_type,
        max_steps=args.max_trajectory_length,
        seed=args.seed
    )
    
    if args.env_type == "code":
        env = CodeEnv(env_config)
    else:
        env = SQLEnv(env_config)
    
    # 创建Paper A组件
    cf_generator = CounterfactualGenerator(
        intervention_types=args.intervention_types,
        k=args.counterfactual_k,
        seed=args.seed
    )
    cf_executor = CounterfactualExecutor(env, use_cache=True, seed=args.seed)
    credit_estimator = CreditEstimator(normalization=args.credit_normalization)
    advantage_mapper = AdvantageMapper(level="step")
    
    tracker.log_event("components", "Paper A components initialized", {
        "intervention_types": args.intervention_types,
        "k": args.counterfactual_k
    })
    
    print(f"Experiment directory: {tracker.experiment_dir}")
    print(f"Config saved to: {tracker.experiment_dir / 'config.json'}")
    print("\n" + "="*50)
    print("Paper A: Counterfactual Credit Assignment")
    print("="*50)
    print(f"Model: {args.model_name}")
    print(f"Environment: {args.env_type}")
    print(f"Counterfactual K: {args.counterfactual_k}")
    print(f"Interventions: {args.intervention_types}")
    print("="*50)

    if args.backend == "hf":
        tracker.log_event("error", "HF backend not implemented yet in this repo skeleton")
        raise NotImplementedError(
            "HF backend is not implemented yet. Start with `--backend toy` to validate the pipeline, "
            "then we can plug in transformers + torch for real LLM RLVR training."
        )

    # === Toy backend: end-to-end runnable pipeline (no torch/transformers required) ===
    toy_tasks = get_toy_tasks(args.env_type, variant="paper_a")
    if not toy_tasks:
        raise RuntimeError("No toy tasks available")

    num_actions = len(toy_tasks[0].candidates)
    if any(len(t.candidates) != num_actions for t in toy_tasks):
        raise ValueError("All toy tasks must share the same number of candidates")

    policy = ToyCategoricalPolicy(num_actions, seed=args.seed)
    tracker.log_event("toy", "Toy backend enabled", {"num_actions": num_actions, "num_tasks": len(toy_tasks)})

    for step in range(1, args.max_steps + 1):
        # Collect rollouts
        trajectories: List[Trajectory] = []
        chosen_indices: List[int] = []

        # Sample tasks with replacement
        for _ in range(args.batch_size):
            task = policy.rng.choice(toy_tasks)
            for _ in range(args.num_rollouts_per_prompt):
                traj, a_idx = _rollout_toy(env, args.env_type, task, policy, temperature=args.temperature)
                trajectories.append(traj)
                chosen_indices.append(a_idx)

        # GRPO-style group advantages (group by task_id)
        advantages = _compute_grpo_advantages(trajectories)

        # Optional: counterfactual credit reweighting
        weights: List[float] = []
        credit_spreads: List[float] = []
        for traj, adv in zip(trajectories, advantages):
            w = float(adv)
            if args.use_counterfactual_credit and cf_generator.should_generate_counterfactuals(traj):
                interventions = cf_generator.generate(traj)
                cf_results = cf_executor.batch_execute(traj, interventions)
                credit_map = credit_estimator.estimate(traj, cf_results)
                credit_spreads.append(credit_map.spread)

                step_adv = _to_list(advantage_mapper.map_to_step_advantages(traj, credit_map))
                lp_steps = [i for i, s in enumerate(traj.steps) if s.logprob is not None]
                if lp_steps:
                    credit_weight = sum(step_adv[i] for i in lp_steps) / len(lp_steps)
                    w *= float(credit_weight)
            weights.append(w)

        update_info = policy.reinforce_update(chosen_indices, weights, lr=args.toy_lr, temperature=args.temperature)

        if step % args.log_interval == 0:
            success_rate = sum(t.r_final for t in trajectories) / max(1, len(trajectories))
            pass_at_1 = _pass_at_k_from_groups(trajectories, k=1)
            pass_at_k = _pass_at_k_from_groups(trajectories, k=args.num_rollouts_per_prompt)
            avg_len = sum(t.length for t in trajectories) / max(1, len(trajectories))
            avg_credit_spread = (sum(credit_spreads) / len(credit_spreads)) if credit_spreads else 0.0

            tracker.log_metrics(RunMetrics(
                step=step,
                train_loss=0.0,
                success_rate=float(success_rate),
                pass_at_1=float(pass_at_1),
                pass_at_k={int(args.num_rollouts_per_prompt): float(pass_at_k)},
                avg_trajectory_length=float(avg_len),
                avg_credit_spread=float(avg_credit_spread),
            ))
            tracker.log_event("train", "logged step metrics", update_info)

        if step % args.eval_interval == 0:
            tracker.log_event("eval", "eval checkpoint", {"step": step})

    tracker.finalize()


if __name__ == "__main__":
    main()

"""
Paper B 训练入口
Conflict-aware RLVR for LLM Reasoning
"""

import sys
import argparse
import statistics
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.append(str(Path(__file__).parent.parent))

from shared.experiment_tracking import ExperimentTracker, ExperimentConfig, RunMetrics
from shared.envs import CodeEnv, EnvConfig, SQLEnv
from shared.envs.base import Action, ActionType, Trajectory
from shared.toy import ToyCategoricalPolicy, get_toy_tasks, pcgrad, compute_conflict_metrics

from paper_b_conflict_aware import (
    GroupAssigner,
    GroupConfig,
)
from paper_b_conflict_aware.group_assignment import GroupStrategy


def parse_args():
    parser = argparse.ArgumentParser(description="Paper B: Conflict-aware RLVR")
    
    # 基本配置
    parser.add_argument("--experiment_name", type=str, default="conflict_aware")
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
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--num_rollouts_per_prompt", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    
    # 环境配置
    parser.add_argument("--env_type", type=str, default="code", choices=["code", "sql"])
    
    # Paper B 特定配置
    parser.add_argument("--group_strategy", type=str, default="task_type",
                        choices=["difficulty", "task_type", "solution_pattern", "hybrid"])
    parser.add_argument("--num_groups", type=int, default=4)
    parser.add_argument("--conflict_threshold", type=float, default=0.0)
    parser.add_argument("--surgery_method", type=str, default="pcgrad",
                        choices=["pcgrad", "cagrad", "mgda", "none"])
    
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


def _rollout_toy(env, env_type: str, toy_task, policy: ToyCategoricalPolicy, *, temperature: float) -> Tuple[Trajectory, int]:
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


def _diversity_from_groups(trajectories: List[Trajectory], k: int) -> float:
    groups: Dict[str, List[Trajectory]] = {}
    for traj in trajectories:
        groups.setdefault(traj.task_id, []).append(traj)
    if not groups:
        return 0.0
    vals: List[float] = []
    for ts in groups.values():
        ts_k = ts[:k]
        idxs = [t.metadata.get("toy_action_index") for t in ts_k]
        idxs = [i for i in idxs if isinstance(i, int)]
        if not idxs:
            vals.append(0.0)
            continue
        vals.append(len(set(idxs)) / max(1, min(k, len(idxs))))
    return float(sum(vals) / len(vals))


def main():
    args = parse_args()
    
    # 创建实验配置
    config = ExperimentConfig(
        experiment_name=args.experiment_name,
        project="paper_b",
        description=args.description,
        model_name=args.model_name,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        algorithm=("TOY_PG" if args.backend == "toy" else "GRPO"),
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        env_type=args.env_type,
        use_conflict_aware=True,
        num_groups=args.num_groups,
        surgery_method=args.surgery_method,
        seed=args.seed
    )
    
    # 初始化实验追踪器
    tracker = ExperimentTracker(config, base_dir=args.output_dir)
    tracker.log_event("init", "Experiment initialized")
    
    # 创建Paper B组件
    group_config = GroupConfig(
        strategy=GroupStrategy(args.group_strategy),
        num_groups=args.num_groups
    )
    group_assigner = GroupAssigner(group_config)
    
    tracker.log_event("components", "Paper B components initialized", {
        "group_strategy": args.group_strategy,
        "num_groups": args.num_groups,
        "surgery_method": args.surgery_method
    })
    
    print(f"Experiment directory: {tracker.experiment_dir}")
    print("\n" + "="*50)
    print("Paper B: Conflict-aware RLVR")
    print("="*50)
    print(f"Model: {args.model_name}")
    print(f"Environment: {args.env_type}")
    print(f"Group Strategy: {args.group_strategy}")
    print(f"Num Groups: {args.num_groups}")
    print(f"Surgery Method: {args.surgery_method}")
    print("="*50)

    if args.backend == "hf":
        tracker.log_event("error", "HF backend not implemented yet in this repo skeleton")
        raise NotImplementedError(
            "HF backend is not implemented yet. Start with `--backend toy` to validate the pipeline, "
            "then we can plug in transformers + torch for real LLM RLVR training."
        )

    # Create env
    env_config = EnvConfig(name=args.env_type, max_steps=20, seed=args.seed)
    if args.env_type == "code":
        env = CodeEnv(env_config)
    else:
        env = SQLEnv(env_config)

    toy_tasks = get_toy_tasks(args.env_type, variant="paper_b")
    if not toy_tasks:
        raise RuntimeError("No toy tasks available")

    num_actions = len(toy_tasks[0].candidates)
    if any(len(t.candidates) != num_actions for t in toy_tasks):
        raise ValueError("All toy tasks must share the same number of candidates")

    policy = ToyCategoricalPolicy(num_actions, seed=args.seed)
    tracker.log_event("toy", "Toy backend enabled", {"num_actions": num_actions, "num_tasks": len(toy_tasks)})

    for step in range(1, args.max_steps + 1):
        trajectories: List[Trajectory] = []
        chosen_indices: List[int] = []

        for _ in range(args.batch_size):
            task = policy.rng.choice(toy_tasks)
            for _ in range(args.num_rollouts_per_prompt):
                traj, a_idx = _rollout_toy(env, args.env_type, task, policy, temperature=args.temperature)
                trajectories.append(traj)
                chosen_indices.append(a_idx)

        advantages = _compute_grpo_advantages(trajectories)

        # Assign groups
        group_ids = group_assigner.assign(trajectories)

        # Compute per-group gradients of the (surrogate) loss: L = -adv * logprob
        probs = policy.probs(temperature=args.temperature)
        group_grads: Dict[int, List[float]] = {}
        for traj, adv, gid in zip(trajectories, advantages, group_ids):
            a = traj.metadata.get("toy_action_index")
            if not isinstance(a, int):
                continue
            grad_loss = [0.0 for _ in range(num_actions)]
            for j in range(num_actions):
                grad_loss[j] = -float(adv) * ((1.0 if j == a else 0.0) - probs[j])
            if gid not in group_grads:
                group_grads[gid] = [0.0 for _ in range(num_actions)]
            for j in range(num_actions):
                group_grads[gid][j] += grad_loss[j]

        # Conflict metrics
        conflict = compute_conflict_metrics(group_grads, conflict_threshold=args.conflict_threshold)

        # Gradient surgery / aggregation
        if args.surgery_method == "pcgrad":
            final_grad = pcgrad(group_grads, conflict_threshold=args.conflict_threshold)
        elif args.surgery_method == "none":
            final_grad = [0.0 for _ in range(num_actions)]
            if group_grads:
                for g in group_grads.values():
                    for j in range(num_actions):
                        final_grad[j] += g[j] / len(group_grads)
        else:
            # Toy backend only implements PCGrad; other methods fall back to mean aggregation.
            final_grad = [0.0 for _ in range(num_actions)]
            if group_grads:
                for g in group_grads.values():
                    for j in range(num_actions):
                        final_grad[j] += g[j] / len(group_grads)

        policy.apply_grad(final_grad, lr=args.toy_lr, ascent=False)

        if step % args.log_interval == 0:
            success_rate = sum(t.r_final for t in trajectories) / max(1, len(trajectories))
            pass_at_1 = _pass_at_k_from_groups(trajectories, k=1)
            pass_at_k = _pass_at_k_from_groups(trajectories, k=args.num_rollouts_per_prompt)
            diversity = _diversity_from_groups(trajectories, k=args.num_rollouts_per_prompt)

            tracker.log_metrics(RunMetrics(
                step=step,
                train_loss=0.0,
                success_rate=float(success_rate),
                pass_at_1=float(pass_at_1),
                pass_at_k={int(args.num_rollouts_per_prompt): float(pass_at_k)},
                avg_trajectory_length=float(sum(t.length for t in trajectories) / max(1, len(trajectories))),
                conflict_ratio=float(conflict.conflict_ratio),
                solution_diversity=float(diversity),
            ))
            tracker.log_event("conflict", "logged conflict metrics", {
                "avg_angle_deg": conflict.avg_angle_deg,
                "group_norms": conflict.group_norms,
            })

    tracker.finalize()


if __name__ == "__main__":
    main()

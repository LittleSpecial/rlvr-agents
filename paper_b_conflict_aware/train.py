"""
Paper B 训练入口
Interference-aware RLVR for Multi-Skill Code Models (toy backend)

This repo currently provides an end-to-end runnable toy backend (no torch/transformers required),
so you can validate the algorithmic pipeline + logging before plugging in a real LLM backend.
"""

import sys
import argparse
import statistics
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

sys.path.append(str(Path(__file__).parent.parent))

from shared.experiment_tracking import ExperimentTracker, ExperimentConfig, RunMetrics
from shared.envs import CodeEnv, EnvConfig, SQLEnv
from shared.envs.base import Action, ActionType, Trajectory
from shared.toy import ToyCategoricalPolicy, get_toy_tasks, compute_conflict_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Paper B: Interference-aware RLVR (toy backend)")

    # NOTE: We avoid argparse.BooleanOptionalAction for Python 3.8 compatibility on some clusters.
    def add_bool_flag(name: str, *, default: bool, help: str):
        dest = name.lstrip("-")
        parser.add_argument(name, dest=dest, action="store_true", help=help)
        parser.add_argument(f"--no-{dest}", dest=dest, action="store_false", help=argparse.SUPPRESS)
        parser.set_defaults(**{dest: default})

    # 基本配置
    parser.add_argument("--experiment_name", type=str, default="interference_aware")
    parser.add_argument("--description", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--backend", type=str, default="toy", choices=["toy", "hf"])

    # 模型配置
    parser.add_argument("--model_name", type=str, default="Qwen2.5-7B")
    add_bool_flag("--use_lora", default=True, help="(hf) enable LoRA adapters")
    parser.add_argument("--lora_rank", type=int, default=64)

    # 训练配置
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--toy_lr", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=32, help="Number of target-skill prompts per step")
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--eval_interval", type=int, default=200)
    parser.add_argument("--num_rollouts_per_prompt", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)

    # 环境配置
    parser.add_argument("--env_type", type=str, default="code", choices=["code", "sql"])
    add_bool_flag("--show_tests", default=True, help="(code env) whether to show unit tests in the observation")

    # Multi-skill protocols
    parser.add_argument("--protocol", type=str, default="mixed", choices=["mixed", "sequential"])
    parser.add_argument("--skill_key", type=str, default="task_type", help="Trajectory.metadata key for skill id")
    parser.add_argument("--skill_sequence", nargs="*", default=None,
                        help="(sequential) skill ids in order, e.g. type0 type1")
    parser.add_argument("--stage_steps", type=int, default=2000, help="(sequential) steps per stage")

    # Skill-safe constraints
    parser.add_argument("--projection", type=str, default="sequential_margin",
                        choices=["sequential_margin", "pcgrad", "none"])
    parser.add_argument("--epsilon", type=float, default=0.0, help="Constraint slack: dot(g_s, Δθ) >= -epsilon")
    parser.add_argument("--memory_per_protected", type=int, default=8,
                        help="(sequential) prompts sampled per protected skill per step to estimate constraint grads")
    add_bool_flag("--normalize_update", default=True, help="Normalize projected update to match candidate gradient norm")

    # Interference stats (analysis only)
    parser.add_argument("--conflict_threshold", type=float, default=0.0,
                        help="Cosine threshold for counting interfering skill pairs")

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


def _skill_id(traj: Trajectory, *, skill_key: str) -> str:
    val = traj.metadata.get(skill_key, "unknown")
    return str(val)


def _dot(a: List[float], b: List[float]) -> float:
    return float(sum(x * y for x, y in zip(a, b)))


def _norm(a: List[float]) -> float:
    return (_dot(a, a) + 1e-12) ** 0.5


def _sequential_margin_project(
    candidate: List[float],
    constraints: List[List[float]],
    *,
    epsilon: float,
    normalize_to: Optional[float],
) -> List[float]:
    """
    Sequentially project `candidate` onto halfspaces:
        dot(g_s, Δθ) >= -epsilon
    using minimal L2 adjustment along g_s.
    """
    out = candidate[:]
    for g_s in constraints:
        d = _dot(g_s, out)
        if d < -epsilon:
            denom = _dot(g_s, g_s) + 1e-12
            alpha = (-epsilon - d) / denom
            out = [x + alpha * y for x, y in zip(out, g_s)]

    if normalize_to is not None:
        n = _norm(out)
        if n > 1e-12:
            scale = normalize_to / n
            out = [x * scale for x in out]
    return out


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


def _eval_greedy_pass_at_1(
    env,
    env_type: str,
    toy_tasks: List,
    policy: ToyCategoricalPolicy,
    *,
    temperature: float,
) -> float:
    probs = policy.probs(temperature=temperature)
    best_idx = max(range(len(probs)), key=lambda i: probs[i])

    successes = 0.0
    for task in toy_tasks:
        candidate = task.candidates[best_idx]
        env.reset(dict(task.task))
        if env_type == "code":
            env.step(Action(ActionType.CODE_WRITE, candidate, metadata={"logprob": 0.0}))
            env.step(Action(ActionType.TOOL_CALL, "", tool_name="run_tests"))
        else:
            env.step(Action(ActionType.TOOL_CALL, candidate, tool_name="submit_query", metadata={"logprob": 0.0}))
        traj = env.get_trajectory()
        successes += float(traj.r_final)
    return successes / max(1, len(toy_tasks))


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
        num_groups=0,
        surgery_method=args.projection,
        seed=args.seed,
        extra={
            "protocol": args.protocol,
            "skill_key": args.skill_key,
            "epsilon": args.epsilon,
            "memory_per_protected": args.memory_per_protected,
        },
    )

    tracker = ExperimentTracker(config, base_dir=args.output_dir)
    tracker.log_event("init", "Experiment initialized")
    exp_wall_start = time.time()

    print(f"Experiment directory: {tracker.experiment_dir}")
    print("\n" + "="*50)
    print("Paper B: Interference-aware RLVR (toy backend)")
    print("="*50)
    print(f"Model: {args.model_name}")
    print(f"Environment: {args.env_type}")
    print(f"Protocol: {args.protocol}")
    print(f"Projection: {args.projection}, epsilon={args.epsilon}")
    print("="*50)

    if args.backend == "hf":
        tracker.log_event("error", "HF backend not implemented yet in this repo skeleton")
        raise NotImplementedError(
            "HF backend is not implemented yet. Start with `--backend toy` to validate the pipeline, "
            "then we can plug in transformers + torch for real LLM RLVR training."
        )

    # Create env
    env_config = EnvConfig(name=args.env_type, max_steps=20, seed=args.seed, extra={"show_tests": args.show_tests})
    if args.env_type == "code":
        env = CodeEnv(env_config)
    else:
        env = SQLEnv(env_config)

    toy_tasks = get_toy_tasks(args.env_type, variant="paper_b")
    if not toy_tasks:
        raise RuntimeError("No toy tasks available")

    # Build skill -> tasks map from dataset metadata
    tasks_by_skill: Dict[str, List] = {}
    for t in toy_tasks:
        meta = t.task.get("metadata", {}) if isinstance(t.task, dict) else {}
        skill = str(meta.get(args.skill_key, "unknown")) if isinstance(meta, dict) else "unknown"
        tasks_by_skill.setdefault(skill, []).append(t)

    skill_ids = sorted(tasks_by_skill.keys())
    if not skill_ids:
        raise RuntimeError("No skills found in toy tasks")

    if args.protocol == "sequential":
        sequence = args.skill_sequence[:] if args.skill_sequence else skill_ids
        # validate sequence
        missing = [s for s in sequence if s not in tasks_by_skill]
        if missing:
            raise ValueError(f"Unknown skills in --skill_sequence: {missing}. Available: {skill_ids}")
    else:
        sequence = skill_ids

    num_actions = len(toy_tasks[0].candidates)
    if any(len(t.candidates) != num_actions for t in toy_tasks):
        raise ValueError("All toy tasks must share the same number of candidates")

    policy = ToyCategoricalPolicy(num_actions, seed=args.seed)
    tracker.log_event("toy", "Toy backend enabled", {"num_actions": num_actions, "skills": skill_ids, "num_tasks": len(toy_tasks)})

    # Eval tracking for forgetting
    best_eval: Dict[str, float] = {}
    current_eval: Dict[str, float] = {}
    for sid in skill_ids:
        val = _eval_greedy_pass_at_1(env, args.env_type, tasks_by_skill[sid], policy, temperature=1.0)
        best_eval[sid] = float(val)
        current_eval[sid] = float(val)
    tracker.log_event("eval", "initial skill eval", {"per_skill": current_eval})

    for step in range(1, args.max_steps + 1):
        # Determine target/protected skills
        if args.protocol == "sequential":
            stage_idx = (step - 1) // max(1, args.stage_steps)
            stage_idx = min(stage_idx, len(sequence) - 1)
            target_skill = sequence[stage_idx]
            protected = sequence[:stage_idx]  # protect previous skills
        else:
            target_skill = "mixed"
            protected = [s for s in skill_ids]

        # Collect rollouts: target skill + (optional) protected memory
        trajectories: List[Trajectory] = []
        chosen_indices: List[int] = []

        # Target skill prompts
        target_pool = toy_tasks if target_skill == "mixed" else tasks_by_skill[target_skill]
        for _ in range(args.batch_size):
            task = policy.rng.choice(target_pool)
            for _ in range(args.num_rollouts_per_prompt):
                traj, a_idx = _rollout_toy(env, args.env_type, task, policy, temperature=args.temperature)
                trajectories.append(traj)
                chosen_indices.append(a_idx)

        # Protected memory prompts (sequential only)
        if args.protocol == "sequential" and protected and args.memory_per_protected > 0:
            for sid in protected:
                pool = tasks_by_skill.get(sid, [])
                if not pool:
                    continue
                for _ in range(args.memory_per_protected):
                    task = policy.rng.choice(pool)
                    for _ in range(args.num_rollouts_per_prompt):
                        traj, a_idx = _rollout_toy(env, args.env_type, task, policy, temperature=args.temperature)
                        trajectories.append(traj)
                        chosen_indices.append(a_idx)

        advantages = _compute_grpo_advantages(trajectories)

        probs = policy.probs(temperature=args.temperature)

        # Per-skill objective gradients in logit space:
        #   g_i = adv_i * (one_hot(a_i) - probs)
        grad_by_skill: Dict[str, List[float]] = {}
        count_by_skill: Dict[str, int] = {}
        for traj, adv in zip(trajectories, advantages):
            a = traj.metadata.get("toy_action_index")
            if not isinstance(a, int):
                continue
            sid = _skill_id(traj, skill_key=args.skill_key)
            g = [0.0 for _ in range(num_actions)]
            for j in range(num_actions):
                g[j] = float(adv) * ((1.0 if j == a else 0.0) - float(probs[j]))
            if sid not in grad_by_skill:
                grad_by_skill[sid] = [0.0 for _ in range(num_actions)]
                count_by_skill[sid] = 0
            for j in range(num_actions):
                grad_by_skill[sid][j] += g[j]
            count_by_skill[sid] += 1

        # Convert sums to means (stabilize scale across skills)
        for sid, g in list(grad_by_skill.items()):
            c = max(1, count_by_skill.get(sid, 1))
            grad_by_skill[sid] = [x / c for x in g]

        # Candidate update direction
        if args.protocol == "sequential":
            candidate = grad_by_skill.get(target_skill, [0.0 for _ in range(num_actions)])[:]
            protected_grads = [grad_by_skill[s] for s in protected if s in grad_by_skill]
        else:
            # Mixed protocol: start from mean gradient, constrain all skills
            candidate = [0.0 for _ in range(num_actions)]
            if grad_by_skill:
                for g in grad_by_skill.values():
                    for j in range(num_actions):
                        candidate[j] += g[j] / len(grad_by_skill)
            protected_grads = [grad_by_skill[s] for s in skill_ids if s in grad_by_skill]

        # Projection / skill-safe adjustment
        if args.projection == "none":
            final_grad = candidate
        else:
            eps = 0.0 if args.projection == "pcgrad" else float(args.epsilon)
            final_grad = _sequential_margin_project(
                candidate,
                protected_grads,
                epsilon=eps,
                normalize_to=(_norm(candidate) if args.normalize_update else None),
            )

        policy.apply_grad(final_grad, lr=args.toy_lr, ascent=True)

        if step % args.log_interval == 0:
            # Batch success per skill
            succ_by_skill: Dict[str, List[float]] = {}
            for t in trajectories:
                sid = _skill_id(t, skill_key=args.skill_key)
                succ_by_skill.setdefault(sid, []).append(float(t.r_final))
            batch_skill_success = {sid: (sum(v) / len(v)) for sid, v in succ_by_skill.items() if v}
            avg_batch_success = float(sum(t.r_final for t in trajectories) / max(1, len(trajectories)))
            worst_batch_success = float(min(batch_skill_success.values())) if batch_skill_success else 0.0

            # Interference matrix from per-skill gradients (cosine similarity)
            gid_map = {sid: i for i, sid in enumerate(sorted(grad_by_skill.keys()))}
            group_grads = {gid_map[sid]: g for sid, g in grad_by_skill.items()}
            conflict = compute_conflict_metrics(group_grads, conflict_threshold=args.conflict_threshold)

            tracker.log_metrics(RunMetrics(
                step=step,
                train_loss=0.0,
                success_rate=avg_batch_success,
                pass_at_1=avg_batch_success,  # proxy (toy backend)
                pass_at_k={},
                avg_trajectory_length=float(sum(t.length for t in trajectories) / max(1, len(trajectories))),
                wall_time=float(time.time() - exp_wall_start),
                conflict_ratio=float(conflict.conflict_ratio),
                solution_diversity=0.0,
                avg_credit_spread=0.0,
                extra={
                    "protocol": args.protocol,
                    "target_skill": target_skill,
                    "protected_skills": protected,
                    "epsilon": float(args.epsilon),
                    "batch_skill_success": batch_skill_success,
                    "avg_batch_success": avg_batch_success,
                    "worst_batch_success": worst_batch_success,
                    "candidate_grad_norm": float(_norm(candidate)),
                    "final_grad_norm": float(_norm(final_grad)),
                    "interference_avg_angle_deg": float(conflict.avg_angle_deg),
                    "interference_group_norms": conflict.group_norms,
                },
            ))
            tracker.log_event("interference", "logged interference stats", {
                "skill_ids": sorted(grad_by_skill.keys()),
                "matrix": conflict.conflict_matrix,
                "avg_angle_deg": conflict.avg_angle_deg,
                "group_norms": conflict.group_norms,
            })

        if step % args.eval_interval == 0:
            # Greedy pass@1 per skill (proxy for retention)
            current_eval = {
                sid: float(_eval_greedy_pass_at_1(env, args.env_type, tasks_by_skill[sid], policy, temperature=1.0))
                for sid in skill_ids
            }
            for sid, val in current_eval.items():
                if val > best_eval.get(sid, -1e9):
                    best_eval[sid] = float(val)
            forgetting = {sid: float(best_eval[sid] - current_eval[sid]) for sid in skill_ids}
            tracker.log_event("eval", "skill eval checkpoint", {
                "step": step,
                "per_skill": current_eval,
                "best_per_skill": best_eval,
                "forgetting": forgetting,
                "avg": float(sum(current_eval.values()) / len(current_eval)) if current_eval else 0.0,
                "worst": float(min(current_eval.values())) if current_eval else 0.0,
                "max_forgetting": float(max(forgetting.values())) if forgetting else 0.0,
            })

    tracker.finalize()


if __name__ == "__main__":
    main()

"""
Toy backend runner for Paper A.
"""

from __future__ import annotations

import time
from typing import Dict, List, Tuple

from shared.analysis import agop_stats_from_grad_vectors
from shared.envs import CodeEnv, EnvConfig, SQLEnv
from shared.envs.base import Action, ActionType, Trajectory
from shared.experiment_tracking import ExperimentConfig, ExperimentTracker, RunMetrics
from shared.toy import ToyCategoricalPolicy, get_toy_tasks

from paper_a_credit_assignment import (
    AdvantageMapper,
    CounterfactualExecutor,
    CounterfactualGenerator,
    CreditEstimator,
)
from paper_a_credit_assignment.train_utils import (
    compute_grpo_advantages,
    pass_at_k_from_groups,
    cf_result_reward,
    to_float_list,
    trajectory_reward,
)


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
        env.step(
            Action(
                ActionType.TOOL_CALL,
                candidate,
                tool_name="submit_query",
                metadata={"logprob": sample.logprob},
            )
        )

    traj = env.get_trajectory()
    traj.metadata["toy_action_index"] = sample.action_index
    return traj, sample.action_index


def _toy_grad_vectors(
    action_indices: List[int],
    weights: List[float],
    probs: List[float],
) -> List[List[float]]:
    """
    Per-sample gradient vectors in logit space for REINFORCE:
        g_i = w_i * (one_hot(a_i) - probs)
    """
    if len(action_indices) != len(weights):
        raise ValueError("action_indices and weights must have same length")
    k = len(probs)
    out: List[List[float]] = []
    for a, w in zip(action_indices, weights):
        g = [0.0 for _ in range(k)]
        for j in range(k):
            g[j] = float(w) * ((1.0 if j == a else 0.0) - float(probs[j]))
        out.append(g)
    return out


def run_toy(args, config: ExperimentConfig) -> None:
    tracker = ExperimentTracker(config, base_dir=args.output_dir)
    tracker.log_event("init", "Experiment initialized")
    exp_wall_start = time.time()

    env_config = EnvConfig(
        name=args.env_type,
        max_steps=args.max_trajectory_length,
        seed=args.seed,
        extra={"show_tests": args.show_tests},
    )
    env = CodeEnv(env_config) if args.env_type == "code" else SQLEnv(env_config)

    cf_generator = CounterfactualGenerator(
        intervention_types=args.intervention_types,
        k=args.counterfactual_k,
        prioritize_high_value=args.prioritize_high_value_cf,
        seed=args.seed,
    )
    cf_executor = CounterfactualExecutor(env, use_cache=True, seed=args.seed)
    credit_estimator = CreditEstimator(normalization=args.credit_normalization)
    advantage_mapper = AdvantageMapper(level="step")

    tracker.log_event(
        "components",
        "Paper A components initialized",
        {
            "intervention_types": args.intervention_types,
            "k": args.counterfactual_k,
            "prioritize_high_value_cf": bool(args.prioritize_high_value_cf),
            "reward_mode": args.reward_mode,
            "flat_group_fallback": args.flat_group_fallback,
            "failure_reward_floor": float(args.failure_reward_floor),
        },
    )

    print(f"Experiment directory: {tracker.experiment_dir}")
    print(f"Config saved to: {tracker.experiment_dir / 'config.json'}")
    print("\n" + "=" * 50)
    print("Paper A: Counterfactual Credit Assignment")
    print("=" * 50)
    print(f"Model: {args.model_name}")
    print(f"Environment: {args.env_type}")
    print(f"Counterfactual K: {args.counterfactual_k}")
    print(f"Interventions: {args.intervention_types}")
    print("=" * 50)

    toy_tasks = get_toy_tasks(args.env_type, variant="paper_a")
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

        reward_values = [
            trajectory_reward(
                t,
                mode=args.reward_mode,
                blend_alpha=args.reward_blend_alpha,
                failure_reward_floor=args.failure_reward_floor,
            )
            for t in trajectories
        ]
        advantages = compute_grpo_advantages(
            trajectories,
            reward_values,
            flat_group_fallback=args.flat_group_fallback,
        )

        weights: List[float] = []
        credit_spreads: List[float] = []
        for traj, adv in zip(trajectories, advantages):
            w = float(adv)
            if args.use_counterfactual_credit and cf_generator.should_generate_counterfactuals(traj):
                interventions = cf_generator.generate(traj)
                cf_results = cf_executor.batch_execute(traj, interventions)
                credit_map = credit_estimator.estimate(
                    traj,
                    cf_results,
                    base_reward=trajectory_reward(
                        traj,
                        mode=args.reward_mode,
                        blend_alpha=args.reward_blend_alpha,
                        failure_reward_floor=args.failure_reward_floor,
                    ),
                    cf_reward_fn=lambda cf: cf_result_reward(
                        cf,
                        mode=args.reward_mode,
                        blend_alpha=args.reward_blend_alpha,
                        failure_reward_floor=args.failure_reward_floor,
                    ),
                )
                credit_spreads.append(credit_map.spread)

                step_adv = to_float_list(advantage_mapper.map_to_step_advantages(traj, credit_map))
                lp_steps = [i for i, s in enumerate(traj.steps) if s.logprob is not None]
                if lp_steps:
                    selected = [float(step_adv[i]) for i in lp_steps]
                    if (
                        args.fallback_to_adv_when_zero_credit
                        and max(abs(v) for v in selected) <= float(args.zero_credit_threshold)
                    ):
                        credit_weight = 1.0
                    else:
                        credit_weight = sum(selected) / len(selected)
                    if args.credit_fallback_when_zero_adv and abs(w) <= float(args.zero_adv_threshold):
                        w = float(args.credit_fallback_scale) * float(credit_weight)
                    else:
                        w *= float(credit_weight)
            weights.append(w)

        agop_extra: Dict[str, float] = {}
        if args.log_agop and (step % args.log_interval == 0):
            probs = policy.probs(temperature=args.temperature)
            agop_base = agop_stats_from_grad_vectors(
                _toy_grad_vectors(chosen_indices, advantages, probs),
                power_iters=args.agop_power_iters,
                power_seed=args.seed,
            )
            agop_used = agop_stats_from_grad_vectors(
                _toy_grad_vectors(chosen_indices, weights, probs),
                power_iters=args.agop_power_iters,
                power_seed=args.seed + 1,
            )
            agop_extra = {
                "agop_trace": float(agop_used["trace"]),
                "agop_effective_rank": float(agop_used["effective_rank"]),
                "agop_top_trace_ratio": float(agop_used["top_trace_ratio"]),
                "agop_baseline_effective_rank": float(agop_base["effective_rank"]),
                "agop_baseline_top_trace_ratio": float(agop_base["top_trace_ratio"]),
            }

        update_info = policy.reinforce_update(
            chosen_indices,
            weights,
            lr=args.toy_lr,
            temperature=args.temperature,
        )

        if step % args.log_interval == 0:
            success_rate = sum(t.r_final for t in trajectories) / max(1, len(trajectories))
            pass_at_1 = pass_at_k_from_groups(trajectories, k=1)
            pass_at_k = pass_at_k_from_groups(trajectories, k=args.num_rollouts_per_prompt)
            avg_len = sum(t.length for t in trajectories) / max(1, len(trajectories))
            avg_credit_spread = (sum(credit_spreads) / len(credit_spreads)) if credit_spreads else 0.0

            tracker.log_metrics(
                RunMetrics(
                    step=step,
                    train_loss=0.0,
                    success_rate=float(success_rate),
                    pass_at_1=float(pass_at_1),
                    pass_at_k={int(args.num_rollouts_per_prompt): float(pass_at_k)},
                    avg_trajectory_length=float(avg_len),
                    wall_time=float(time.time() - exp_wall_start),
                    avg_credit_spread=float(avg_credit_spread),
                    extra={
                        **agop_extra,
                        "reward_mode": args.reward_mode,
                        "failure_reward_floor": float(args.failure_reward_floor),
                        "mean_reward": float(sum(reward_values) / max(1, len(reward_values))),
                        "mean_weight": float(sum(weights) / max(1, len(weights))),
                        "mean_abs_weight": float(sum(abs(w) for w in weights) / max(1, len(weights))),
                        "nonzero_weight_ratio": float(
                            sum(1 for w in weights if abs(w) > float(args.zero_adv_threshold))
                            / max(1, len(weights))
                        ),
                    },
                )
            )
            tracker.log_event("train", "logged step metrics", update_info)

        if step % args.eval_interval == 0:
            tracker.log_event("eval", "eval checkpoint", {"step": step})

    tracker.finalize()

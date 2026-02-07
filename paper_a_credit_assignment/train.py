"""
Paper A training entrypoint.
Verifiable Agent RL with Counterfactual Credit Assignment.
"""

import sys
from pathlib import Path

# Keep script execution compatibility: `python paper_a_credit_assignment/train.py ...`
sys.path.append(str(Path(__file__).parent.parent))

from shared.experiment_tracking import ExperimentConfig

from paper_a_credit_assignment.hf_runner import run_hf
from paper_a_credit_assignment.toy_runner import run_toy
from paper_a_credit_assignment.train_args import parse_args


def _build_experiment_config(args) -> ExperimentConfig:
    return ExperimentConfig(
        experiment_name=args.experiment_name,
        project="paper_a",
        description=args.description,
        model_name=str(args.model_path or args.model_name),
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
        seed=args.seed,
        extra={
            "backend": args.backend,
            "reward_mode": args.reward_mode,
            "flat_group_fallback": args.flat_group_fallback,
            "failure_reward_floor": float(args.failure_reward_floor),
            "failure_replay_ratio": float(args.failure_replay_ratio),
            "failure_buffer_unique": bool(args.failure_buffer_unique),
            "max_policy_turns": int(args.max_policy_turns),
            "task_timeout_seconds": float(args.task_timeout_seconds),
            "heartbeat_interval": int(args.heartbeat_interval),
            "sync_eval_and_save": bool(args.sync_eval_and_save),
            "truncate_to_global_min_samples": bool(args.truncate_to_global_min_samples),
            "fallback_to_adv_when_zero_credit": bool(args.fallback_to_adv_when_zero_credit),
            "zero_credit_threshold": float(args.zero_credit_threshold),
        },
    )


def main() -> None:
    args = parse_args()
    config = _build_experiment_config(args)

    if args.backend == "hf":
        run_hf(args, config)
    else:
        run_toy(args, config)


if __name__ == "__main__":
    main()

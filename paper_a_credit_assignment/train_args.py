"""
Paper A training argument parser.
"""

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Paper A: Counterfactual Credit Assignment")

    # NOTE: We avoid argparse.BooleanOptionalAction for Python 3.8 compatibility on some clusters.
    def add_bool_flag(name: str, *, default: bool, help: str):
        dest = name.lstrip("-")
        parser.add_argument(name, dest=dest, action="store_true", help=help)
        parser.add_argument(f"--no-{dest}", dest=dest, action="store_false", help=argparse.SUPPRESS)
        parser.set_defaults(**{dest: default})

    # Base config
    parser.add_argument("--experiment_name", type=str, default="credit_assignment")
    parser.add_argument("--description", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--backend", type=str, default="toy", choices=["toy", "hf"])

    # Model config
    parser.add_argument("--model_name", type=str, default="Qwen2.5-7B")
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="(hf) local path or HF name; overrides --model_name",
    )
    add_bool_flag("--use_lora", default=True, help="(hf) enable LoRA adapters")
    parser.add_argument("--lora_rank", type=int, default=64)

    # Training config
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--toy_lr", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--num_rollouts_per_prompt", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0, help="(hf) nucleus sampling p")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="(hf) max tokens to generate")
    parser.add_argument("--max_prompt_tokens", type=int, default=1024, help="(hf) truncate prompt to this length")
    parser.add_argument(
        "--max_policy_turns",
        type=int,
        default=1,
        help="(hf, code env) max write-test turns per rollout trajectory",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="(hf) model dtype on GPU",
    )
    parser.add_argument("--grad_clip", type=float, default=1.0, help="(hf) gradient clipping")
    add_bool_flag("--trust_remote_code", default=False, help="(hf) set True for some community models")
    add_bool_flag(
        "--use_chat_template",
        default=True,
        help="(hf) use tokenizer.apply_chat_template if available",
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=None,
        help="(hf) system prompt override (default is a code-only prompt)",
    )
    parser.add_argument("--save_interval", type=int, default=500, help="(hf) save checkpoint every N steps")
    parser.add_argument("--eval_tasks", type=int, default=64, help="(hf) number of eval tasks per checkpoint")
    parser.add_argument(
        "--eval_sample_k",
        type=int,
        default=4,
        help="(hf) sampled decoding count per eval task for pass@k-style eval (<=1 disables sampled eval)",
    )
    add_bool_flag(
        "--require_cuda",
        default=True,
        help="(hf) exit if CUDA is not available (prevents accidentally running a 7B model on CPU)",
    )
    parser.add_argument(
        "--rl_microbatch_size",
        type=int,
        default=0,
        help="(hf) microbatch size for policy-gradient loss forward. 0=auto (recommended for multi-GPU).",
    )
    add_bool_flag(
        "--gradient_checkpointing",
        default=True,
        help="(hf) enable gradient checkpointing to reduce activation memory (slower but prevents OOM).",
    )
    parser.add_argument(
        "--heartbeat_interval",
        type=int,
        default=1,
        help="(hf) rank0 heartbeat interval in steps; <=0 disables.",
    )
    add_bool_flag(
        "--sync_eval_and_save",
        default=True,
        help="(hf, ddp) add barriers before/after rank0 eval/checkpoint to avoid rank desync hangs.",
    )
    add_bool_flag(
        "--truncate_to_global_min_samples",
        default=True,
        help="(hf, ddp) truncate per-rank sampled trajectories to global min sample count each step.",
    )

    # Environment config
    parser.add_argument("--env_type", type=str, default="code", choices=["code", "sql"])
    parser.add_argument("--max_trajectory_length", type=int, default=20)
    parser.add_argument(
        "--task_timeout_seconds",
        type=float,
        default=8.0,
        help="(code env) per-run test execution timeout to avoid DDP straggler stalls",
    )
    add_bool_flag("--show_tests", default=True, help="(code env) whether to show unit tests in the observation")
    parser.add_argument(
        "--train_dataset",
        type=str,
        default=None,
        help="(hf) JSONL dataset path for training (CodeEnv schema)",
    )
    parser.add_argument(
        "--eval_dataset",
        type=str,
        default=None,
        help="(hf) optional JSONL dataset path for evaluation",
    )
    parser.add_argument("--max_train_samples", type=int, default=None, help="(hf) cap train dataset size")
    parser.add_argument("--max_eval_samples", type=int, default=None, help="(hf) cap eval dataset size")

    # Paper A specific
    add_bool_flag("--use_counterfactual_credit", default=True, help="enable counterfactual credit reweighting")
    add_bool_flag(
        "--prioritize_high_value_cf",
        default=True,
        help="only generate counterfactuals for high-value trajectories (disable for early-stage sparse rewards)",
    )
    parser.add_argument("--counterfactual_k", type=int, default=4)
    parser.add_argument("--intervention_types", nargs="+", default=["delete", "truncate"])
    parser.add_argument(
        "--credit_normalization",
        type=str,
        default="signed",
        choices=["signed", "minmax", "zscore", "softmax", "none"],
    )
    parser.add_argument(
        "--reward_mode",
        type=str,
        default="binary",
        choices=["binary", "score", "mixed"],
        help="reward used for GRPO/credit: binary(r_final), score(verifier score), or mixed",
    )
    parser.add_argument(
        "--reward_blend_alpha",
        type=float,
        default=0.5,
        help="for reward_mode=mixed: reward=(1-alpha)*binary + alpha*score",
    )
    parser.add_argument(
        "--failure_reward_floor",
        type=float,
        default=0.0,
        help="if trajectory fails and reward<=0, use this floor reward to avoid all-zero updates",
    )
    parser.add_argument(
        "--flat_group_fallback",
        type=str,
        default="batch_centered",
        choices=["zero", "batch_centered", "raw"],
        help="fallback advantage when group reward std is zero",
    )
    add_bool_flag(
        "--credit_fallback_when_zero_adv",
        default=True,
        help="when group advantage is ~0, directly use credit-weight as update weight",
    )
    parser.add_argument("--zero_adv_threshold", type=float, default=1e-8)
    parser.add_argument("--credit_fallback_scale", type=float, default=1.0)
    add_bool_flag(
        "--fallback_to_adv_when_zero_credit",
        default=True,
        help="(hf/toy) if per-step credit map is near-zero, fallback to trajectory advantage instead of zero update",
    )
    parser.add_argument(
        "--zero_credit_threshold",
        type=float,
        default=1e-8,
        help="threshold for treating mapped step-credit as all-zero",
    )
    parser.add_argument(
        "--failure_replay_ratio",
        type=float,
        default=0.0,
        help="fraction of prompts sampled from recent failure buffer (0 disables replay)",
    )
    parser.add_argument("--failure_buffer_size", type=int, default=4096)
    parser.add_argument("--failure_replay_warmup_steps", type=int, default=0)
    parser.add_argument(
        "--replay_min_success_ema",
        type=float,
        default=0.2,
        help="(hf) enable failure replay only when smoothed success_rate >= this threshold",
    )
    parser.add_argument(
        "--replay_ema_alpha",
        type=float,
        default=0.1,
        help="(hf) EMA alpha for replay gating success rate",
    )
    add_bool_flag(
        "--failure_buffer_unique",
        default=True,
        help="(hf) keep failure replay buffer task ids unique to avoid over-concentrating a few hard examples",
    )
    add_bool_flag(
        "--guard_all_negative_batch",
        default=True,
        help="(hf) if a batch is pure-failure with all non-positive weights, zero-out weights to avoid collapse",
    )
    parser.add_argument(
        "--all_negative_reward_span_threshold",
        type=float,
        default=1e-8,
        help="(hf) reward-span threshold for pure-failure negative-batch guard",
    )
    add_bool_flag(
        "--log_agop",
        default=True,
        help="Log AGOP (Average Gradient Outer Product) stats on toy policy gradients.",
    )
    parser.add_argument(
        "--agop_power_iters",
        type=int,
        default=30,
        help="Power-iteration steps for estimating top AGOP eigenvalue.",
    )

    # Output
    parser.add_argument("--output_dir", type=str, default="./experiments")

    return parser.parse_args()

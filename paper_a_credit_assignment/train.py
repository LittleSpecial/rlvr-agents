"""
Paper A 训练入口
Verifiable Agent RL with Counterfactual Credit Assignment
"""

import sys
import argparse
import statistics
import time
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from shared.experiment_tracking import ExperimentTracker, ExperimentConfig, RunMetrics
from shared.envs import CodeEnv, EnvConfig, SQLEnv
from shared.envs.base import Action, ActionType, Trajectory
from shared.analysis import agop_stats_from_grad_vectors
from shared.toy import ToyCategoricalPolicy, get_toy_tasks

from paper_a_credit_assignment import (
    CounterfactualGenerator,
    CounterfactualExecutor,
    CreditEstimator,
    AdvantageMapper
)


def parse_args():
    parser = argparse.ArgumentParser(description="Paper A: Counterfactual Credit Assignment")

    # NOTE: We avoid argparse.BooleanOptionalAction for Python 3.8 compatibility on some clusters.
    def add_bool_flag(name: str, *, default: bool, help: str):
        dest = name.lstrip("-")
        parser.add_argument(name, dest=dest, action="store_true", help=help)
        parser.add_argument(f"--no-{dest}", dest=dest, action="store_false", help=argparse.SUPPRESS)
        parser.set_defaults(**{dest: default})
    
    # 基本配置
    parser.add_argument("--experiment_name", type=str, default="credit_assignment")
    parser.add_argument("--description", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--backend", type=str, default="toy", choices=["toy", "hf"])
    
    # 模型配置
    parser.add_argument("--model_name", type=str, default="Qwen2.5-7B")
    parser.add_argument("--model_path", type=str, default=None,
                        help="(hf) local path or HF name; overrides --model_name")
    add_bool_flag("--use_lora", default=True, help="(hf) enable LoRA adapters")
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
    parser.add_argument("--top_p", type=float, default=1.0, help="(hf) nucleus sampling p")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="(hf) max tokens to generate")
    parser.add_argument("--max_prompt_tokens", type=int, default=1024, help="(hf) truncate prompt to this length")
    parser.add_argument(
        "--max_policy_turns",
        type=int,
        default=1,
        help="(hf, code env) max write-test turns per rollout trajectory",
    )
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"],
                        help="(hf) model dtype on GPU")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="(hf) gradient clipping")
    add_bool_flag("--trust_remote_code", default=False, help="(hf) set True for some community models")
    add_bool_flag("--use_chat_template", default=True, help="(hf) use tokenizer.apply_chat_template if available")
    parser.add_argument("--system_prompt", type=str, default=None,
                        help="(hf) system prompt override (default is a code-only prompt)")
    parser.add_argument("--save_interval", type=int, default=500, help="(hf) save checkpoint every N steps")
    parser.add_argument("--eval_tasks", type=int, default=64, help="(hf) number of eval tasks per checkpoint")
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
    
    # 环境配置
    parser.add_argument("--env_type", type=str, default="code", choices=["code", "sql"])
    parser.add_argument("--max_trajectory_length", type=int, default=20)
    parser.add_argument(
        "--task_timeout_seconds",
        type=float,
        default=8.0,
        help="(code env) per-run test execution timeout to avoid DDP straggler stalls",
    )
    add_bool_flag("--show_tests", default=True, help="(code env) whether to show unit tests in the observation")
    parser.add_argument("--train_dataset", type=str, default=None,
                        help="(hf) JSONL dataset path for training (CodeEnv schema)")
    parser.add_argument("--eval_dataset", type=str, default=None,
                        help="(hf) optional JSONL dataset path for evaluation")
    parser.add_argument("--max_train_samples", type=int, default=None, help="(hf) cap train dataset size")
    parser.add_argument("--max_eval_samples", type=int, default=None, help="(hf) cap eval dataset size")

    # Paper A 特定配置
    add_bool_flag("--use_counterfactual_credit", default=True, help="enable counterfactual credit reweighting")
    add_bool_flag(
        "--prioritize_high_value_cf",
        default=True,
        help="only generate counterfactuals for high-value trajectories (disable for early-stage sparse rewards)",
    )
    parser.add_argument("--counterfactual_k", type=int, default=4)
    parser.add_argument("--intervention_types", nargs="+", default=["delete", "truncate"])
    parser.add_argument("--credit_normalization", type=str, default="signed",
                        choices=["signed", "minmax", "zscore", "softmax", "none"])
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
    parser.add_argument(
        "--failure_replay_ratio",
        type=float,
        default=0.0,
        help="fraction of prompts sampled from recent failure buffer (0 disables replay)",
    )
    parser.add_argument("--failure_buffer_size", type=int, default=4096)
    parser.add_argument("--failure_replay_warmup_steps", type=int, default=0)
    add_bool_flag("--log_agop", default=True, help="Log AGOP (Average Gradient Outer Product) stats on toy policy gradients.")
    parser.add_argument("--agop_power_iters", type=int, default=30,
                        help="Power-iteration steps for estimating top AGOP eigenvalue.")
    
    # 输出配置
    parser.add_argument("--output_dir", type=str, default="./experiments")
    
    return parser.parse_args()


def _clone_task_with_group_id(task: Dict, *, group_id: str) -> Dict:
    """
    Make a shallow-cloned task dict with a unique task_id for GRPO grouping.
    Keep the original dataset task id in metadata.base_task_id.
    """
    cloned = dict(task)
    base_task_id = str(cloned.get("task_id", "unknown"))
    cloned["task_id"] = group_id
    meta = dict(cloned.get("metadata") or {})
    meta.setdefault("base_task_id", base_task_id)
    meta.setdefault("group_id", group_id)
    cloned["metadata"] = meta
    return cloned


def _clamp01(x: float) -> float:
    try:
        v = float(x)
    except Exception:
        return 0.0
    if not (v == v):  # NaN
        return 0.0
    return max(0.0, min(1.0, v))


def _trajectory_reward(
    trajectory: Trajectory,
    *,
    mode: str,
    blend_alpha: float,
    failure_reward_floor: float = 0.0,
) -> float:
    binary = _clamp01(float(trajectory.r_final))
    score = _clamp01(float(getattr(trajectory.verifier_info, "score", binary)))
    if mode == "binary":
        reward = binary
    elif mode == "score":
        reward = score
    else:
        alpha = max(0.0, min(1.0, float(blend_alpha)))
        reward = (1.0 - alpha) * binary + alpha * score
    if binary < 0.5 and reward <= 0.0 and failure_reward_floor != 0.0:
        reward = float(failure_reward_floor)
    return float(reward)


def _cf_result_reward(
    cf_result,
    *,
    mode: str,
    blend_alpha: float,
    failure_reward_floor: float = 0.0,
) -> float:
    binary = _clamp01(float(getattr(cf_result, "r_final_cf", 0.0)))
    score = binary
    cf_traj = getattr(cf_result, "cf_trajectory", None)
    if cf_traj is not None and getattr(cf_traj, "verifier_info", None) is not None:
        score = _clamp01(float(getattr(cf_traj.verifier_info, "score", binary)))
    if mode == "binary":
        reward = binary
    elif mode == "score":
        reward = score
    else:
        alpha = max(0.0, min(1.0, float(blend_alpha)))
        reward = (1.0 - alpha) * binary + alpha * score
    if binary < 0.5 and reward <= 0.0 and failure_reward_floor != 0.0:
        reward = float(failure_reward_floor)
    return float(reward)


def _base_task_id_from_trajectory(trajectory: Trajectory) -> Optional[str]:
    task_meta = ((trajectory.metadata or {}).get("task") or {}).get("metadata") or {}
    base_id = task_meta.get("base_task_id")
    if base_id is None:
        return None
    return str(base_id)


def _compute_grpo_advantages(
    trajectories: List[Trajectory],
    rewards: List[float],
    *,
    flat_group_fallback: str = "batch_centered",
) -> List[float]:
    if len(trajectories) != len(rewards):
        raise ValueError("trajectories/rewards length mismatch")
    if not trajectories:
        return []

    rewards = [float(r) for r in rewards]
    batch_mean = statistics.fmean(rewards)
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
                    if flat_group_fallback == "zero":
                        advantages[i] = 0.0
                    elif flat_group_fallback == "batch_centered":
                        advantages[i] = rewards[i] - batch_mean
                    elif flat_group_fallback == "raw":
                        advantages[i] = rewards[i]
                    else:
                        raise ValueError(f"Unknown flat_group_fallback: {flat_group_fallback}")
        else:
            i = indices[0]
            if flat_group_fallback == "zero":
                advantages[i] = 0.0
            elif flat_group_fallback == "batch_centered":
                advantages[i] = rewards[i] - batch_mean
            elif flat_group_fallback == "raw":
                advantages[i] = rewards[i]
            else:
                raise ValueError(f"Unknown flat_group_fallback: {flat_group_fallback}")
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


@dataclass
class _RolloutState:
    task: Dict
    observation: str
    done: bool = False
    trajectory: Optional[Trajectory] = None
    action_count: int = 0
    latest_feedback: str = ""


def _build_code_followup_observation(
    task: Dict,
    *,
    current_code: str,
    last_feedback: str,
    show_tests: bool,
) -> str:
    """
    Build a richer multi-turn observation so the model can iteratively refine code.
    """
    language = str(task.get("language", "python"))
    parts: List[str] = [f"Task: {task.get('prompt', '')}"]
    if current_code:
        parts.append(f"Current Code:\n```{language}\n{current_code}\n```")
    if last_feedback:
        parts.append(f"Last Test Feedback:\n{last_feedback}")
    test_code = task.get("test_code")
    if show_tests and isinstance(test_code, str) and test_code:
        parts.append(f"Test Cases:\n```{language}\n{test_code}\n```")
    return "\n\n".join(parts).strip()


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


def _guess_lora_target_modules(model) -> List[str]:
    """
    Best-effort LoRA target module guessing for common decoder-only LMs.
    """
    candidates = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    names = {name.split(".")[-1] for name, _ in model.named_modules()}
    picked = [c for c in candidates if c in names]
    return picked if picked else ["q_proj", "k_proj", "v_proj", "o_proj"]


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
            _trajectory_reward(
                t,
                mode=args.reward_mode,
                blend_alpha=args.reward_blend_alpha,
                failure_reward_floor=args.failure_reward_floor,
            )
            for t in trajectories
        ]
        advantages = _compute_grpo_advantages(
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
                    base_reward=_trajectory_reward(
                        traj,
                        mode=args.reward_mode,
                        blend_alpha=args.reward_blend_alpha,
                        failure_reward_floor=args.failure_reward_floor,
                    ),
                    cf_reward_fn=lambda cf: _cf_result_reward(
                        cf,
                        mode=args.reward_mode,
                        blend_alpha=args.reward_blend_alpha,
                        failure_reward_floor=args.failure_reward_floor,
                    ),
                )
                credit_spreads.append(credit_map.spread)

                step_adv = _to_list(advantage_mapper.map_to_step_advantages(traj, credit_map))
                lp_steps = [i for i, s in enumerate(traj.steps) if s.logprob is not None]
                if lp_steps:
                    credit_weight = sum(step_adv[i] for i in lp_steps) / len(lp_steps)
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
            chosen_indices, weights, lr=args.toy_lr, temperature=args.temperature
        )

        if step % args.log_interval == 0:
            success_rate = sum(t.r_final for t in trajectories) / max(1, len(trajectories))
            pass_at_1 = _pass_at_k_from_groups(trajectories, k=1)
            pass_at_k = _pass_at_k_from_groups(trajectories, k=args.num_rollouts_per_prompt)
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


def run_hf(args, config: ExperimentConfig) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        from peft import LoraConfig, get_peft_model  # type: ignore
    except ModuleNotFoundError as e:
        raise SystemExit(
            "Missing dependency `peft` for LoRA. Install requirements on the login node.\n"
            f"Original error: {e}"
        )

    from shared.hf import (
        DistInfo,
        init_distributed,
        broadcast_object,
        all_reduce_mean,
        load_jsonl,
        build_code_prompt,
        build_sql_prompt,
        extract_python_code,
        weighted_rl_loss,
    )
    from shared.hf.prompts import DEFAULT_SYSTEM_PROMPT, DEFAULT_SQL_SYSTEM_PROMPT
    from shared.hf.distributed import barrier

    dist_info: DistInfo = init_distributed()
    has_cuda = torch.cuda.is_available()
    if args.require_cuda and not has_cuda:
        if dist_info.is_rank0:
            print(
                "\n[ERROR] CUDA is not available in this environment.\n"
                "This run is configured with --require_cuda to prevent accidental CPU training.\n"
                "On BSCC N32-H (aarch64), you typically need the cluster-provided CUDA PyTorch wheel + env.sh.\n",
                flush=True,
            )
            print(f"torch.__version__={torch.__version__} torch.version.cuda={torch.version.cuda}", flush=True)
        raise SystemExit(2)

    if has_cuda and dist_info.local_rank >= torch.cuda.device_count():
        raise SystemExit(
            f"[ERROR] local_rank={dist_info.local_rank} but visible CUDA devices={torch.cuda.device_count()}."
        )

    if has_cuda:
        torch.cuda.set_device(dist_info.local_rank)

    device = torch.device(f"cuda:{dist_info.local_rank}" if has_cuda else "cpu")

    if args.train_dataset is None:
        raise ValueError("--train_dataset is required for --backend hf")

    # Tracker: only rank0 writes logs/checkpoints.
    tracker = None
    exp_dir = ""
    if dist_info.is_rank0:
        tracker = ExperimentTracker(config, base_dir=args.output_dir)
        tracker.log_event("init", "Experiment initialized", {"world_size": dist_info.world_size})
        exp_dir = str(tracker.experiment_dir)
    exp_dir = str(broadcast_object(exp_dir, src=0, dist_info=dist_info))
    barrier(dist_info=dist_info)

    exp_wall_start = time.time()

    if dist_info.is_rank0:
        print(f"Experiment directory: {exp_dir}")
        print("\n" + "=" * 50)
        print("Paper A: Counterfactual Credit Assignment (HF backend)")
        print("=" * 50)
        print(f"Model: {args.model_path or args.model_name}")
        print(f"Environment: {args.env_type}")
        print(f"Counterfactual K: {args.counterfactual_k}")
        print(f"Interventions: {args.intervention_types}")
        print(f"World size: {dist_info.world_size}")
        print("=" * 50)

    # Env
    env_config = EnvConfig(
        name=args.env_type,
        max_steps=args.max_trajectory_length,
        seed=args.seed + dist_info.rank,
        extra={
            "show_tests": args.show_tests,
            "default_timeout": float(args.task_timeout_seconds),
            "cap_task_timeout": True,
        },
    )
    env = CodeEnv(env_config) if args.env_type == "code" else SQLEnv(env_config)

    # Paper A components
    cf_generator = CounterfactualGenerator(
        intervention_types=args.intervention_types,
        k=args.counterfactual_k,
        prioritize_high_value=args.prioritize_high_value_cf,
        seed=args.seed + dist_info.rank,
    )
    cf_executor = CounterfactualExecutor(env, use_cache=True, seed=args.seed + dist_info.rank)
    credit_estimator = CreditEstimator(normalization=args.credit_normalization)
    advantage_mapper = AdvantageMapper(level="step")

    if tracker:
        tracker.log_event("components", "Paper A components initialized", {
            "intervention_types": args.intervention_types,
            "k": args.counterfactual_k,
            "prioritize_high_value_cf": bool(args.prioritize_high_value_cf),
            "reward_mode": args.reward_mode,
            "flat_group_fallback": args.flat_group_fallback,
            "failure_reward_floor": float(args.failure_reward_floor),
            "failure_replay_ratio": float(args.failure_replay_ratio),
            "max_policy_turns": int(args.max_policy_turns),
            "task_timeout_seconds": float(args.task_timeout_seconds),
            "cap_task_timeout": True,
        })

    # Dataset
    train_path = Path(args.train_dataset)
    train_records = load_jsonl(train_path, max_samples=args.max_train_samples)
    if not train_records:
        raise RuntimeError(f"Empty train dataset: {train_path}")

    eval_records: List[Dict] = []
    if args.eval_dataset:
        eval_path = Path(args.eval_dataset)
        eval_records = load_jsonl(eval_path, max_samples=args.max_eval_samples)

    rng = random.Random(args.seed + 9973 * dist_info.rank)
    train_record_by_id: Dict[str, Dict] = {
        str(rec.get("task_id", f"idx_{i}")): rec for i, rec in enumerate(train_records)
    }
    failure_buffer: List[str] = []

    # Model / tokenizer
    model_name_or_path = args.model_path or args.model_name
    if dist_info.is_rank0:
        print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        padding_side="left",  # Required for decoder-only generation
    )
    added_pad = False
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        added_pad = True

    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.dtype]
    if dist_info.is_rank0:
        print(f"Loading model to {device} with dtype={dtype}...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=dtype,
    )
    if added_pad:
        model.resize_token_embeddings(len(tokenizer))

    if args.use_lora:
        targets = _guess_lora_target_modules(model)
        lora_cfg = LoraConfig(
            r=int(args.lora_rank),
            lora_alpha=int(args.lora_rank) * 2,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=targets,
        )
        model = get_peft_model(model, lora_cfg)
        if dist_info.is_rank0 and hasattr(model, "print_trainable_parameters"):
            model.print_trainable_parameters()

    if dist_info.is_rank0:
        print("Model loaded. Applying LoRA if enabled...", flush=True)
    if device.type == "cuda":
        print(f"[Rank {dist_info.rank}] Moving model to {device}...", flush=True)
        model = model.to(device)
        print(f"[Rank {dist_info.rank}] Model moved to {device}.", flush=True)

    # Memory helpers for long sequences + DDP.
    raw_model = model.module if hasattr(model, "module") else model
    if hasattr(raw_model, "config") and hasattr(raw_model.config, "use_cache"):
        raw_model.config.use_cache = False
    if args.gradient_checkpointing and hasattr(raw_model, "gradient_checkpointing_enable"):
        raw_model.gradient_checkpointing_enable()
        # With LoRA/frozen embeddings, checkpointed blocks may receive inputs without requires_grad.
        # HF/PEFT training typically calls enable_input_require_grads() to allow gradients to flow.
        if hasattr(model, "enable_input_require_grads"):
            try:
                model.enable_input_require_grads()
            except Exception:
                pass
        if hasattr(raw_model, "enable_input_require_grads"):
            try:
                raw_model.enable_input_require_grads()
            except Exception:
                pass

    if dist_info.distributed:
        print(f"[Rank {dist_info.rank}] Creating DDP wrapper...", flush=True)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[dist_info.local_rank] if device.type == "cuda" else None
        )
        print(f"[Rank {dist_info.rank}] DDP wrapper created.", flush=True)

    print(f"[Rank {dist_info.rank}] Creating optimizer...", flush=True)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters (did LoRA attach correctly?)")
    optimizer = torch.optim.AdamW(trainable_params, lr=float(args.learning_rate))
    print(f"[Rank {dist_info.rank}] Optimizer created.", flush=True)

    if args.system_prompt is not None:
        sys_prompt = args.system_prompt
    else:
        sys_prompt = DEFAULT_SYSTEM_PROMPT if args.env_type == "code" else DEFAULT_SQL_SYSTEM_PROMPT

    max_policy_turns = max(1, int(args.max_policy_turns))
    if args.env_type != "code":
        max_policy_turns = 1

    print(f"[Rank {dist_info.rank}] Entering training loop...", flush=True)
    for step in range(1, args.max_steps + 1):
        if dist_info.is_rank0 and step == 1:
            print(f"[Rank 0] Step 1 started.", flush=True)
        # 1) Sample prompts (per-rank batch)
        prompt_tasks: List[Dict] = []
        replay_quota = 0
        if (
            float(args.failure_replay_ratio) > 0.0
            and step > int(args.failure_replay_warmup_steps)
            and len(failure_buffer) > 0
        ):
            replay_quota = int(round(args.batch_size * float(args.failure_replay_ratio)))
            replay_quota = max(1, replay_quota)
            replay_quota = min(int(args.batch_size), replay_quota)

        for i in range(args.batch_size):
            if i < replay_quota and len(failure_buffer) > 0:
                base_id = rng.choice(failure_buffer)
                rec = train_record_by_id.get(base_id, rng.choice(train_records))
            else:
                rec = rng.choice(train_records)
            base_id = str(rec.get("task_id", f"idx{i}"))
            group_id = f"{base_id}::s{step}::r{dist_info.rank}::p{i}"
            task = _clone_task_with_group_id(rec, group_id=group_id)
            prompt_tasks.append(task)

        r = int(args.num_rollouts_per_prompt)
        rollout_states: List[_RolloutState] = []
        for task in prompt_tasks:
            obs = env.reset(task)
            for _ in range(r):
                rollout_states.append(_RolloutState(task=task, observation=obs.content))

        # 2) Generate rollouts (supports multi-turn write->test trajectories for code env)
        gen_model = model.module if hasattr(model, "module") else model
        gen_model.eval()
        sampled_sequences: List[List[int]] = []
        sampled_prompt_lens: List[int] = []
        sampled_state_indices: List[int] = []
        sampled_turn_indices: List[int] = []

        for _turn in range(max_policy_turns):
            active_indices = [i for i, st in enumerate(rollout_states) if not st.done]
            if not active_indices:
                break

            active_prompts: List[str] = []
            for idx in active_indices:
                state = rollout_states[idx]
                if args.env_type == "code":
                    active_prompts.append(
                        build_code_prompt(
                            state.observation,
                            tokenizer,
                            use_chat_template=args.use_chat_template,
                            system_prompt=sys_prompt,
                        )
                    )
                else:
                    active_prompts.append(
                        build_sql_prompt(
                            state.observation,
                            tokenizer,
                            use_chat_template=args.use_chat_template,
                            system_prompt=sys_prompt,
                        )
                    )

            enc = tokenizer(
                active_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=int(args.max_prompt_tokens),
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            prompt_lens = [int(x) for x in attention_mask.sum(dim=1).tolist()]

            with torch.no_grad():
                generated = gen_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    do_sample=True,
                    temperature=float(args.temperature),
                    top_p=float(args.top_p),
                    max_new_tokens=int(args.max_new_tokens),
                    num_return_sequences=1,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            for seq, state_idx, p_len in zip(generated, active_indices, prompt_lens):
                ids = seq.tolist()
                end = len(ids)
                if tokenizer.eos_token_id is not None:
                    try:
                        end = ids.index(int(tokenizer.eos_token_id), p_len) + 1
                    except ValueError:
                        pass

                sampled_sequences.append(ids)
                sampled_prompt_lens.append(int(p_len))
                sampled_state_indices.append(int(state_idx))
                sampled_turn_indices.append(int(rollout_states[state_idx].action_count))

                completion_text = tokenizer.decode(ids[p_len:end], skip_special_tokens=True)
                content = extract_python_code(completion_text) if args.env_type == "code" else completion_text

                state = rollout_states[state_idx]
                env.reset(state.task)
                if state.trajectory is not None:
                    for prev_step in state.trajectory.steps:
                        _, _, done_prev, _ = env.step(prev_step.action)
                        if done_prev:
                            break

                if args.env_type == "code":
                    # Mark stochastic policy actions so counterfactual credit can target multi-turn writes.
                    write_action = Action(
                        ActionType.CODE_WRITE,
                        content,
                        metadata={"logprob": 0.0, "turn_index": int(state.action_count)},
                    )
                    env.step(write_action)
                    obs_test, _, done, _ = env.step(Action(ActionType.TOOL_CALL, "", tool_name="run_tests"))
                    state.latest_feedback = obs_test.content
                else:
                    obs_sql, _, done, _ = env.step(
                        Action(ActionType.TOOL_CALL, content, tool_name="submit_query", metadata={"logprob": 0.0})
                    )
                    state.latest_feedback = obs_sql.content

                trajectory = env.get_trajectory()
                state.trajectory = trajectory
                state.action_count += 1
                state.done = bool(done) or trajectory.length >= int(args.max_trajectory_length)
                if not state.done:
                    if args.env_type == "code":
                        current_code = str((trajectory.metadata or {}).get("final_code", ""))
                        state.observation = _build_code_followup_observation(
                            state.task,
                            current_code=current_code,
                            last_feedback=state.latest_feedback,
                            show_tests=bool(args.show_tests),
                        )
                    else:
                        state.observation = state.latest_feedback

        gen_model.train()

        # 3) Collect trajectories and rollout samples for RL loss
        trajectories: List[Trajectory] = []
        for state in rollout_states:
            if state.trajectory is None:
                env.reset(state.task)
                state.trajectory = env.get_trajectory()
            trajectories.append(state.trajectory)

        local_sample_count = len(sampled_sequences)
        global_min_sample_count = local_sample_count
        if dist_info.distributed:
            import torch.distributed as dist

            sample_count_t = torch.tensor([local_sample_count], device=device, dtype=torch.long)
            dist.all_reduce(sample_count_t, op=dist.ReduceOp.MIN)
            global_min_sample_count = int(sample_count_t.item())

        if global_min_sample_count <= 0:
            # If any rank has no rollout sample, all ranks must skip this step.
            # Otherwise some ranks call backward(all-reduce) while others do not, which deadlocks DDP.
            if tracker and dist_info.is_rank0:
                tracker.log_event(
                    "train",
                    "skip step: empty rollout sample on at least one rank",
                    {
                        "step": int(step),
                        "local_sample_count": int(local_sample_count),
                        "world_size": int(dist_info.world_size),
                    },
                )
            continue

        pad_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)
        max_seq_len = max(len(ids) for ids in sampled_sequences)
        sequences = torch.full(
            (len(sampled_sequences), max_seq_len),
            fill_value=pad_id,
            dtype=torch.long,
            device=device,
        )
        for i, ids in enumerate(sampled_sequences):
            seq_len = len(ids)
            sequences[i, :seq_len] = torch.tensor(ids, dtype=torch.long, device=device)

        # Update failure replay buffer (prompt-level success/failure).
        if float(args.failure_replay_ratio) > 0.0 and args.failure_buffer_size > 0:
            group_success: Dict[str, bool] = {}
            group_base_id: Dict[str, str] = {}
            for traj in trajectories:
                gid = str(traj.task_id)
                group_success[gid] = group_success.get(gid, False) or bool(traj.success)
                base_id = _base_task_id_from_trajectory(traj)
                if base_id is not None:
                    group_base_id[gid] = base_id

            for gid, ok in group_success.items():
                base_id = group_base_id.get(gid)
                if not base_id:
                    continue
                if ok:
                    # If task now succeeds, drop one stale failure sample.
                    try:
                        failure_buffer.remove(base_id)
                    except ValueError:
                        pass
                else:
                    failure_buffer.append(base_id)

            if len(failure_buffer) > int(args.failure_buffer_size):
                failure_buffer = failure_buffer[-int(args.failure_buffer_size):]

        reward_values = [
            _trajectory_reward(
                t,
                mode=args.reward_mode,
                blend_alpha=args.reward_blend_alpha,
                failure_reward_floor=args.failure_reward_floor,
            )
            for t in trajectories
        ]

        # 4) GRPO-style advantages + optional counterfactual reweighting
        advantages = _compute_grpo_advantages(
            trajectories,
            reward_values,
            flat_group_fallback=args.flat_group_fallback,
        )

        trajectory_policy_weights: List[List[float]] = []
        credit_spreads: List[float] = []
        for traj, adv in zip(trajectories, advantages):
            w = float(adv)
            step_adv: Optional[List[float]] = None
            if args.use_counterfactual_credit and cf_generator.should_generate_counterfactuals(traj):
                interventions = cf_generator.generate(traj)
                cf_results = cf_executor.batch_execute(traj, interventions)
                credit_map = credit_estimator.estimate(
                    traj,
                    cf_results,
                    base_reward=_trajectory_reward(
                        traj,
                        mode=args.reward_mode,
                        blend_alpha=args.reward_blend_alpha,
                        failure_reward_floor=args.failure_reward_floor,
                    ),
                    cf_reward_fn=lambda cf: _cf_result_reward(
                        cf,
                        mode=args.reward_mode,
                        blend_alpha=args.reward_blend_alpha,
                        failure_reward_floor=args.failure_reward_floor,
                    ),
                )
                credit_spreads.append(credit_map.spread)
                step_adv = _to_list(advantage_mapper.map_to_step_advantages(traj, credit_map))

            lp_steps = [i for i, s in enumerate(traj.steps) if s.logprob is not None]
            if lp_steps:
                if step_adv is None:
                    per_policy = [w for _ in lp_steps]
                else:
                    if args.credit_fallback_when_zero_adv and abs(w) <= float(args.zero_adv_threshold):
                        per_policy = [float(args.credit_fallback_scale) * float(step_adv[i]) for i in lp_steps]
                    else:
                        per_policy = [w * float(step_adv[i]) for i in lp_steps]
            else:
                per_policy = [w]

            trajectory_policy_weights.append(per_policy)

        weights: List[float] = []
        for state_idx, action_idx in zip(sampled_state_indices, sampled_turn_indices):
            per_policy = trajectory_policy_weights[state_idx]
            if not per_policy:
                weights.append(0.0)
            elif action_idx < len(per_policy):
                weights.append(float(per_policy[action_idx]))
            else:
                weights.append(float(per_policy[-1]))
        if len(weights) != len(sampled_prompt_lens) or len(weights) != int(sequences.size(0)):
            raise RuntimeError(
                "Mismatch between rollout samples and RL weights: "
                f"weights={len(weights)} prompts={len(sampled_prompt_lens)} seqs={int(sequences.size(0))}"
            )

        # 5) RL loss and update
        optimizer.zero_grad(set_to_none=True)
        mb = int(args.rl_microbatch_size)
        if mb <= 0:
            # Auto: microbatch to avoid OOM under DDP / long sequences.
            mb = 2 if dist_info.distributed else 0
        loss, loss_extra = weighted_rl_loss(
            model,
            sequences.to(device),
            prompt_lens=sampled_prompt_lens,
            weights=weights,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            micro_batch_size=mb,
        )
        loss.backward()
        if args.grad_clip and args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, float(args.grad_clip))
        optimizer.step()

        if step % args.log_interval == 0:
            success_rate = sum(t.r_final for t in trajectories) / max(1, len(trajectories))
            pass_at_1 = _pass_at_k_from_groups(trajectories, k=1)
            pass_at_k = _pass_at_k_from_groups(trajectories, k=r)
            local_len_sum = float(sum(t.length for t in trajectories))
            local_action_sum = float(sum(sum(1 for s in t.steps if s.logprob is not None) for t in trajectories))
            local_traj_count = float(len(trajectories))
            avg_credit_spread = (sum(credit_spreads) / len(credit_spreads)) if credit_spreads else 0.0
            mean_reward = sum(reward_values) / max(1, len(reward_values))
            nonzero_weight_ratio = (
                sum(1 for w in weights if abs(w) > float(args.zero_adv_threshold)) / max(1, len(weights))
            )

            # Distributed mean for scalar metrics
            loss_mean = all_reduce_mean(float(loss.item()), dist_info=dist_info)
            success_mean = all_reduce_mean(float(success_rate), dist_info=dist_info)
            p1_mean = all_reduce_mean(float(pass_at_1), dist_info=dist_info)
            pk_mean = all_reduce_mean(float(pass_at_k), dist_info=dist_info)
            credit_mean = all_reduce_mean(float(avg_credit_spread), dist_info=dist_info)
            reward_mean = all_reduce_mean(float(mean_reward), dist_info=dist_info)
            nonzero_weight_mean = all_reduce_mean(float(nonzero_weight_ratio), dist_info=dist_info)
            global_len_sum_mean = all_reduce_mean(local_len_sum, dist_info=dist_info)
            global_action_sum_mean = all_reduce_mean(local_action_sum, dist_info=dist_info)
            global_traj_count_mean = all_reduce_mean(local_traj_count, dist_info=dist_info)
            avg_len_global = (
                float(global_len_sum_mean) / float(global_traj_count_mean)
                if global_traj_count_mean > 0
                else 0.0
            )
            avg_policy_actions_global = (
                float(global_action_sum_mean) / float(global_traj_count_mean)
                if global_traj_count_mean > 0
                else 0.0
            )

            if tracker:
                tracker.log_metrics(
                    RunMetrics(
                        step=step,
                        train_loss=float(loss_mean),
                        success_rate=float(success_mean),
                        pass_at_1=float(p1_mean),
                        pass_at_k={r: float(pk_mean)},
                        avg_trajectory_length=float(avg_len_global),
                        wall_time=float(time.time() - exp_wall_start),
                        avg_credit_spread=float(credit_mean),
                        extra={
                            "backend": "hf",
                            "world_size": dist_info.world_size,
                            "reward_mode": args.reward_mode,
                            "failure_reward_floor": float(args.failure_reward_floor),
                            "mean_reward": float(reward_mean),
                            "nonzero_weight_ratio": float(nonzero_weight_mean),
                            "avg_policy_actions": float(avg_policy_actions_global),
                            "failure_buffer_size": int(len(failure_buffer)),
                            "failure_replay_quota": int(replay_quota),
                            **loss_extra,
                        },
                    )
                )
                tracker.log_event("train", "logged step metrics", {
                    "loss": float(loss_mean),
                    "success_rate": float(success_mean),
                })

        if tracker and (step % args.eval_interval == 0):
            if eval_records:
                eval_subset = eval_records[: max(1, int(args.eval_tasks))]
            else:
                eval_subset = train_records[: max(1, int(args.eval_tasks))]

            # Simple pass@1 eval (greedy)
            eval_tasks: List[Dict] = []
            eval_prompts: List[str] = []
            for i, rec in enumerate(eval_subset):
                gid = f"{rec.get('task_id','eval')}::eval_s{step}::i{i}"
                task = _clone_task_with_group_id(rec, group_id=gid)
                eval_tasks.append(task)
                obs = env.reset(task)
                if args.env_type == "code":
                    eval_prompts.append(
                        build_code_prompt(
                            obs.content,
                            tokenizer,
                            use_chat_template=args.use_chat_template,
                            system_prompt=sys_prompt,
                        )
                    )
                else:
                    eval_prompts.append(
                        build_sql_prompt(
                            obs.content,
                            tokenizer,
                            use_chat_template=args.use_chat_template,
                            system_prompt=sys_prompt,
                        )
                    )

            enc = tokenizer(
                eval_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=int(args.max_prompt_tokens),
            )
            in_ids = enc["input_ids"].to(device)
            attn = enc["attention_mask"].to(device)
            p_lens = [int(x) for x in attn.sum(dim=1).tolist()]

            gen_model = model.module if hasattr(model, "module") else model
            gen_model.eval()
            with torch.no_grad():
                seqs = gen_model.generate(
                    input_ids=in_ids,
                    attention_mask=attn,
                    do_sample=False,
                    max_new_tokens=int(args.max_new_tokens),
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            gen_model.train()

            succ = 0.0
            for seq, task, p_len in zip(seqs, eval_tasks, p_lens):
                ids = seq.tolist()
                end = len(ids)
                if tokenizer.eos_token_id is not None:
                    try:
                        end = ids.index(int(tokenizer.eos_token_id), p_len) + 1
                    except ValueError:
                        pass
                completion_text = tokenizer.decode(ids[p_len:end], skip_special_tokens=True)
                content = extract_python_code(completion_text)
                env.reset(task)
                if args.env_type == "code":
                    env.step(Action(ActionType.CODE_WRITE, content))
                    _, r, _, _ = env.step(Action(ActionType.TOOL_CALL, "", tool_name="run_tests"))
                else:
                    _, r, _, _ = env.step(Action(ActionType.TOOL_CALL, content, tool_name="submit_query"))
                succ += float(r)

            pass1 = succ / max(1.0, float(len(eval_tasks)))
            tracker.log_event("eval", "eval checkpoint", {"step": step, "pass_at_1": pass1, "n": len(eval_tasks)})

        if tracker and args.save_interval and (step % args.save_interval == 0):
            ckpt_dir = Path(exp_dir) / "checkpoints" / f"step_{step}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            to_save = model.module if hasattr(model, "module") else model
            to_save.save_pretrained(str(ckpt_dir))
            tokenizer.save_pretrained(str(ckpt_dir))
            tracker.log_event("checkpoint", "saved checkpoint", {"step": step, "path": str(ckpt_dir)})

    if tracker:
        tracker.finalize()

    barrier(dist_info=dist_info)


def main():
    args = parse_args()

    config = ExperimentConfig(
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
            "max_policy_turns": int(args.max_policy_turns),
            "task_timeout_seconds": float(args.task_timeout_seconds),
        },
    )

    if args.backend == "hf":
        run_hf(args, config)
    else:
        run_toy(args, config)


if __name__ == "__main__":
    main()

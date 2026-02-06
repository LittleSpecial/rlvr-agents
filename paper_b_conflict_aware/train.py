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
import random
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Optional

sys.path.append(str(Path(__file__).parent.parent))

from shared.experiment_tracking import ExperimentTracker, ExperimentConfig, RunMetrics
from shared.envs import CodeEnv, EnvConfig, SQLEnv
from shared.envs.base import Action, ActionType, Trajectory
from shared.toy import ToyCategoricalPolicy, get_toy_tasks, compute_conflict_metrics
from paper_b_conflict_aware.conflict_detector import ConflictDetector
from paper_b_conflict_aware.gradient_surgery import GradientSurgery


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
    parser.add_argument("--model_path", type=str, default=None,
                        help="(hf) local path or HF model name; overrides --model_name")
    add_bool_flag("--use_lora", default=True, help="(hf) enable LoRA adapters")
    parser.add_argument("--lora_rank", type=int, default=64)
    add_bool_flag("--trust_remote_code", default=False, help="(hf) allow remote model code")
    add_bool_flag("--use_chat_template", default=True, help="(hf) use tokenizer chat template if available")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])

    # 训练配置
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--toy_lr", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=32, help="Number of target-skill prompts per step")
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--eval_interval", type=int, default=200)
    parser.add_argument("--num_rollouts_per_prompt", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0, help="(hf) nucleus sampling p")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="(hf) max tokens to generate")
    parser.add_argument("--max_prompt_tokens", type=int, default=1024, help="(hf) prompt truncation length")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--eval_tasks_per_skill", type=int, default=32)
    parser.add_argument("--rl_microbatch_size", type=int, default=0,
                        help="(hf) microbatch size for policy loss forward; 0=auto")
    add_bool_flag("--gradient_checkpointing", default=True, help="(hf) gradient checkpointing")
    add_bool_flag("--require_cuda", default=True, help="(hf) fail early when CUDA is unavailable")

    # 环境配置
    parser.add_argument("--env_type", type=str, default="code", choices=["code", "sql"])
    add_bool_flag("--show_tests", default=True, help="(code env) whether to show unit tests in the observation")
    parser.add_argument("--train_dataset", type=str, default=None, help="(hf) path to training JSONL")
    parser.add_argument("--eval_dataset", type=str, default=None, help="(hf) path to eval JSONL")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--max_trajectory_length", type=int, default=20)
    parser.add_argument(
        "--task_timeout_seconds",
        type=float,
        default=8.0,
        help="(code env) per-task timeout cap to avoid long stuck executions under DDP",
    )

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
    parser.add_argument("--max_protected_skills", type=int, default=2,
                        help="Upper bound of protected skills used for constraints each step")
    add_bool_flag("--shuffle_protected_skills", default=True, help="Shuffle and subsample protected skills")

    # Interference stats (analysis only)
    parser.add_argument("--conflict_threshold", type=float, default=0.0,
                        help="Cosine threshold for counting interfering skill pairs")
    parser.add_argument("--reward_mode", type=str, default="score", choices=["binary", "score", "mixed"])
    parser.add_argument("--reward_blend_alpha", type=float, default=0.7)
    parser.add_argument("--failure_reward_floor", type=float, default=-0.01,
                        help="Failed trajectories use this reward floor when reward<=0")
    parser.add_argument("--flat_group_fallback", type=str, default="raw",
                        choices=["zero", "batch_centered", "raw"])
    add_bool_flag("--auto_split_skills_if_missing", default=True,
                  help="When all tasks miss skill_key, split tasks into pseudo-skills by hash")
    parser.add_argument("--auto_num_skills", type=int, default=2)
    parser.add_argument("--grad_param_include", nargs="*", default=["lora_", "adapter"],
                        help="Only parameters containing these substrings are used for conflict stats/surgery")
    parser.add_argument("--max_tracked_tensors", type=int, default=0,
                        help="0 means no cap; >0 caps tensors used for conflict gradients")
    parser.add_argument("--system_prompt", type=str, default=None,
                        help="(hf) system prompt override")
    add_bool_flag("--apply_surgery_every_step", default=True,
                  help="If false, only compute conflict stats; update uses plain mean gradient")

    # 输出配置
    parser.add_argument("--output_dir", type=str, default="./experiments")

    return parser.parse_args()


def _clamp01(x: float) -> float:
    v = float(x)
    if not (v == v):
        return 0.0
    return max(0.0, min(1.0, v))


def _trajectory_reward(
    trajectory: Trajectory,
    *,
    mode: str,
    blend_alpha: float,
    failure_reward_floor: float,
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


def _clone_task_with_group_id(task: Dict, *, group_id: str) -> Dict:
    cloned = dict(task)
    base_task_id = str(cloned.get("task_id", "unknown"))
    cloned["task_id"] = group_id
    meta = dict(cloned.get("metadata") or {})
    meta.setdefault("base_task_id", base_task_id)
    meta.setdefault("group_id", group_id)
    cloned["metadata"] = meta
    return cloned


def _compute_grpo_advantages(
    trajectories: List[Trajectory],
    rewards: List[float],
    *,
    flat_group_fallback: str = "raw",
) -> List[float]:
    if len(trajectories) != len(rewards):
        raise ValueError("trajectories/rewards length mismatch")
    rewards = [float(r) for r in rewards]
    batch_mean = statistics.fmean(rewards) if rewards else 0.0
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


def _guess_lora_target_modules(model) -> List[str]:
    candidates = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    names = {name.split(".")[-1] for name, _ in model.named_modules()}
    picked = [c for c in candidates if c in names]
    return picked if picked else ["q_proj", "k_proj", "v_proj", "o_proj"]


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


def _task_skill(task: Dict, *, skill_key: str) -> Optional[str]:
    meta = task.get("metadata") if isinstance(task, dict) else None
    if not isinstance(meta, dict):
        return None
    val = meta.get(skill_key)
    if val is None:
        return None
    sval = str(val).strip()
    return sval if sval else None


def _inject_auto_skills(
    records: List[Dict],
    *,
    skill_key: str,
    auto_num_skills: int,
) -> None:
    buckets = max(2, int(auto_num_skills))
    for rec in records:
        tid = str(rec.get("task_id", "unknown"))
        skill = f"auto_{abs(hash(tid)) % buckets}"
        meta = dict(rec.get("metadata") or {})
        meta.setdefault(skill_key, skill)
        rec["metadata"] = meta


def _build_tasks_by_skill(records: List[Dict], *, skill_key: str) -> Dict[str, List[Dict]]:
    by_skill: Dict[str, List[Dict]] = {}
    for rec in records:
        sid = _task_skill(rec, skill_key=skill_key) or "unknown"
        by_skill.setdefault(sid, []).append(rec)
    return by_skill


def _sequential_margin_project_torch(
    candidate,
    constraints: List,
    *,
    epsilon: float,
    normalize_to: Optional[float],
):
    import torch

    out = candidate.clone()
    for g_s in constraints:
        d = torch.dot(g_s, out)
        if d.item() < -float(epsilon):
            denom = torch.dot(g_s, g_s) + 1e-12
            alpha = (-float(epsilon) - float(d.item())) / float(denom.item())
            out = out + alpha * g_s

    if normalize_to is not None:
        n = float(torch.norm(out).item())
        if n > 1e-12:
            out = out * (float(normalize_to) / n)
    return out


def _build_param_filter(include_names: Optional[List[str]]) -> Optional[Callable[[str, object], bool]]:
    if include_names is None:
        return None
    names = [str(s) for s in include_names if str(s).strip()]
    if not names:
        return None
    return lambda n, _p: any(s in n for s in names)


def run_hf(args, config: ExperimentConfig) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        from peft import LoraConfig, get_peft_model  # type: ignore
    except ModuleNotFoundError as e:
        raise SystemExit(
            "Missing dependency `peft` for LoRA. Install requirements first.\n"
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
        build_attention_and_labels,
        per_sample_nll,
    )
    from shared.hf.prompts import DEFAULT_SYSTEM_PROMPT, DEFAULT_SQL_SYSTEM_PROMPT
    from shared.hf.distributed import barrier

    dist_info: DistInfo = init_distributed()
    has_cuda = torch.cuda.is_available()
    if args.require_cuda and not has_cuda:
        if dist_info.is_rank0:
            print(
                "\n[ERROR] CUDA is not available in current environment.\n"
                "This run requires CUDA to avoid accidental CPU training.\n",
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

    tracker = None
    exp_dir = ""
    if dist_info.is_rank0:
        tracker = ExperimentTracker(config, base_dir=args.output_dir)
        tracker.log_event("init", "Experiment initialized", {"world_size": dist_info.world_size})
        exp_dir = str(tracker.experiment_dir)
    exp_dir = str(broadcast_object(exp_dir, src=0, dist_info=dist_info))
    barrier(dist_info=dist_info)
    exp_wall_start = time.time()

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

    train_records = load_jsonl(Path(args.train_dataset), max_samples=args.max_train_samples)
    if not train_records:
        raise RuntimeError(f"Empty train dataset: {args.train_dataset}")
    eval_records: List[Dict] = []
    if args.eval_dataset:
        eval_records = load_jsonl(Path(args.eval_dataset), max_samples=args.max_eval_samples)

    train_by_skill = _build_tasks_by_skill(train_records, skill_key=args.skill_key)
    if args.auto_split_skills_if_missing and len(train_by_skill) <= 1:
        _inject_auto_skills(train_records, skill_key=args.skill_key, auto_num_skills=args.auto_num_skills)
        train_by_skill = _build_tasks_by_skill(train_records, skill_key=args.skill_key)
        if eval_records:
            _inject_auto_skills(eval_records, skill_key=args.skill_key, auto_num_skills=args.auto_num_skills)

    eval_source = eval_records if eval_records else train_records
    eval_by_skill = _build_tasks_by_skill(eval_source, skill_key=args.skill_key)
    skill_ids = sorted(train_by_skill.keys())
    if not skill_ids:
        raise RuntimeError("No skills found in dataset")

    if args.protocol == "sequential":
        sequence = args.skill_sequence[:] if args.skill_sequence else skill_ids
        missing = [s for s in sequence if s not in train_by_skill]
        if missing:
            raise ValueError(f"Unknown skills in --skill_sequence: {missing}; available={skill_ids}")
    else:
        sequence = skill_ids

    rng = random.Random(args.seed + 1337 * dist_info.rank)

    model_name_or_path = args.model_path or args.model_name
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        padding_side="left",
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

    model = model.to(device)
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            try:
                model.enable_input_require_grads()
            except Exception:
                pass

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters found; check LoRA attachment")
    optimizer = torch.optim.AdamW(trainable_params, lr=float(args.learning_rate))

    detector = ConflictDetector(
        num_groups=max(2, len(skill_ids)),
        conflict_threshold=float(args.conflict_threshold),
        prefer_adapter_params=True,
        max_tracked_tensors=int(args.max_tracked_tensors),
    )
    surgery = GradientSurgery(
        method="pcgrad" if args.projection == "pcgrad" else "none",
        conflict_threshold=float(args.conflict_threshold),
        normalize=bool(args.normalize_update),
        only_adapter_layers=True,
    )
    include_param_names = [s for s in args.grad_param_include if str(s).strip()]
    include_names = include_param_names if include_param_names else None
    param_filter = _build_param_filter(include_names)
    tracked_preview = detector.select_tracked_parameters(
        model,
        include_param_names=include_names,
        param_filter=param_filter,
    )
    if not tracked_preview and include_names is not None:
        if dist_info.is_rank0:
            print(
                "[WARN] --grad_param_include matched no trainable params; "
                "fallback to all trainable params for conflict stats/surgery.",
                flush=True,
            )
        include_names = None
        param_filter = None
        tracked_preview = detector.select_tracked_parameters(
            model,
            include_param_names=include_names,
            param_filter=param_filter,
        )
    if not tracked_preview:
        raise RuntimeError("No trainable parameters available for conflict stats/surgery")

    if dist_info.is_rank0:
        print(f"Experiment directory: {exp_dir}", flush=True)
        print("\n" + "=" * 50, flush=True)
        print("Paper B: Interference-aware RLVR (HF backend)", flush=True)
        print("=" * 50, flush=True)
        print(f"Model: {model_name_or_path}", flush=True)
        print(f"Environment: {args.env_type}", flush=True)
        print(f"Protocol: {args.protocol}", flush=True)
        print(f"Projection: {args.projection}, epsilon={args.epsilon}", flush=True)
        print(f"Skill key: {args.skill_key}, skills: {skill_ids}", flush=True)
        print("=" * 50, flush=True)

    if tracker:
        tracker.log_event("components", "Paper B components initialized", {
            "protocol": args.protocol,
            "projection": args.projection,
            "epsilon": float(args.epsilon),
            "skill_key": args.skill_key,
            "skills": skill_ids,
            "reward_mode": args.reward_mode,
            "failure_reward_floor": float(args.failure_reward_floor),
            "flat_group_fallback": args.flat_group_fallback,
            "grad_param_include": include_names,
            "tracked_param_tensors": len(tracked_preview),
        })

    sys_prompt = (
        args.system_prompt
        if args.system_prompt is not None
        else (DEFAULT_SYSTEM_PROMPT if args.env_type == "code" else DEFAULT_SQL_SYSTEM_PROMPT)
    )

    best_eval: Dict[str, float] = {sid: 0.0 for sid in eval_by_skill.keys()}
    if tracker and dist_info.is_rank0:
        tracker.log_event("eval", "initial skill eval", {"skills": list(eval_by_skill.keys())})

    for step in range(1, args.max_steps + 1):
        if args.protocol == "sequential":
            stage_idx = min((step - 1) // max(1, int(args.stage_steps)), len(sequence) - 1)
            target_skill = sequence[stage_idx]
            protected_skills = sequence[:stage_idx]
        else:
            target_skill = "mixed"
            protected_skills = [s for s in skill_ids]

        if args.shuffle_protected_skills and protected_skills:
            rng.shuffle(protected_skills)
        if args.max_protected_skills > 0:
            protected_skills = protected_skills[: int(args.max_protected_skills)]

        prompt_tasks: List[Dict] = []
        prompt_texts: List[str] = []

        target_pool = train_records if target_skill == "mixed" else train_by_skill.get(target_skill, [])
        if not target_pool:
            target_pool = train_records

        for i in range(int(args.batch_size)):
            rec = rng.choice(target_pool)
            base_id = str(rec.get("task_id", f"idx{i}"))
            gid = f"{base_id}::s{step}::r{dist_info.rank}::p{i}"
            task = _clone_task_with_group_id(rec, group_id=gid)
            prompt_tasks.append(task)

        if args.protocol == "sequential" and protected_skills and int(args.memory_per_protected) > 0:
            for sid in protected_skills:
                pool = train_by_skill.get(sid, [])
                if not pool:
                    continue
                for j in range(int(args.memory_per_protected)):
                    rec = rng.choice(pool)
                    base_id = str(rec.get("task_id", f"{sid}_{j}"))
                    gid = f"{base_id}::s{step}::r{dist_info.rank}::m{sid}_{j}"
                    task = _clone_task_with_group_id(rec, group_id=gid)
                    prompt_tasks.append(task)

        for task in prompt_tasks:
            obs = env.reset(task)
            if args.env_type == "code":
                prompt_texts.append(
                    build_code_prompt(
                        obs.content,
                        tokenizer,
                        use_chat_template=args.use_chat_template,
                        system_prompt=sys_prompt,
                    )
                )
            else:
                prompt_texts.append(
                    build_sql_prompt(
                        obs.content,
                        tokenizer,
                        use_chat_template=args.use_chat_template,
                        system_prompt=sys_prompt,
                    )
                )

        enc = tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(args.max_prompt_tokens),
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        prompt_lens_base = [int(x) for x in attention_mask.sum(dim=1).tolist()]

        model.eval()
        with torch.no_grad():
            sequences = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=True,
                temperature=float(args.temperature),
                top_p=float(args.top_p),
                max_new_tokens=int(args.max_new_tokens),
                num_return_sequences=int(args.num_rollouts_per_prompt),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        model.train()

        r = int(args.num_rollouts_per_prompt)
        expanded_prompt_lens = [prompt_lens_base[i // r] for i in range(len(prompt_tasks) * r)]
        expanded_tasks = [prompt_tasks[i // r] for i in range(len(prompt_tasks) * r)]

        trajectories: List[Trajectory] = []
        for seq, task, p_len in zip(sequences, expanded_tasks, expanded_prompt_lens):
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
                env.step(Action(ActionType.CODE_WRITE, content, metadata={"logprob": 0.0}))
                env.step(Action(ActionType.TOOL_CALL, "", tool_name="run_tests"))
            else:
                env.step(Action(ActionType.TOOL_CALL, content, tool_name="submit_query", metadata={"logprob": 0.0}))
            trajectories.append(env.get_trajectory())

        reward_values = [
            _trajectory_reward(
                traj,
                mode=args.reward_mode,
                blend_alpha=args.reward_blend_alpha,
                failure_reward_floor=args.failure_reward_floor,
            )
            for traj in trajectories
        ]
        advantages = _compute_grpo_advantages(
            trajectories,
            reward_values,
            flat_group_fallback=args.flat_group_fallback,
        )
        weights = [float(a) for a in advantages]

        sample_skill_ids: List[str] = []
        for traj in trajectories:
            sid = str(traj.metadata.get(args.skill_key) or "unknown")
            if sid == "unknown":
                task_meta = ((traj.metadata or {}).get("task") or {}).get("metadata") or {}
                sid = str(task_meta.get(args.skill_key, "unknown"))
            sample_skill_ids.append(sid)

        group_indices: Dict[str, List[int]] = {}
        for idx, sid in enumerate(sample_skill_ids):
            group_indices.setdefault(sid, []).append(idx)

        attn, labels, comp_lens = build_attention_and_labels(
            sequences.to(device),
            prompt_lens=expanded_prompt_lens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        out = model(input_ids=sequences.to(device), attention_mask=attn, use_cache=False)
        nll_sum = per_sample_nll(out.logits, labels)
        nll_mean = nll_sum / comp_lens.to(nll_sum.dtype)
        w_t = torch.tensor(weights, device=device, dtype=nll_mean.dtype)
        weighted = w_t * nll_mean
        objective_loss = weighted.mean()

        skill_to_gid: Dict[str, int] = {sid: i for i, sid in enumerate(sorted(group_indices.keys()))}
        group_losses: Dict[int, torch.Tensor] = {}
        for sid, indices in group_indices.items():
            idx_t = torch.tensor(indices, device=device, dtype=torch.long)
            group_losses[skill_to_gid[sid]] = weighted.index_select(0, idx_t).mean()

        group_grads = detector.compute_group_gradients(
            model,
            group_losses,
            include_param_names=include_names,
            param_filter=param_filter,
        )
        conflict = detector.detect_conflicts(group_grads, step=step)

        if args.protocol == "sequential":
            target_gid = skill_to_gid.get(target_skill)
            if target_gid is None and group_grads:
                target_gid = sorted(group_grads.keys())[0]
            candidate = group_grads[target_gid].clone() if target_gid is not None else torch.stack(list(group_grads.values())).mean(dim=0)
            protected_gids = [skill_to_gid[s] for s in protected_skills if s in skill_to_gid]
            constraints = [group_grads[g] for g in protected_gids]
        else:
            candidate = torch.stack(list(group_grads.values())).mean(dim=0)
            constraints = list(group_grads.values())

        if args.apply_surgery_every_step:
            if args.projection == "none":
                final_grad = candidate
            elif args.projection == "pcgrad":
                if args.protocol == "sequential":
                    involved: Dict[int, torch.Tensor] = {}
                    if args.protocol == "sequential":
                        for sid in [target_skill] + list(protected_skills):
                            gid = skill_to_gid.get(sid)
                            if gid is not None and gid in group_grads:
                                involved[gid] = group_grads[gid]
                    if not involved:
                        involved = group_grads
                    final_grad = surgery.apply(involved)
                else:
                    final_grad = surgery.apply(group_grads)
            else:
                base_norm = float(torch.norm(candidate).item()) if args.normalize_update else None
                final_grad = _sequential_margin_project_torch(
                    candidate,
                    constraints,
                    epsilon=float(args.epsilon),
                    normalize_to=base_norm,
                )
        else:
            final_grad = candidate

        tracked = detector.select_tracked_parameters(
            model,
            include_param_names=include_names,
            param_filter=param_filter,
        )
        if not tracked:
            raise RuntimeError("No tracked parameters for Paper B HF gradients")

        total_numel = sum(int(p.numel()) for _, p in tracked)
        if int(final_grad.numel()) != total_numel:
            raise RuntimeError(
                f"Gradient size mismatch: final_grad={int(final_grad.numel())} tracked_params={total_numel}"
            )

        optimizer.zero_grad(set_to_none=True)
        cursor = 0
        for _, p in tracked:
            n = int(p.numel())
            grad_chunk = final_grad[cursor: cursor + n].view_as(p).to(dtype=p.dtype, device=p.device)
            if p.grad is None:
                p.grad = grad_chunk.clone()
            else:
                p.grad.copy_(grad_chunk)
            cursor += n

        if dist_info.distributed:
            import torch.distributed as dist
            for _, p in tracked:
                if p.grad is None:
                    continue
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                p.grad /= float(dist_info.world_size)

        if args.grad_clip and args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_([p for _, p in tracked], float(args.grad_clip))
        optimizer.step()
        del out

        if step % args.log_interval == 0:
            success_rate = sum(t.r_final for t in trajectories) / max(1, len(trajectories))
            pass_at_1 = _pass_at_k_from_groups(trajectories, k=1)
            pass_at_k = _pass_at_k_from_groups(trajectories, k=r)
            avg_len = sum(t.length for t in trajectories) / max(1, len(trajectories))
            group_success: Dict[str, float] = {}
            for sid, indices in group_indices.items():
                group_success[sid] = float(sum(float(trajectories[i].r_final) for i in indices) / max(1, len(indices)))

            loss_mean = all_reduce_mean(float(objective_loss.item()), dist_info=dist_info)
            success_mean = all_reduce_mean(float(success_rate), dist_info=dist_info)
            p1_mean = all_reduce_mean(float(pass_at_1), dist_info=dist_info)
            pk_mean = all_reduce_mean(float(pass_at_k), dist_info=dist_info)
            conflict_mean = all_reduce_mean(float(conflict.conflict_ratio), dist_info=dist_info)
            angle_mean = all_reduce_mean(float(conflict.avg_conflict_angle), dist_info=dist_info)
            reward_mean = all_reduce_mean(float(sum(reward_values) / max(1, len(reward_values))), dist_info=dist_info)
            weight_abs_mean = all_reduce_mean(float(sum(abs(w) for w in weights) / max(1, len(weights))), dist_info=dist_info)
            nz_weight_mean = all_reduce_mean(
                float(sum(1 for w in weights if abs(w) > 1e-8) / max(1, len(weights))),
                dist_info=dist_info,
            )

            if tracker:
                tracker.log_metrics(
                    RunMetrics(
                        step=step,
                        train_loss=float(loss_mean),
                        success_rate=float(success_mean),
                        pass_at_1=float(p1_mean),
                        pass_at_k={r: float(pk_mean)},
                        avg_trajectory_length=float(avg_len),
                        wall_time=float(time.time() - exp_wall_start),
                        conflict_ratio=float(conflict_mean),
                        solution_diversity=0.0,
                        avg_credit_spread=0.0,
                        extra={
                            "backend": "hf",
                            "world_size": dist_info.world_size,
                            "target_skill": target_skill,
                            "protected_skills": protected_skills,
                            "reward_mode": args.reward_mode,
                            "mean_reward": float(reward_mean),
                            "mean_abs_weight": float(weight_abs_mean),
                            "nonzero_weight_ratio": float(nz_weight_mean),
                            "interference_avg_angle_deg": float(angle_mean),
                            "candidate_grad_norm": float(torch.norm(candidate).item()),
                            "final_grad_norm": float(torch.norm(final_grad).item()),
                            "group_skill_success": group_success,
                            "group_sizes": {sid: len(indices) for sid, indices in group_indices.items()},
                            "n_groups": int(len(group_indices)),
                        },
                    )
                )
                tracker.log_event("interference", "logged interference stats", {
                    "skill_ids": sorted(group_indices.keys()),
                    "conflict_ratio": float(conflict.conflict_ratio),
                    "avg_conflict_angle": float(conflict.avg_conflict_angle),
                })

        if tracker and dist_info.is_rank0 and (step % args.eval_interval == 0):
            eval_per_skill: Dict[str, float] = {}
            model.eval()
            with torch.no_grad():
                for sid, pool in eval_by_skill.items():
                    if not pool:
                        continue
                    sample_n = min(len(pool), max(1, int(args.eval_tasks_per_skill)))
                    subset = rng.sample(pool, sample_n) if len(pool) > sample_n else list(pool)
                    succ = 0.0
                    for i, rec in enumerate(subset):
                        gid = f"{rec.get('task_id','eval')}::eval_s{step}::{sid}::{i}"
                        task = _clone_task_with_group_id(rec, group_id=gid)
                        obs = env.reset(task)
                        if args.env_type == "code":
                            prompt = build_code_prompt(
                                obs.content,
                                tokenizer,
                                use_chat_template=args.use_chat_template,
                                system_prompt=sys_prompt,
                            )
                        else:
                            prompt = build_sql_prompt(
                                obs.content,
                                tokenizer,
                                use_chat_template=args.use_chat_template,
                                system_prompt=sys_prompt,
                            )
                        enc_e = tokenizer(
                            [prompt],
                            return_tensors="pt",
                            padding=True,
                            truncation=True,
                            max_length=int(args.max_prompt_tokens),
                        )
                        in_ids = enc_e["input_ids"].to(device)
                        attn_e = enc_e["attention_mask"].to(device)
                        p_len = int(attn_e.sum(dim=1).item())
                        seq = model.generate(
                            input_ids=in_ids,
                            attention_mask=attn_e,
                            do_sample=False,
                            max_new_tokens=int(args.max_new_tokens),
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                        )[0]
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
                            _, r_eval, _, _ = env.step(Action(ActionType.TOOL_CALL, "", tool_name="run_tests"))
                        else:
                            _, r_eval, _, _ = env.step(Action(ActionType.TOOL_CALL, content, tool_name="submit_query"))
                        succ += float(r_eval)
                    eval_per_skill[sid] = succ / max(1.0, float(sample_n))
            model.train()

            for sid, cur in eval_per_skill.items():
                best_eval[sid] = max(float(best_eval.get(sid, 0.0)), float(cur))
            forgetting = {sid: float(best_eval[sid] - eval_per_skill.get(sid, 0.0)) for sid in best_eval.keys()}
            tracker.log_event("eval", "skill eval checkpoint", {
                "step": step,
                "per_skill": eval_per_skill,
                "best_per_skill": best_eval,
                "forgetting": forgetting,
                "avg": float(sum(eval_per_skill.values()) / max(1, len(eval_per_skill))),
                "worst": float(min(eval_per_skill.values())) if eval_per_skill else 0.0,
                "max_forgetting": float(max(forgetting.values())) if forgetting else 0.0,
            })

        if tracker and dist_info.is_rank0 and args.save_interval and (step % args.save_interval == 0):
            ckpt_dir = Path(exp_dir) / "checkpoints" / f"step_{step}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(ckpt_dir))
            tokenizer.save_pretrained(str(ckpt_dir))
            tracker.log_event("checkpoint", "saved checkpoint", {"step": step, "path": str(ckpt_dir)})

    if tracker and dist_info.is_rank0:
        tracker.finalize()
    barrier(dist_info=dist_info)


def main():
    args = parse_args()

    # 创建实验配置
    config = ExperimentConfig(
        experiment_name=args.experiment_name,
        project="paper_b",
        description=args.description,
        model_name=str(args.model_path or args.model_name),
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        algorithm=("TOY_PG" if args.backend == "toy" else "CONFLICT_GRPO"),
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        env_type=args.env_type,
        max_trajectory_length=args.max_trajectory_length,
        use_conflict_aware=True,
        num_groups=0,
        surgery_method=args.projection,
        seed=args.seed,
        extra={
            "backend": args.backend,
            "protocol": args.protocol,
            "skill_key": args.skill_key,
            "epsilon": float(args.epsilon),
            "memory_per_protected": args.memory_per_protected,
            "reward_mode": args.reward_mode,
            "failure_reward_floor": float(args.failure_reward_floor),
            "flat_group_fallback": args.flat_group_fallback,
        },
    )

    if args.backend == "hf":
        run_hf(args, config)
        return

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

        rewards = [float(t.r_final) for t in trajectories]
        advantages = _compute_grpo_advantages(
            trajectories,
            rewards,
            flat_group_fallback="raw",
        )

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

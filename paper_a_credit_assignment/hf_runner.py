"""
HF backend runner for Paper A.
"""

from __future__ import annotations

import random
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional

from shared.envs import CodeEnv, EnvConfig, SQLEnv
from shared.envs.base import Action, ActionType, Trajectory
from shared.experiment_tracking import ExperimentConfig, ExperimentTracker, RunMetrics

from paper_a_credit_assignment import (
    AdvantageMapper,
    CounterfactualExecutor,
    CounterfactualGenerator,
    CreditEstimator,
)
from paper_a_credit_assignment.train_utils import (
    RolloutState,
    base_task_id_from_trajectory,
    build_code_followup_observation,
    cf_result_reward,
    clone_task_with_group_id,
    compute_grpo_advantages,
    guess_lora_target_modules,
    pass_at_k_from_groups,
    to_float_list,
    trajectory_reward,
)


def _rank_print(dist_info, message: str, *, all_ranks: bool = False) -> None:
    if all_ranks or dist_info.is_rank0:
        prefix = f"[Rank {dist_info.rank}] " if all_ranks else ""
        print(f"{prefix}{message}", flush=True)


def _evaluate_pass_at_1(
    *,
    model,
    tokenizer,
    env,
    args,
    eval_records: List[Dict],
    train_records: List[Dict],
    sys_prompt: str,
    step: int,
    max_eval_tasks: int,
) -> Dict[str, float]:
    from shared.hf import build_code_prompt, build_sql_prompt, extract_python_code

    if eval_records:
        eval_subset = eval_records[: max(1, int(max_eval_tasks))]
    else:
        eval_subset = train_records[: max(1, int(max_eval_tasks))]

    eval_tasks: List[Dict] = []
    eval_prompts: List[str] = []
    for i, rec in enumerate(eval_subset):
        gid = f"{rec.get('task_id', 'eval')}::eval_s{step}::i{i}"
        task = clone_task_with_group_id(rec, group_id=gid)
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

    if not eval_prompts:
        return {"pass_at_1": 0.0, "n": 0.0}

    import torch

    device = next(model.parameters()).device
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
        content = extract_python_code(completion_text) if args.env_type == "code" else completion_text
        env.reset(task)
        if args.env_type == "code":
            env.step(Action(ActionType.CODE_WRITE, content))
            _, r, _, _ = env.step(Action(ActionType.TOOL_CALL, "", tool_name="run_tests"))
        else:
            _, r, _, _ = env.step(Action(ActionType.TOOL_CALL, content, tool_name="submit_query"))
        succ += float(r)

    pass1 = succ / max(1.0, float(len(eval_tasks)))
    return {"pass_at_1": float(pass1), "n": float(len(eval_tasks))}


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
        all_reduce_mean,
        broadcast_object,
        build_code_prompt,
        build_sql_prompt,
        extract_python_code,
        init_distributed,
        load_jsonl,
        weighted_rl_loss,
    )
    from shared.hf.distributed import barrier
    from shared.hf.prompts import DEFAULT_SQL_SYSTEM_PROMPT, DEFAULT_SYSTEM_PROMPT

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
                "failure_replay_ratio": float(args.failure_replay_ratio),
                "failure_buffer_unique": bool(args.failure_buffer_unique),
                "replay_min_success_ema": float(args.replay_min_success_ema),
                "replay_ema_alpha": float(args.replay_ema_alpha),
                "guard_all_negative_batch": bool(args.guard_all_negative_batch),
                "all_negative_reward_span_threshold": float(args.all_negative_reward_span_threshold),
                "max_policy_turns": int(args.max_policy_turns),
                "task_timeout_seconds": float(args.task_timeout_seconds),
                "cap_task_timeout": True,
                "sync_eval_and_save": bool(args.sync_eval_and_save),
                "truncate_to_global_min_samples": bool(args.truncate_to_global_min_samples),
                "fallback_to_adv_when_zero_credit": bool(args.fallback_to_adv_when_zero_credit),
                "zero_credit_threshold": float(args.zero_credit_threshold),
            },
        )

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
    failure_buffer_set = set()

    model_name_or_path = args.model_path or args.model_name
    if dist_info.is_rank0:
        print("Loading tokenizer...", flush=True)
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
        targets = guess_lora_target_modules(model)
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

    if device.type == "cuda":
        _rank_print(dist_info, f"Moving model to {device}...", all_ranks=True)
        model = model.to(device)
        _rank_print(dist_info, f"Model moved to {device}.", all_ranks=True)

    raw_model = model.module if hasattr(model, "module") else model
    if hasattr(raw_model, "config") and hasattr(raw_model.config, "use_cache"):
        raw_model.config.use_cache = False
    if args.gradient_checkpointing and hasattr(raw_model, "gradient_checkpointing_enable"):
        raw_model.gradient_checkpointing_enable()
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
        _rank_print(dist_info, "Creating DDP wrapper...", all_ranks=True)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dist_info.local_rank] if device.type == "cuda" else None,
        )
        _rank_print(dist_info, "DDP wrapper created.", all_ranks=True)

    _rank_print(dist_info, "Creating optimizer...", all_ranks=True)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters (did LoRA attach correctly?)")
    optimizer = torch.optim.AdamW(trainable_params, lr=float(args.learning_rate))
    _rank_print(dist_info, "Optimizer created.", all_ranks=True)

    if args.system_prompt is not None:
        sys_prompt = args.system_prompt
    else:
        sys_prompt = DEFAULT_SYSTEM_PROMPT if args.env_type == "code" else DEFAULT_SQL_SYSTEM_PROMPT

    max_policy_turns = max(1, int(args.max_policy_turns))
    if args.env_type != "code":
        max_policy_turns = 1

    heartbeat_interval = int(args.heartbeat_interval)
    replay_success_ema: Optional[float] = None
    replay_ema_alpha = max(0.0, min(1.0, float(args.replay_ema_alpha)))
    replay_min_success_ema = max(0.0, min(1.0, float(args.replay_min_success_ema)))
    _rank_print(dist_info, "Entering training loop...")

    for step in range(1, args.max_steps + 1):
        step_start = time.time()
        heartbeat = heartbeat_interval > 0 and (step == 1 or step % heartbeat_interval == 0)
        if heartbeat and dist_info.is_rank0:
            print(f"[Rank 0] Step {step}/{args.max_steps} start", flush=True)

        try:
            prompt_tasks: List[Dict] = []
            replay_quota = 0
            replay_gate_blocked = False
            if (
                float(args.failure_replay_ratio) > 0.0
                and step > int(args.failure_replay_warmup_steps)
                and len(failure_buffer) > 0
            ):
                if replay_success_ema is not None and replay_success_ema < replay_min_success_ema:
                    replay_gate_blocked = True
                else:
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
                task = clone_task_with_group_id(rec, group_id=group_id)
                prompt_tasks.append(task)

            r = int(args.num_rollouts_per_prompt)
            rollout_states: List[RolloutState] = []
            for task in prompt_tasks:
                obs = env.reset(task)
                for _ in range(r):
                    rollout_states.append(RolloutState(task=task, observation=obs.content))

            rollout_build_end = time.time()

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
                            state.observation = build_code_followup_observation(
                                state.task,
                                current_code=current_code,
                                last_feedback=state.latest_feedback,
                                show_tests=bool(args.show_tests),
                            )
                        else:
                            state.observation = state.latest_feedback

            gen_model.train()
            rollout_end = time.time()

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
                if heartbeat and dist_info.is_rank0:
                    print(f"[Rank 0] Step {step} skipped: global_min_sample_count=0", flush=True)
                continue

            if bool(args.truncate_to_global_min_samples) and local_sample_count > global_min_sample_count:
                sampled_sequences = sampled_sequences[:global_min_sample_count]
                sampled_prompt_lens = sampled_prompt_lens[:global_min_sample_count]
                sampled_state_indices = sampled_state_indices[:global_min_sample_count]
                sampled_turn_indices = sampled_turn_indices[:global_min_sample_count]

            pad_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)
            max_seq_len = max(len(ids) for ids in sampled_sequences)
            sequences = torch.full(
                (len(sampled_sequences), max_seq_len),
                fill_value=pad_id,
                dtype=torch.long,
                device=device,
            )
            for i, ids in enumerate(sampled_sequences):
                sequences[i, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)

            if float(args.failure_replay_ratio) > 0.0 and args.failure_buffer_size > 0:
                group_success: Dict[str, bool] = {}
                group_base_id: Dict[str, str] = {}
                for traj in trajectories:
                    gid = str(traj.task_id)
                    group_success[gid] = group_success.get(gid, False) or bool(traj.success)
                    base_id = base_task_id_from_trajectory(traj)
                    if base_id is not None:
                        group_base_id[gid] = base_id

                for gid, ok in group_success.items():
                    base_id = group_base_id.get(gid)
                    if not base_id:
                        continue
                    if ok:
                        if bool(args.failure_buffer_unique):
                            if base_id in failure_buffer_set:
                                failure_buffer_set.discard(base_id)
                                try:
                                    failure_buffer.remove(base_id)
                                except ValueError:
                                    pass
                        else:
                            try:
                                failure_buffer.remove(base_id)
                            except ValueError:
                                pass
                    else:
                        if bool(args.failure_buffer_unique):
                            if base_id not in failure_buffer_set:
                                failure_buffer.append(base_id)
                                failure_buffer_set.add(base_id)
                        else:
                            failure_buffer.append(base_id)

                if len(failure_buffer) > int(args.failure_buffer_size):
                    overflow = len(failure_buffer) - int(args.failure_buffer_size)
                    removed = failure_buffer[:overflow]
                    failure_buffer = failure_buffer[overflow:]
                    if bool(args.failure_buffer_unique):
                        for rid in removed:
                            failure_buffer_set.discard(rid)

            reward_values = [
                trajectory_reward(
                    t,
                    mode=args.reward_mode,
                    blend_alpha=args.reward_blend_alpha,
                    failure_reward_floor=args.failure_reward_floor,
                )
                for t in trajectories
            ]
            step_success_rate = sum(t.r_final for t in trajectories) / max(1, len(trajectories))
            step_success_rate_mean = all_reduce_mean(float(step_success_rate), dist_info=dist_info)
            if replay_success_ema is None:
                replay_success_ema = float(step_success_rate_mean)
            else:
                replay_success_ema = (1.0 - replay_ema_alpha) * replay_success_ema + replay_ema_alpha * float(
                    step_success_rate_mean
                )

            advantages = compute_grpo_advantages(
                trajectories,
                reward_values,
                flat_group_fallback=args.flat_group_fallback,
            )

            trajectory_policy_weights: List[List[float]] = []
            credit_spreads: List[float] = []
            zero_credit_fallback_hits = 0
            for traj, adv in zip(trajectories, advantages):
                w = float(adv)
                step_adv: Optional[List[float]] = None
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
                    if step_adv is None:
                        per_policy = [w for _ in lp_steps]
                    else:
                        selected_step_adv = [float(step_adv[i]) for i in lp_steps]
                        if (
                            args.fallback_to_adv_when_zero_credit
                            and max(abs(v) for v in selected_step_adv) <= float(args.zero_credit_threshold)
                        ):
                            zero_credit_fallback_hits += 1
                            per_policy = [w for _ in lp_steps]
                        elif args.credit_fallback_when_zero_adv and abs(w) <= float(args.zero_adv_threshold):
                            per_policy = [float(args.credit_fallback_scale) * v for v in selected_step_adv]
                        else:
                            per_policy = [w * v for v in selected_step_adv]
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

            reward_span = (max(reward_values) - min(reward_values)) if reward_values else 0.0
            all_non_positive_weights = bool(weights) and all(w <= float(args.zero_adv_threshold) for w in weights)
            all_negative_guard_triggered = bool(args.guard_all_negative_batch) and (
                all_non_positive_weights and reward_span <= float(args.all_negative_reward_span_threshold)
            )
            if all_negative_guard_triggered:
                weights = [0.0 for _ in weights]

            optimizer.zero_grad(set_to_none=True)
            mb = int(args.rl_microbatch_size)
            if mb <= 0:
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
            local_max_abs_weight = max((abs(w) for w in weights), default=0.0)
            global_max_abs_weight = float(local_max_abs_weight)
            if dist_info.distributed:
                import torch.distributed as dist

                max_w_t = torch.tensor([global_max_abs_weight], device=device, dtype=torch.float32)
                dist.all_reduce(max_w_t, op=dist.ReduceOp.MAX)
                global_max_abs_weight = float(max_w_t.item())
            skip_optimizer_step = global_max_abs_weight <= float(args.zero_adv_threshold)

            if not skip_optimizer_step:
                loss.backward()
                if args.grad_clip and args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, float(args.grad_clip))
                optimizer.step()
            update_end = time.time()

            if step % args.log_interval == 0:
                success_rate = step_success_rate
                pass_at_1 = pass_at_k_from_groups(trajectories, k=1)
                pass_at_k = pass_at_k_from_groups(trajectories, k=r)
                local_len_sum = float(sum(t.length for t in trajectories))
                local_action_sum = float(sum(sum(1 for s in t.steps if s.logprob is not None) for t in trajectories))
                local_traj_count = float(len(trajectories))
                avg_credit_spread = (sum(credit_spreads) / len(credit_spreads)) if credit_spreads else 0.0
                mean_reward = sum(reward_values) / max(1, len(reward_values))
                nonzero_weight_ratio = (
                    sum(1 for w in weights if abs(w) > float(args.zero_adv_threshold)) / max(1, len(weights))
                )
                zero_credit_fallback_ratio = float(zero_credit_fallback_hits) / max(1.0, float(len(trajectories)))

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
                                "failure_buffer_unique_size": int(
                                    len(failure_buffer_set) if bool(args.failure_buffer_unique) else len(set(failure_buffer))
                                ),
                                "failure_replay_quota": int(replay_quota),
                                "replay_gate_blocked": bool(replay_gate_blocked),
                                "replay_success_ema": float(replay_success_ema or 0.0),
                                "zero_credit_fallback_hits": int(zero_credit_fallback_hits),
                                "zero_credit_fallback_ratio": float(zero_credit_fallback_ratio),
                                "all_negative_guard_triggered": bool(all_negative_guard_triggered),
                                "reward_span": float(reward_span),
                                "global_max_abs_weight": float(global_max_abs_weight),
                                "optimizer_step_skipped": bool(skip_optimizer_step),
                                "step_wall_time": float(time.time() - step_start),
                                "rollout_wall_time": float(rollout_end - rollout_build_end),
                                "update_wall_time": float(update_end - rollout_end),
                                **loss_extra,
                            },
                        )
                    )
                    tracker.log_event(
                        "train",
                        "logged step metrics",
                        {"loss": float(loss_mean), "success_rate": float(success_mean)},
                    )

            need_eval = bool(args.eval_interval and step % args.eval_interval == 0)
            need_save = bool(args.save_interval and step % args.save_interval == 0)

            if dist_info.distributed and args.sync_eval_and_save and (need_eval or need_save):
                barrier(dist_info=dist_info)

            if tracker and need_eval:
                eval_res = _evaluate_pass_at_1(
                    model=model,
                    tokenizer=tokenizer,
                    env=env,
                    args=args,
                    eval_records=eval_records,
                    train_records=train_records,
                    sys_prompt=sys_prompt,
                    step=step,
                    max_eval_tasks=int(args.eval_tasks),
                )
                tracker.log_event(
                    "eval",
                    "eval checkpoint",
                    {"step": step, "pass_at_1": eval_res["pass_at_1"], "n": int(eval_res["n"])},
                )

            if tracker and need_save:
                ckpt_dir = Path(exp_dir) / "checkpoints" / f"step_{step}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                to_save = model.module if hasattr(model, "module") else model
                to_save.save_pretrained(str(ckpt_dir))
                tokenizer.save_pretrained(str(ckpt_dir))
                tracker.log_event("checkpoint", "saved checkpoint", {"step": step, "path": str(ckpt_dir)})

            if dist_info.distributed and args.sync_eval_and_save and (need_eval or need_save):
                barrier(dist_info=dist_info)

            if heartbeat and dist_info.is_rank0:
                total_step_time = time.time() - step_start
                print(
                    f"[Rank 0] Step {step} done in {total_step_time:.1f}s "
                    f"(samples={len(sampled_sequences)}, replay_quota={replay_quota}, "
                    f"replay_gate_blocked={int(replay_gate_blocked)})",
                    flush=True,
                )

        except Exception:
            print(
                f"[Rank {dist_info.rank}] Exception in training loop at step={step}:\n{traceback.format_exc()}",
                flush=True,
            )
            raise

    if tracker:
        tracker.finalize()

    barrier(dist_info=dist_info)

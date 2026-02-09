#!/usr/bin/env python3
"""
One-click data preparation for ContraDiff locomotion experiments.

What this script prepares:
1) Download raw D4RL datasets (triggered by env.get_dataset()).
2) Build value-of-states files under value_of_states/*.pkl.
3) Build dataset infos under dataset_infos/dataset_info_*.pkl.
4) Build mixed cluster infos under dataset_infos/cluster_infos_*.pkl.

Default scope matches ContraDiff paper setting:
- envs: halfcheetah, hopper, walker2d
- source datasets: expert, medium, medium-replay, random
- mixed ratios: 0.1, 0.2, 0.3 for {medium, medium-replay, random}
"""

from __future__ import annotations

import argparse
import json
import os
import pickle as pkl
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
CONTRADIFF_DIR = Path(os.environ.get("CONTRADIFF_DIR", str(PROJECT_ROOT / "contradiff"))).resolve()
MAIN_DIR = CONTRADIFF_DIR / "main"
if not MAIN_DIR.is_dir():
    raise SystemExit(
        "[ERR] CONTRADIFF_DIR/main does not exist.\n"
        f"CONTRADIFF_DIR={CONTRADIFF_DIR}\n"
        "Set CONTRADIFF_DIR to your ContraDiff repo path."
    )
sys.path.insert(0, str(MAIN_DIR))
sys.path.insert(0, str(CONTRADIFF_DIR))

np = None
cdist = None
MiniBatchKMeans = None
load_environment = None
sequence_dataset_mix = None
d4rl_offline_dataset = None
DATA_BACKEND = None
DOWNLOAD_RETRIES = 5
DOWNLOAD_RETRY_WAIT = 5


def _lazy_imports() -> None:
    global np, cdist, MiniBatchKMeans, load_environment, sequence_dataset_mix, d4rl_offline_dataset, DATA_BACKEND
    try:
        import numpy as _np  # type: ignore
        from scipy.spatial.distance import cdist as _cdist  # type: ignore
        from sklearn.cluster import MiniBatchKMeans as _MiniBatchKMeans  # type: ignore
    except Exception as e:
        py = sys.executable
        raise SystemExit(
            "[ERR] Missing dependencies for dataset prep.\n"
            "Please install: numpy, scipy, scikit-learn.\n"
            f"Try: {py} -m pip install numpy scipy scikit-learn\n"
            f"Original error: {type(e).__name__}: {e}"
        )

    _load_environment = None
    _sequence_dataset_mix = None
    _d4rl_offline_dataset = None
    backend = None
    d4rl_import_error = None
    just_d4rl_import_error = None

    try:
        from diffuser.datasets.d4rl import load_environment as __load_environment  # type: ignore
        from diffuser.datasets.d4rl import sequence_dataset_mix as __sequence_dataset_mix  # type: ignore

        _load_environment = __load_environment
        _sequence_dataset_mix = __sequence_dataset_mix
        backend = "d4rl"
    except Exception as e:
        d4rl_import_error = e

    if backend is None:
        try:
            from just_d4rl import d4rl_offline_dataset as __d4rl_offline_dataset  # type: ignore

            _d4rl_offline_dataset = __d4rl_offline_dataset
            backend = "just_d4rl"
        except Exception as e:
            just_d4rl_import_error = e

    if backend is None:
        py = sys.executable
        raise SystemExit(
            "[ERR] No supported D4RL backend found.\n"
            f"Option A (preferred): {py} -m pip install just-d4rl\n"
            "Option B: install d4rl + gym + mujoco runtime.\n"
            f"d4rl import error: {type(d4rl_import_error).__name__}: {d4rl_import_error}\n"
            f"just-d4rl import error: {type(just_d4rl_import_error).__name__}: {just_d4rl_import_error}"
        )

    np = _np
    cdist = _cdist
    MiniBatchKMeans = _MiniBatchKMeans
    load_environment = _load_environment
    sequence_dataset_mix = _sequence_dataset_mix
    d4rl_offline_dataset = _d4rl_offline_dataset
    DATA_BACKEND = backend


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_csv_arg(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_ratio_arg(value: str) -> List[float]:
    out: List[float] = []
    for cell in parse_csv_arg(value):
        out.append(float(cell))
    return out


def _max_episode_steps_for_env(env_full_name: str) -> int:
    # D4RL locomotion tasks used by ContraDiff are all length-1000.
    if env_full_name.startswith(("halfcheetah-", "hopper-", "walker2d-")):
        return 1000
    raise ValueError(f"Unsupported env for fallback backend: {env_full_name}")


def _normalize_tag(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())


def _find_cached_hdf5_files(env_full_name: str) -> List[Path]:
    cache_dir = Path(os.environ.get("D4RL_DATASET_DIR", str(Path.home() / ".d4rl" / "datasets"))).expanduser()
    if not cache_dir.exists():
        return []

    stem_tokens = [tok for tok in env_full_name.replace(".hdf5", "").split("-") if tok and tok != "v2"]
    token_norms = [_normalize_tag(tok) for tok in stem_tokens]

    candidates: List[Path] = []
    for fp in cache_dir.glob("*.hdf5"):
        stem_norm = _normalize_tag(fp.stem)
        if all(tok in stem_norm for tok in token_norms):
            candidates.append(fp)

    if candidates:
        return sorted(candidates)

    env_name = env_full_name.split("-")[0]
    return sorted(cache_dir.glob(f"{env_name}*.hdf5"))


def _clear_partial_cache_files(env_full_name: str) -> None:
    files = _find_cached_hdf5_files(env_full_name)
    if not files:
        print(f"[retry] no cached hdf5 found for {env_full_name}")
        return
    for fp in files:
        try:
            size = fp.stat().st_size
        except OSError:
            size = -1
        try:
            fp.unlink()
            if size >= 0:
                print(f"[retry] removed partial cache: {fp} ({size} bytes)")
            else:
                print(f"[retry] removed partial cache: {fp}")
        except OSError as e:
            print(f"[retry][warn] failed to remove {fp}: {e}")


def _dataset_like_to_dict(dataset_like) -> Dict[str, np.ndarray]:
    if isinstance(dataset_like, dict):
        out = {}
        for k, v in dataset_like.items():
            out[k] = np.asarray(v)
        return out

    # best effort: object with attributes
    keys = [k for k in dir(dataset_like) if not k.startswith("_")]
    out: Dict[str, np.ndarray] = {}
    for k in keys:
        try:
            v = getattr(dataset_like, k)
        except Exception:
            continue
        if callable(v):
            continue
        try:
            arr = np.asarray(v)
        except Exception:
            continue
        if arr.size == 0:
            continue
        out[k] = arr
    return out


def _sequence_dataset_mix_from_dataset_dict(dataset: Dict[str, np.ndarray], env_full_name: str) -> Tuple[List[Dict], Dict[str, np.ndarray]]:
    rewards = np.asarray(dataset["rewards"]).reshape(-1)
    terminals = np.asarray(dataset["terminals"]).reshape(-1)
    n = int(rewards.shape[0])
    max_episode_steps = _max_episode_steps_for_env(env_full_name)

    use_timeouts = "timeouts" in dataset
    timeouts = np.asarray(dataset.get("timeouts", np.zeros_like(terminals))).reshape(-1)

    data_ = {}
    all_data: List[Dict] = []
    episode_step = 0
    start = 0

    keys = [k for k in dataset.keys() if "metadata" not in k]
    for k in keys:
        data_[k] = []

    for i in range(n):
        done_bool = bool(terminals[i])
        if use_timeouts:
            final_timestep = bool(timeouts[i])
        else:
            final_timestep = episode_step == (max_episode_steps - 1)

        for k in keys:
            data_[k].append(dataset[k][i])

        if done_bool or final_timestep:
            end = i
            episode_data: Dict[str, np.ndarray] = {}
            for k in data_.keys():
                episode_data[k] = np.asarray(data_[k])
            episode_data["start"] = int(start)
            episode_data["end"] = int(end)
            episode_data["accumulated_reward"] = float(np.sum(episode_data["rewards"]))
            all_data.append(episode_data)
            data_ = {k: [] for k in keys}
            episode_step = 0
            start = end + 1
        else:
            episode_step += 1

    return all_data, dataset


def _load_trajs_and_dataset(env_full_name: str) -> Tuple[List[Dict], Dict[str, np.ndarray], int]:
    if DATA_BACKEND == "d4rl":
        env = load_environment(env_full_name)
        trajs, dataset = sequence_dataset_mix(env)
        dataset = _dataset_like_to_dict(dataset)
        return trajs, dataset, int(env.max_episode_steps)

    if DATA_BACKEND == "just_d4rl":
        attempts = max(1, int(DOWNLOAD_RETRIES))
        last_err = None
        for attempt in range(1, attempts + 1):
            try:
                dataset_like = d4rl_offline_dataset(env_full_name)
                break
            except Exception as e:
                last_err = e
                if e.__class__.__name__ == "ContentTooShortError" and attempt < attempts:
                    print(
                        f"[retry] partial download for {env_full_name} "
                        f"(attempt {attempt}/{attempts}); cleaning cache and retrying..."
                    )
                    _clear_partial_cache_files(env_full_name)
                    wait_s = max(0, int(DOWNLOAD_RETRY_WAIT)) * attempt
                    if wait_s > 0:
                        print(f"[retry] waiting {wait_s}s before retry")
                        time.sleep(wait_s)
                    continue
                raise
        else:
            raise RuntimeError(f"Failed to fetch {env_full_name}: {last_err}")

        dataset = _dataset_like_to_dict(dataset_like)
        trajs, dataset = _sequence_dataset_mix_from_dataset_dict(dataset, env_full_name)
        return trajs, dataset, _max_episode_steps_for_env(env_full_name)

    raise RuntimeError(f"Unknown DATA_BACKEND={DATA_BACKEND}")


def calculate_vs_like_contradiff(dataset: Dict, max_path_length: int) -> List[float]:
    """
    Keep behavior compatible with existing ContraDiff preprocessing.
    """
    vs_discount = 1e-4 ** (1 / max_path_length)
    vs_discounts = vs_discount ** np.arange(max_path_length)[:, None]

    rewards_raw = np.asarray(dataset["rewards"]).reshape(-1).astype(np.float32)
    terminals = np.asarray(dataset["terminals"]).reshape(-1)
    n = rewards_raw.shape[0]

    terminal_points = list(np.where(terminals)[0].astype(int))
    rewards = rewards_raw
    if not bool(terminals[-1]):
        rewards = np.concatenate((rewards, np.zeros(max_path_length, dtype=np.float32)), axis=-1)
    terminal_points.append(n + max_path_length)

    values: List[float] = []
    for i in range(n):
        if i > terminal_points[0]:
            terminal_points = terminal_points[1:]
        end = min(i + max_path_length, terminal_points[0])
        rewards_si = rewards[i:end]
        padding_length = max_path_length - len(rewards_si)
        if padding_length > 0:
            padding = np.ones(padding_length, dtype=np.float32) * -1
            rewards_si = np.concatenate((rewards_si, padding), axis=-1)
        value_si = np.sum(vs_discounts * rewards_si)
        values.append(float(value_si))
    return values


def build_value_file(env_full_name: str, value_dir: Path, *, skip_existing: bool) -> Path:
    out_file = value_dir / f"{env_full_name}.pkl"
    if skip_existing and out_file.exists():
        print(f"[skip] {out_file}")
        return out_file

    print(f"[value] building {env_full_name}")
    _, dataset, max_path_length = _load_trajs_and_dataset(env_full_name)
    values = calculate_vs_like_contradiff(dataset, max_path_length=max_path_length)

    with out_file.open("wb") as f:
        pkl.dump(values, f)
    print(f"[done] {out_file}")
    return out_file


def build_dataset_info(env_short: str, dataset_name: str, dataset_infos_dir: Path, value_dir: Path, *, skip_existing: bool) -> Path:
    env_full_name = f"{env_short}-{dataset_name}-v2"
    out_file = dataset_infos_dir / f"dataset_info_{env_full_name}.pkl"
    if skip_existing and out_file.exists():
        print(f"[skip] {out_file}")
        return out_file

    value_file = value_dir / f"{env_full_name}.pkl"
    if not value_file.exists():
        raise FileNotFoundError(f"Missing value file: {value_file}")

    print(f"[dataset_info] building {env_full_name}")
    trajs, dataset, _ = _load_trajs_and_dataset(env_full_name)

    with value_file.open("rb") as f:
        values_of_states = pkl.load(f)

    accumulated_rewards = [float(np.sum(traj["rewards"])) for traj in trajs]
    high_to_low = np.argsort(np.asarray(accumulated_rewards))[::-1]

    payload = {
        "dataset": dataset,
        "trajs": trajs,
        "accumulated_rewards": accumulated_rewards,
        "high_to_low": high_to_low,
        "vlaues_of_states": values_of_states,  # keep original key typo for compatibility
    }
    with out_file.open("wb") as f:
        pkl.dump(payload, f)
    print(f"[done] {out_file}")
    return out_file


@dataclass
class ClusterBuildConfig:
    env_short: str
    dataset_name: str
    ratio: float
    metrics: str
    max_iter: int
    seed: int


def _trim_original_states_if_needed(orig_trajs: List[Dict], orig_states: np.ndarray, orig_vs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    orig_states_in_trajs = sum(len(traj["observations"]) for traj in orig_trajs)
    if len(orig_states) != orig_states_in_trajs:
        print(
            "[warn] inconsistent raw state count; trimming to trajectory-covered states: "
            f"{len(orig_states)} -> {orig_states_in_trajs}"
        )
        orig_states = orig_states[:orig_states_in_trajs]
        orig_vs = orig_vs[:orig_states_in_trajs]
    return orig_states, orig_vs


def build_cluster_info(cfg: ClusterBuildConfig, dataset_infos_dir: Path, *, skip_existing: bool) -> Path:
    ratio_str = f"{cfg.ratio:.2f}"
    out_file = dataset_infos_dir / f"cluster_infos_{cfg.env_short}-{cfg.dataset_name}-{ratio_str}-v2.pkl"
    if skip_existing and out_file.exists():
        print(f"[skip] {out_file}")
        return out_file

    expert_file = dataset_infos_dir / f"dataset_info_{cfg.env_short}-expert-v2.pkl"
    orig_file = dataset_infos_dir / f"dataset_info_{cfg.env_short}-{cfg.dataset_name}-v2.pkl"
    if not expert_file.exists():
        raise FileNotFoundError(f"Missing expert dataset info: {expert_file}")
    if not orig_file.exists():
        raise FileNotFoundError(f"Missing source dataset info: {orig_file}")

    print(f"[cluster] building {cfg.env_short}-{cfg.dataset_name} ratio={ratio_str}")
    with expert_file.open("rb") as f:
        expert_infos = pkl.load(f)
    with orig_file.open("rb") as f:
        original_infos = pkl.load(f)

    orig_env_full_name = f"{cfg.env_short}-{cfg.dataset_name}-v2"
    orig_trajs, _, _ = _load_trajs_and_dataset(orig_env_full_name)

    orig_states = np.asarray(original_infos["dataset"]["observations"], dtype=np.float32)
    orig_vs = np.asarray(original_infos["vlaues_of_states"], dtype=np.float32)
    orig_states, orig_vs = _trim_original_states_if_needed(orig_trajs, orig_states, orig_vs)

    nums_to_mixin = int(len(orig_states) * cfg.ratio)
    mixin_idxs: List[int] = []
    mixin_lengths: List[int] = []
    mixed_trajectories = list(orig_trajs)
    for idx in expert_infos["high_to_low"]:
        idx_int = int(idx)
        length = len(expert_infos["trajs"][idx_int]["observations"])
        mixin_idxs.append(idx_int)
        mixin_lengths.append(length)
        mixed_trajectories.append(expert_infos["trajs"][idx_int])
        if int(np.sum(mixin_lengths)) >= nums_to_mixin:
            break

    tomix_states: List[np.ndarray] = []
    tomix_vs: List[float] = []
    tomix_state_idxs: List[int] = []
    expert_vs = np.asarray(expert_infos["vlaues_of_states"])
    for idx_int in mixin_idxs:
        traj = expert_infos["trajs"][idx_int]
        obs = np.asarray(traj["observations"])
        s = int(traj["start"])
        e = int(traj["end"])
        vals = np.asarray(expert_vs[s : e + 1])
        n = min(len(obs), len(vals))
        if n <= 0:
            continue
        tomix_states.extend(list(obs[:n]))
        tomix_vs.extend(list(vals[:n]))
        tomix_state_idxs.extend(list(range(s, s + n)))

    tomix_states_np = np.asarray(tomix_states, dtype=np.float32) if tomix_states else np.zeros((0, orig_states.shape[1]), dtype=np.float32)
    tomix_vs_np = np.asarray(tomix_vs, dtype=np.float32)

    mixed_states = np.concatenate((orig_states, tomix_states_np), axis=0) if len(tomix_states_np) > 0 else np.asarray(orig_states)
    mixed_vs = np.concatenate((orig_vs, tomix_vs_np), axis=0) if len(tomix_vs_np) > 0 else np.asarray(orig_vs)

    do_not_select_me: List[int] = []
    cursor = -1
    for traj in mixed_trajectories:
        cursor += len(traj["observations"])
        do_not_select_me.append(int(cursor))

    nums_clusters = max(2, int(np.sqrt(len(mixed_states))))
    cluster = MiniBatchKMeans(
        n_clusters=nums_clusters,
        batch_size=10240,
        max_iter=cfg.max_iter,
        verbose=0,
        random_state=cfg.seed,
        n_init="auto",
    )
    results = cluster.fit_predict(mixed_states)

    samples_per_cluster_idx: List[np.ndarray] = []
    for c in range(nums_clusters):
        samples_per_cluster_idx.append(np.where(results == c)[0].astype(np.int64))

    results_pure = np.array(results, copy=True)
    for idx in do_not_select_me:
        if 0 <= idx < len(results_pure):
            results_pure[idx] = nums_clusters
    samples_per_cluster_idx_pure: List[np.ndarray] = []
    for c in range(nums_clusters + 1):
        samples_per_cluster_idx_pure.append(np.where(results_pure == c)[0].astype(np.int64))

    tomix_state_idxs_per_cluster: List[List[int]] = [[] for _ in range(nums_clusters)]
    if len(tomix_states_np) > 0:
        for local_idx, c in enumerate(results[len(orig_states) :]):
            tomix_state_idxs_per_cluster[int(c)].append(int(local_idx))

    centers = cluster.cluster_centers_.astype(np.float32)
    similarity_of_clusters = cdist(centers, centers, metric=cfg.metrics).astype(np.float32)

    payload = {
        "cluster_of_samples": results,
        "num_actual_mixed_states": int(len(tomix_states_np)),
        "num_trajectories_to_mixin": int(nums_to_mixin),
        "origional_states": orig_states,
        "origional_vs": orig_vs,
        "tomix_state": tomix_states_np,
        "tomix_state_idxs_per_cluster": tomix_state_idxs_per_cluster,
        "samples_per_cluster_idx_pure": samples_per_cluster_idx_pure,
        "num_samples_of_clusters_pure": [len(x) for x in samples_per_cluster_idx_pure],
        "tomix_state_idxs": tomix_state_idxs,
        "tomix_state_count": int(len(tomix_states_np)),
        "tomix_vs": tomix_vs_np,
        "positive_hash_table": "Please Compute With Run. File too large.",
        "cluster_representation_centro": centers,
        "nums_clusters": int(nums_clusters),
        "num_mixed_samples": int(len(mixed_states)),
        "num_origional_states": int(len(orig_states)),
        "ratio": float(cfg.ratio),
        "num_samples_of_clusters": [len(x) for x in samples_per_cluster_idx],
        "samples_per_cluster_idx": samples_per_cluster_idx,
        "similarity_in_cluster": "Please Compute With Run. File too large.",
        "similarity_of_clusters": similarity_of_clusters,
        "similarity_sample_to_clusters": "Please Compute With Run. File too large.",
        "mixed_do_not_select_me": do_not_select_me,
        "mixed_states": mixed_states,
        "mixed_vs": mixed_vs,
        "mixed_trajectories": mixed_trajectories,
        "generator": "download_locomotion_datasets.py",
    }
    with out_file.open("wb") as f:
        pkl.dump(payload, f)
    print(f"[done] {out_file}")
    return out_file


def main() -> None:
    global DOWNLOAD_RETRIES, DOWNLOAD_RETRY_WAIT
    parser = argparse.ArgumentParser(description="One-click ContraDiff locomotion dataset preparation")
    parser.add_argument("--envs", type=str, default="halfcheetah,hopper,walker2d")
    parser.add_argument("--datasets", type=str, default="expert,medium,medium-replay,random")
    parser.add_argument("--mix-datasets", type=str, default="medium,medium-replay,random")
    parser.add_argument("--ratios", type=str, default="0.1,0.2,0.3")
    parser.add_argument("--metrics", type=str, default="canberra")
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--dataset-infos-dir", type=str, default=str(MAIN_DIR / "dataset_infos"))
    parser.add_argument("--value-dir", type=str, default=str(MAIN_DIR / "value_of_states"))
    parser.add_argument("--download-retries", type=int, default=5)
    parser.add_argument("--download-retry-wait", type=int, default=5)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    _lazy_imports()
    DOWNLOAD_RETRIES = int(args.download_retries)
    DOWNLOAD_RETRY_WAIT = int(args.download_retry_wait)

    envs = parse_csv_arg(args.envs)
    datasets = parse_csv_arg(args.datasets)
    mix_datasets = parse_csv_arg(args.mix_datasets)
    ratios = parse_ratio_arg(args.ratios)

    random.seed(args.seed)
    np.random.seed(args.seed)

    dataset_infos_dir = Path(args.dataset_infos_dir).resolve()
    value_dir = Path(args.value_dir).resolve()
    ensure_dir(dataset_infos_dir)
    ensure_dir(value_dir)

    print("=== ContraDiff locomotion data prep ===")
    print(json.dumps(
        {
            "backend": DATA_BACKEND,
            "envs": envs,
            "datasets": datasets,
            "mix_datasets": mix_datasets,
            "ratios": ratios,
            "metrics": args.metrics,
            "max_iter": args.max_iter,
            "dataset_infos_dir": str(dataset_infos_dir),
            "value_dir": str(value_dir),
            "download_retries": int(DOWNLOAD_RETRIES),
            "download_retry_wait": int(DOWNLOAD_RETRY_WAIT),
            "skip_existing": bool(args.skip_existing),
        },
        indent=2,
        ensure_ascii=False,
    ))

    # 1) Raw D4RL + values + dataset infos
    for env_short in envs:
        for dataset_name in datasets:
            env_full_name = f"{env_short}-{dataset_name}-v2"
            build_value_file(env_full_name, value_dir, skip_existing=args.skip_existing)
            build_dataset_info(
                env_short=env_short,
                dataset_name=dataset_name,
                dataset_infos_dir=dataset_infos_dir,
                value_dir=value_dir,
                skip_existing=args.skip_existing,
            )

    # 2) Mixed cluster infos for sub-optimal settings
    for env_short in envs:
        for dataset_name in mix_datasets:
            for ratio in ratios:
                cfg = ClusterBuildConfig(
                    env_short=env_short,
                    dataset_name=dataset_name,
                    ratio=float(ratio),
                    metrics=args.metrics,
                    max_iter=int(args.max_iter),
                    seed=int(args.seed),
                )
                build_cluster_info(cfg, dataset_infos_dir=dataset_infos_dir, skip_existing=args.skip_existing)

    print("=== Data prep complete ===")


if __name__ == "__main__":
    main()

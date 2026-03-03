#!/usr/bin/env python3
"""
Prepare ContraDiff paper non-locomotion datasets (maze2d / kitchen).

Outputs:
1) value_of_states/<dataset>.pkl
2) dataset_infos/dataset_info_<dataset>.pkl
3) dataset_infos/cluster_infos_<dataset>.pkl
"""

from __future__ import annotations

import argparse
import collections
import copy
import json
import os
import pickle as pkl
import sys
import time
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
d4rl_offline_dataset = None

DOWNLOAD_RETRIES = 10
DOWNLOAD_RETRY_WAIT = 10


def _lazy_imports() -> None:
    global np, cdist, MiniBatchKMeans, d4rl_offline_dataset
    try:
        import numpy as _np  # type: ignore
        from scipy.spatial.distance import cdist as _cdist  # type: ignore
        from sklearn.cluster import MiniBatchKMeans as _MiniBatchKMeans  # type: ignore
    except Exception as e:
        py = sys.executable
        raise SystemExit(
            "[ERR] Missing dependencies.\n"
            f"Install with: {py} -m pip install numpy scipy scikit-learn\n"
            f"Original error: {type(e).__name__}: {e}"
        )

    try:
        from just_d4rl import d4rl_offline_dataset as _d4rl_offline_dataset  # type: ignore
    except Exception as e:
        py = sys.executable
        raise SystemExit(
            "[ERR] just_d4rl is required for maze2d/kitchen data prep.\n"
            f"Install with: {py} -m pip install just-d4rl\n"
            f"Original error: {type(e).__name__}: {e}"
        )

    np = _np
    cdist = _cdist
    MiniBatchKMeans = _MiniBatchKMeans
    d4rl_offline_dataset = _d4rl_offline_dataset


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_csv_arg(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def _normalize_tag(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())


def _find_cached_hdf5_files(dataset_name: str) -> List[Path]:
    cache_dir = Path(os.environ.get("D4RL_DATASET_DIR", str(Path.home() / ".d4rl" / "datasets"))).expanduser()
    if not cache_dir.exists():
        return []

    stem_tokens = [tok for tok in dataset_name.replace(".hdf5", "").split("-") if tok and not tok.startswith("v")]
    token_norms = [_normalize_tag(tok) for tok in stem_tokens]

    candidates: List[Path] = []
    for fp in cache_dir.glob("*.hdf5"):
        stem_norm = _normalize_tag(fp.stem)
        if all(tok in stem_norm for tok in token_norms):
            candidates.append(fp)
    return sorted(candidates)


def _clear_partial_cache_files(dataset_name: str) -> None:
    files = _find_cached_hdf5_files(dataset_name)
    if not files:
        print(f"[retry] no cached hdf5 found for {dataset_name}")
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


def _can_retry_download_error(exc: Exception) -> bool:
    name = type(exc).__name__
    text = str(exc).lower()
    if name == "ContentTooShortError":
        return True
    if name == "OSError":
        if "truncated file" in text or "unable to synchronously open file" in text:
            return True
    return False


def _dataset_like_to_dict(dataset_like) -> Dict[str, np.ndarray]:
    if isinstance(dataset_like, dict):
        return {k: np.asarray(v) for k, v in dataset_like.items()}

    out: Dict[str, np.ndarray] = {}
    for k in dir(dataset_like):
        if k.startswith("_"):
            continue
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


def _load_dataset_with_retry(dataset_name: str, retries: int, wait_s: int) -> Dict[str, np.ndarray]:
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            data = d4rl_offline_dataset(dataset_name)
            dataset = _dataset_like_to_dict(data)
            if "observations" not in dataset or "rewards" not in dataset:
                raise RuntimeError(f"dataset {dataset_name} is missing required keys")
            return dataset
        except Exception as e:  # noqa: BLE001
            last_exc = e
            if attempt >= retries or not _can_retry_download_error(e):
                break
            print(
                f"[retry] partial/corrupt cache for {dataset_name} "
                f"(attempt {attempt}/{retries}); cleaning cache and retrying..."
            )
            _clear_partial_cache_files(dataset_name)
            print(f"[retry] waiting {wait_s}s before retry")
            time.sleep(wait_s)

    assert last_exc is not None
    raise RuntimeError(f"Failed to load dataset {dataset_name}: {type(last_exc).__name__}: {last_exc}") from last_exc


def _process_maze2d_episode(episode: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    if "observations" not in episode:
        return episode
    length = len(episode["observations"])
    if length <= 1:
        return episode
    next_observations = episode["observations"][1:].copy()
    for key, val in list(episode.items()):
        episode[key] = val[:-1]
    episode["next_observations"] = next_observations
    return episode


def _split_episodes(dataset_name: str, dataset: Dict[str, np.ndarray]) -> List[Dict[str, np.ndarray]]:
    rewards = np.asarray(dataset["rewards"]).reshape(-1)
    n = int(rewards.shape[0])
    terminals = np.asarray(dataset.get("terminals", np.zeros(n, dtype=np.bool_))).reshape(-1).astype(bool)
    use_timeouts = "timeouts" in dataset
    timeouts = np.asarray(dataset.get("timeouts", np.zeros(n, dtype=np.bool_))).reshape(-1).astype(bool)
    is_kitchen = dataset_name.startswith("kitchen-")
    fallback_max_steps = _known_env_horizon(dataset_name)

    data_ = collections.defaultdict(list)
    all_data: List[Dict[str, np.ndarray]] = []
    start = 0
    episode_step = 0

    keys = [k for k in dataset.keys() if "metadata" not in k]
    for i in range(n):
        done_bool = bool(terminals[i])
        if is_kitchen:
            # Keep old kitchen path behavior: split by terminals only.
            final_timestep = False
        elif use_timeouts:
            final_timestep = bool(timeouts[i])
        else:
            # Keep old D4RL fallback behavior when timeout field is unavailable.
            final_timestep = episode_step == fallback_max_steps - 1

        for k in keys:
            data_[k].append(dataset[k][i])

        if done_bool or final_timestep:
            episode = {k: np.asarray(v) for k, v in data_.items()}
            if "maze2d" in dataset_name:
                episode = _process_maze2d_episode(episode)
            if len(episode.get("observations", [])) > 0:
                episode["start"] = start
                episode["end"] = i
                episode["accumulated_reward"] = float(np.sum(episode.get("rewards", np.zeros(1))))
                all_data.append(episode)
            start = i + 1
            episode_step = 0
            data_ = collections.defaultdict(list)

        episode_step += 1

    if data_:
        episode = {k: np.asarray(v) for k, v in data_.items()}
        if "maze2d" in dataset_name:
            episode = _process_maze2d_episode(episode)
        if len(episode.get("observations", [])) > 0:
            episode["start"] = start
            episode["end"] = n - 1
            episode["accumulated_reward"] = float(np.sum(episode.get("rewards", np.zeros(1))))
            all_data.append(episode)

    return all_data


def _known_env_horizon(dataset_name: str) -> int:
    if dataset_name.startswith("kitchen-"):
        return 280
    if dataset_name == "maze2d-umaze-v1":
        return 300
    if dataset_name == "maze2d-medium-v1":
        return 600
    if dataset_name == "maze2d-large-v1":
        return 800
    return 1000


def _infer_horizon_from_timeouts(dataset: Dict[str, np.ndarray], default_horizon: int) -> int:
    if "timeouts" not in dataset:
        return default_horizon
    terminals = np.asarray(dataset.get("terminals", np.zeros_like(dataset["timeouts"]))).reshape(-1).astype(bool)
    timeouts = np.asarray(dataset["timeouts"]).reshape(-1).astype(bool)
    n = int(timeouts.shape[0])
    lengths: List[int] = []
    cur = 0
    for i in range(n):
        cur += 1
        if bool(timeouts[i]) or bool(terminals[i]):
            lengths.append(cur)
            cur = 0
    if cur > 0:
        lengths.append(cur)
    if not lengths:
        return default_horizon
    return int(max(lengths))


def _calculate_values_constant_pad(
    rewards_raw: np.ndarray,
    terminals: np.ndarray,
    horizon: int,
    discount_sum: float,
    pad_const: float,
) -> List[float]:
    n = int(rewards_raw.shape[0])
    if n == 0:
        return []
    rewards = rewards_raw.astype(np.float64)
    if not bool(terminals[-1]):
        rewards = np.concatenate((rewards, np.zeros(horizon, dtype=np.float64)), axis=0)
    prefix = np.concatenate(([0.0], np.cumsum(rewards, dtype=np.float64)), axis=0)

    terminal_points = list(np.where(terminals)[0].astype(np.int64))
    terminal_points.append(n + horizon)
    values = np.empty(n, dtype=np.float64)

    seg_start = 0
    for terminal_point in terminal_points:
        seg_end = min(n - 1, int(terminal_point))
        if seg_start > seg_end:
            seg_start = int(terminal_point) + 1
            continue
        idx = np.arange(seg_start, seg_end + 1, dtype=np.int64)
        end = np.minimum(idx + horizon, int(terminal_point))
        sums = prefix[end] - prefix[idx]
        padding = horizon - (end - idx)
        values[idx] = discount_sum * (sums + padding * pad_const)
        seg_start = int(terminal_point) + 1
        if seg_start >= n:
            break

    return values.astype(np.float32).tolist()


def _calculate_values_pad_with_end_reward(
    rewards_raw: np.ndarray,
    terminals: np.ndarray,
    horizon: int,
    discount_sum: float,
) -> List[float]:
    n = int(rewards_raw.shape[0])
    if n == 0:
        return []
    rewards = rewards_raw.astype(np.float64)
    if not bool(terminals[-1]):
        pad_tail = np.ones(horizon, dtype=np.float64) * rewards[-1]
        rewards = np.concatenate((rewards, pad_tail), axis=0)
    prefix = np.concatenate(([0.0], np.cumsum(rewards, dtype=np.float64)), axis=0)

    terminal_points = list(np.where(terminals)[0].astype(np.int64))
    terminal_points.append(n + horizon)
    values = np.empty(n, dtype=np.float64)

    seg_start = 0
    for terminal_point in terminal_points:
        seg_end = min(n - 1, int(terminal_point))
        if seg_start > seg_end:
            seg_start = int(terminal_point) + 1
            continue
        idx = np.arange(seg_start, seg_end + 1, dtype=np.int64)
        end = np.minimum(idx + horizon, int(terminal_point))
        sums = prefix[end] - prefix[idx]
        padding = horizon - (end - idx)
        pad_vals = rewards[end]
        values[idx] = discount_sum * (sums + padding * pad_vals)
        seg_start = int(terminal_point) + 1
        if seg_start >= n:
            break

    return values.astype(np.float32).tolist()


def build_value_file(
    dataset_name: str,
    dataset: Dict[str, np.ndarray],
    trajs: List[Dict[str, np.ndarray]],
    value_dir: Path,
    *,
    skip_existing: bool,
) -> Tuple[Path, List[float]]:
    out_file = value_dir / f"{dataset_name}.pkl"
    if skip_existing and out_file.exists():
        print(f"[skip] {out_file}")
        with out_file.open("rb") as f:
            return out_file, pkl.load(f)

    print(f"[value] building {dataset_name}")
    rewards_raw = np.asarray(dataset["rewards"]).reshape(-1).astype(np.float32)
    terminals = np.asarray(dataset.get("terminals", np.zeros_like(rewards_raw))).reshape(-1).astype(bool)

    known_horizon = _known_env_horizon(dataset_name)
    env_horizon = _infer_horizon_from_timeouts(dataset, known_horizon)
    traj_max_len = max((len(traj.get("observations", [])) for traj in trajs), default=env_horizon)
    traj_max_len = max(1, int(traj_max_len))

    if dataset_name.startswith("kitchen-"):
        discount = float(pow(1e-3, 1 / max(1, env_horizon)))
        discount_sum = float(np.sum(discount ** np.arange(traj_max_len)))
        values = _calculate_values_pad_with_end_reward(rewards_raw, terminals, traj_max_len, discount_sum)
    else:
        # maze2d follows same value discount style as locomotion code path.
        discount = float(pow(1e-4, 1 / max(1, env_horizon)))
        discount_sum = float(np.sum(discount ** np.arange(env_horizon)))
        values = _calculate_values_constant_pad(rewards_raw, terminals, env_horizon, discount_sum, pad_const=-1.0)

    with out_file.open("wb") as f:
        pkl.dump(values, f)
    print(f"[done] {out_file}")
    return out_file, values


def build_dataset_info(
    dataset_name: str,
    dataset: Dict[str, np.ndarray],
    trajs: List[Dict[str, np.ndarray]],
    values: Sequence[float],
    dataset_infos_dir: Path,
    *,
    skip_existing: bool,
) -> Path:
    out_file = dataset_infos_dir / f"dataset_info_{dataset_name}.pkl"
    if skip_existing and out_file.exists():
        print(f"[skip] {out_file}")
        return out_file

    print(f"[dataset_info] building {dataset_name}")
    accumulated_rewards = [float(np.sum(traj.get("rewards", np.zeros(1)))) for traj in trajs]
    high_to_low = np.argsort(np.asarray(accumulated_rewards))[::-1]

    payload = {
        "dataset": dataset,
        "trajs": trajs,
        "accumulated_rewards": accumulated_rewards,
        "high_to_low": high_to_low,
        "vlaues_of_states": list(values),  # keep historical key typo
    }
    with out_file.open("wb") as f:
        pkl.dump(payload, f)
    print(f"[done] {out_file}")
    return out_file


def build_cluster_info_nomix(
    dataset_name: str,
    dataset: Dict[str, np.ndarray],
    trajs: List[Dict[str, np.ndarray]],
    values: Sequence[float],
    dataset_infos_dir: Path,
    *,
    metrics: str,
    max_iter: int,
    seed: int,
    skip_existing: bool,
) -> Path:
    out_file = dataset_infos_dir / f"cluster_infos_{dataset_name}.pkl"
    if skip_existing and out_file.exists():
        print(f"[skip] {out_file}")
        return out_file

    print(f"[cluster] building {dataset_name}")
    original_states = np.asarray(dataset["observations"], dtype=np.float32)
    original_vs = np.asarray(values, dtype=np.float32)
    states_in_trajs = int(sum(len(traj.get("observations", [])) for traj in trajs))

    if len(original_states) != len(original_vs):
        min_len = min(len(original_states), len(original_vs))
        print(f"[warn] value/state length mismatch; trimming {len(original_states)}/{len(original_vs)} -> {min_len}")
        original_states = original_states[:min_len]
        original_vs = original_vs[:min_len]
    if len(original_states) != states_in_trajs:
        print(
            "[warn] inconsistent raw state count; trimming to trajectory-covered states: "
            f"{len(original_states)} -> {states_in_trajs}"
        )
        original_states = original_states[:states_in_trajs]
        original_vs = original_vs[:states_in_trajs]

    do_not_select_me: List[int] = []
    offset = -1
    for traj in trajs:
        offset += len(traj.get("observations", []))
        do_not_select_me.append(offset)
    do_not_select_me = [x for x in do_not_select_me if 0 <= x < len(original_states)]

    nums_clusters = max(2, int(np.sqrt(max(1, len(original_states)))))
    cluster = MiniBatchKMeans(
        n_clusters=nums_clusters,
        batch_size=10240,
        max_iter=max_iter,
        random_state=seed,
        verbose=0,
    )
    results = cluster.fit_predict(original_states).astype(np.int64)

    samples_per_cluster_idx: List[np.ndarray] = []
    for c in range(nums_clusters):
        samples_per_cluster_idx.append(np.where(results == c)[0].astype(np.int64))

    results_pure = copy.deepcopy(results)
    if do_not_select_me:
        results_pure[np.asarray(do_not_select_me, dtype=np.int64)] = nums_clusters
    samples_per_cluster_idx_pure: List[np.ndarray] = []
    for c in range(nums_clusters + 1):
        samples_per_cluster_idx_pure.append(np.where(results_pure == c)[0].astype(np.int64))

    centers = cluster.cluster_centers_.astype(np.float32)
    similarity_of_clusters = cdist(centers, centers, metric=metrics).astype(np.float32)

    metadata = {
        "cluster_of_samples": results,
        "num_actual_mixed_states": "invalid",
        "num_trajectories_to_mixin": "Invalid",
        "origional_states": original_states,
        "origional_vs": original_vs,
        "tomix_state": "Invalid",
        "tomix_state_idxs_per_cluster": "Invalid",
        "samples_per_cluster_idx_pure": samples_per_cluster_idx_pure,
        "num_samples_of_clusters_pure": [len(x) for x in samples_per_cluster_idx_pure],
        "tomix_state_idxs": "Invalid",
        "tomix_state_count": 0,
        "tomix_vs": [],
        "positive_hash_table": "Please Compute With Run. File too large.",
        "cluster_representation_centro": centers,
        "nums_clusters": int(nums_clusters),
        "num_mixed_samples": len(trajs),
        "num_origional_states": len(original_states),
        "ratio": "Invalid",
        "num_samples_of_clusters": [len(x) for x in samples_per_cluster_idx],
        "samples_per_cluster_idx": samples_per_cluster_idx,
        "similarity_in_cluster": "Please Compute With Run. File too large.",
        "similarity_of_clusters": similarity_of_clusters,
        "similarity_sample_to_clusters": "Please Compute With Run. File too large.",
        "mixed_do_not_select_me": do_not_select_me,
        "mixed_states": original_states,
        "mixed_vs": original_vs,
        "mixed_trajectories": trajs,
        "generator": "download_maze_kitchen_datasets.py",
    }

    with out_file.open("wb") as f:
        pkl.dump(metadata, f)
    print(f"[done] {out_file}")
    return out_file


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        type=str,
        default="maze2d-umaze-v1,maze2d-medium-v1,maze2d-large-v1,kitchen-mixed-v0,kitchen-partial-v0,kitchen-complete-v0",
    )
    parser.add_argument("--metrics", type=str, default="canberra")
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--dataset-infos-dir", type=str, default=str(MAIN_DIR / "dataset_infos"))
    parser.add_argument("--value-dir", type=str, default=str(MAIN_DIR / "value_of_states"))
    parser.add_argument("--download-retries", type=int, default=10)
    parser.add_argument("--download-retry-wait", type=int, default=10)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    _lazy_imports()
    global DOWNLOAD_RETRIES, DOWNLOAD_RETRY_WAIT
    DOWNLOAD_RETRIES = max(1, int(args.download_retries))
    DOWNLOAD_RETRY_WAIT = max(0, int(args.download_retry_wait))

    dataset_infos_dir = Path(args.dataset_infos_dir).resolve()
    value_dir = Path(args.value_dir).resolve()
    ensure_dir(dataset_infos_dir)
    ensure_dir(value_dir)

    datasets = parse_csv_arg(args.datasets)
    if not datasets:
        raise SystemExit("[ERR] No dataset specified.")

    print("=== ContraDiff maze/kitchen data prep ===")
    print(
        json.dumps(
            {
                "datasets": datasets,
                "metrics": args.metrics,
                "max_iter": args.max_iter,
                "seed": args.seed,
                "dataset_infos_dir": str(dataset_infos_dir),
                "value_dir": str(value_dir),
                "download_retries": DOWNLOAD_RETRIES,
                "download_retry_wait": DOWNLOAD_RETRY_WAIT,
                "skip_existing": bool(args.skip_existing),
            },
            indent=2,
        )
    )

    for name in datasets:
        if not (name.startswith("maze2d-") or name.startswith("kitchen-")):
            raise SystemExit(f"[ERR] Unsupported non-locomotion dataset: {name}")

        need_value = not (args.skip_existing and (value_dir / f"{name}.pkl").exists())
        need_info = not (args.skip_existing and (dataset_infos_dir / f"dataset_info_{name}.pkl").exists())
        need_cluster = not (args.skip_existing and (dataset_infos_dir / f"cluster_infos_{name}.pkl").exists())
        if not (need_value or need_info or need_cluster):
            print(f"[skip] all outputs exist for {name}")
            continue

        dataset = _load_dataset_with_retry(name, DOWNLOAD_RETRIES, DOWNLOAD_RETRY_WAIT)
        trajs = _split_episodes(name, dataset)

        _, values = build_value_file(name, dataset, trajs, value_dir, skip_existing=args.skip_existing)
        build_dataset_info(name, dataset, trajs, values, dataset_infos_dir, skip_existing=args.skip_existing)
        build_cluster_info_nomix(
            name,
            dataset,
            trajs,
            values,
            dataset_infos_dir,
            metrics=args.metrics,
            max_iter=args.max_iter,
            seed=args.seed,
            skip_existing=args.skip_existing,
        )

    print("=== Data prep complete ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

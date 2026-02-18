# ContraDiff Dual-Track Quickstart

## 1) Single-GPU training script

Use `/Users/zhaoxu/Developer/projects/neurips2026_plans/scripts/slurm/run_contradiff_ideaA_1gpu.sh` to avoid requesting 4 GPUs for single-process jobs.

## 2) Submit two tracks in parallel

Run from repo root:

```bash
bash /Users/zhaoxu/Developer/projects/neurips2026_plans/scripts/slurm/submit_contradiff_dual_tracks.sh
```

Default behavior:
- `Track A` (always on): aarch64-compatible path (`USE_JUST_D4RL_BACKEND=1`, proxy eval).
- `Track B` (optional): standard online protocol (`STRICT_REAL_EVAL=1`, full d4rl+mujoco_py runtime).

## 3) Common overrides

```bash
DATASET=hopper-random-v2 \
EXPERT_RATIO=0.2 \
MAX_STEPS=200000 \
BATCH_SIZE=128 \
COUNTERFACTUAL_K=8 \
ENABLE_TRACK_B=1 \
TRACK_B_CONDA_ENV=$HOME/.conda/envs/rlvr_full \
bash /Users/zhaoxu/Developer/projects/neurips2026_plans/scripts/slurm/submit_contradiff_dual_tracks.sh
```

## 4) Dry-run (print sbatch commands only)

```bash
DRY_RUN=1 ENABLE_TRACK_B=1 TRACK_B_PYTHON_BIN=/usr/bin/python3 \
bash /Users/zhaoxu/Developer/projects/neurips2026_plans/scripts/slurm/submit_contradiff_dual_tracks.sh
```

## 5) Scan logs quickly

```bash
bash /Users/zhaoxu/Developer/projects/neurips2026_plans/scripts/slurm/scan_contradiff_logs.sh
```


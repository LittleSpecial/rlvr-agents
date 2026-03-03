#!/bin/bash
# CACDP (Criticality-Aware Contrastive Diffusion Planning) wrapper
# Uses contradiff_ideaA_common.sh with CACDP-specific defaults
#
# Usage:
#   DATASET=hopper-medium-v2 sbatch run_cacdp.sh
#   DATASET=walker2d-medium-v2 SEED=2000 sbatch run_cacdp.sh

#SBATCH --job-name=cacdp
#SBATCH --partition=gpu
#SBATCH --gpus=4
#SBATCH --cpus-per-task=128
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/cacdp_%j.out
#SBATCH --error=logs/cacdp_%j.err

set -eo pipefail

# ---- CACDP-specific defaults ----
export BRANCH="${BRANCH:-plan9b_cacdp}"
export USE_CRITICALITY_PREDICTOR="${USE_CRITICALITY_PREDICTOR:-1}"
export CRITICALITY_HIDDEN_DIM="${CRITICALITY_HIDDEN_DIM:-128}"
export CRITICALITY_LOSS_WEIGHT="${CRITICALITY_LOSS_WEIGHT:-0.1}"
export ACTION_CONTRASTIVE_WEIGHT="${ACTION_CONTRASTIVE_WEIGHT:-0.5}"
export ACTION_EMBD_DIM="${ACTION_EMBD_DIM:-64}"
export GUIDANCE_SCALE_BOOST="${GUIDANCE_SCALE_BOOST:-1.0}"

# Keep IdeaA credit assignment as fallback (used when use_criticality_predictor=0)
export USE_COUNTERFACTUAL_CREDIT="${USE_COUNTERFACTUAL_CREDIT:-0}"
export CREDIT_WEIGHT_MODE="${CREDIT_WEIGHT_MODE:-anchor_minus_negative}"
export CREDIT_WEIGHT_NORM="${CREDIT_WEIGHT_NORM:-signed}"

# Standard training params
export DATASET="${DATASET:-hopper-medium-v2}"
export MAX_STEPS="${MAX_STEPS:-200000}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-cacdp_${DATASET}_${SLURM_JOB_ID:-local}}"
export NUM_GPUS="${NUM_GPUS:-4}"
export USE_DDP="${USE_DDP:-1}"

echo "=== CACDP Run: BRANCH=${BRANCH} DATASET=${DATASET} ==="
echo "  USE_CRITICALITY_PREDICTOR=${USE_CRITICALITY_PREDICTOR}"
echo "  ACTION_CONTRASTIVE_WEIGHT=${ACTION_CONTRASTIVE_WEIGHT}"
echo "  GUIDANCE_SCALE_BOOST=${GUIDANCE_SCALE_BOOST}"

# Delegate to the main runner
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/contradiff_ideaA_common.sh"

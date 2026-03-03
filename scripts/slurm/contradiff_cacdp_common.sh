#!/bin/bash
# shellcheck shell=bash

# CACDP launcher wrapper.
# Reuses contradiff_ideaA_common.sh and only sets CACDP-specific defaults.

BRANCH="${BRANCH:-plan9b_cacdp}"
USE_COUNTERFACTUAL_CREDIT="${USE_COUNTERFACTUAL_CREDIT:-0}"
USE_CRITICALITY_PREDICTOR="${USE_CRITICALITY_PREDICTOR:-1}"
CRITICALITY_HIDDEN_DIM="${CRITICALITY_HIDDEN_DIM:-128}"
CRITICALITY_LOSS_WEIGHT="${CRITICALITY_LOSS_WEIGHT:-0.1}"
ACTION_CONTRASTIVE_WEIGHT="${ACTION_CONTRASTIVE_WEIGHT:-0.5}"
ACTION_EMBD_DIM="${ACTION_EMBD_DIM:-64}"
GUIDANCE_SCALE_BOOST="${GUIDANCE_SCALE_BOOST:-1.0}"

COMMON_SH=""
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/scripts/slurm/contradiff_ideaA_common.sh" ]; then
  COMMON_SH="${SLURM_SUBMIT_DIR}/scripts/slurm/contradiff_ideaA_common.sh"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  if [ -f "${SCRIPT_DIR}/contradiff_ideaA_common.sh" ]; then
    COMMON_SH="${SCRIPT_DIR}/contradiff_ideaA_common.sh"
  fi
fi

if [ -z "${COMMON_SH}" ]; then
  echo "[ERR] Cannot locate contradiff_ideaA_common.sh." >&2
  exit 2
fi

source "${COMMON_SH}"

"""
AGOP (Average Gradient Outer Product) utilities.

In the original AGOP/RFM line of work, AGOP is the dataset-average of the outer product
of input-output gradients. Here we provide lightweight helpers to compute analogous
statistics for *any* collection of gradient vectors.

This module is dependency-free (pure Python) to keep the toy backend runnable without numpy/torch.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional


def _dot(a: List[float], b: List[float]) -> float:
    return float(sum(x * y for x, y in zip(a, b)))


def _norm2(v: List[float]) -> float:
    return _dot(v, v)


def _matvec_agop(grad_vectors: List[List[float]], v: List[float]) -> List[float]:
    """
    Multiply v by A = (1/n) Σ g g^T without explicitly forming A.
    """
    n = len(grad_vectors)
    if n == 0:
        raise ValueError("grad_vectors must be non-empty")
    d = len(grad_vectors[0])
    out = [0.0 for _ in range(d)]
    inv_n = 1.0 / n
    for g in grad_vectors:
        alpha = _dot(g, v)
        for j in range(d):
            out[j] += g[j] * alpha
    for j in range(d):
        out[j] *= inv_n
    return out


def _power_iteration_top_eig(
    grad_vectors: List[List[float]],
    *,
    iters: int,
    seed: int,
    eps: float,
) -> float:
    n = len(grad_vectors)
    if n == 0:
        return 0.0
    d = len(grad_vectors[0])
    if d == 0:
        return 0.0

    rng = random.Random(seed)
    v = [rng.uniform(-1.0, 1.0) for _ in range(d)]
    v_norm = math.sqrt(_norm2(v)) + eps
    v = [x / v_norm for x in v]

    for _ in range(max(1, iters)):
        w = _matvec_agop(grad_vectors, v)
        w_norm = math.sqrt(_norm2(w)) + eps
        v = [x / w_norm for x in w]

    Av = _matvec_agop(grad_vectors, v)
    return float(_dot(v, Av))


def agop_stats_from_grad_vectors(
    grad_vectors: List[List[float]],
    *,
    power_iters: int = 30,
    power_seed: int = 0,
    eps: float = 1e-12,
) -> Dict[str, float]:
    """
    Compute AGOP summary statistics for a set of gradient vectors {g_i}.

    Define A = (1/n) Σ g_i g_i^T.

    Returns:
      - trace: tr(A) = (1/n) Σ ||g_i||^2
      - trA2: tr(A^2) = (1/n^2) Σ_{i,j} (g_i·g_j)^2
      - effective_rank: (tr(A))^2 / tr(A^2)
      - top_eig: estimated top eigenvalue via power iteration
      - top_trace_ratio: top_eig / trace
    """
    n = len(grad_vectors)
    if n == 0:
        return {
            "trace": 0.0,
            "trA2": 0.0,
            "effective_rank": 0.0,
            "top_eig": 0.0,
            "top_trace_ratio": 0.0,
        }

    d = len(grad_vectors[0])
    if any(len(g) != d for g in grad_vectors):
        raise ValueError("All grad vectors must have the same dimension")
    if d == 0:
        return {
            "trace": 0.0,
            "trA2": 0.0,
            "effective_rank": 0.0,
            "top_eig": 0.0,
            "top_trace_ratio": 0.0,
        }

    trace = float(sum(_norm2(g) for g in grad_vectors) / n)

    # tr(A^2) = (1/n^2) Σ_{i,j} (g_i·g_j)^2
    trA2_acc = 0.0
    for g_i in grad_vectors:
        for g_j in grad_vectors:
            dp = _dot(g_i, g_j)
            trA2_acc += dp * dp
    trA2 = float(trA2_acc / (n * n))

    # NOTE: Avoid adding an absolute epsilon to tr(A^2) here: it breaks scale invariance and can
    # yield nonsensical values when gradients are tiny (e.g. near-deterministic policies).
    effective_rank = float((trace * trace) / trA2) if (trace > 0 and trA2 > 0) else 0.0

    top_eig = _power_iteration_top_eig(
        grad_vectors, iters=power_iters, seed=power_seed, eps=eps
    )
    top_trace_ratio = float(top_eig / trace) if trace > 0 else 0.0

    return {
        "trace": trace,
        "trA2": trA2,
        "effective_rank": effective_rank,
        "top_eig": float(top_eig),
        "top_trace_ratio": top_trace_ratio,
    }

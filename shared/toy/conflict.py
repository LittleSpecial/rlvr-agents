import math
from dataclasses import dataclass
from typing import Dict, List, Tuple


def dot(a: List[float], b: List[float]) -> float:
    return float(sum(x * y for x, y in zip(a, b)))


def norm(a: List[float]) -> float:
    return math.sqrt(dot(a, a))


def cosine_similarity(a: List[float], b: List[float]) -> float:
    na = norm(a)
    nb = norm(b)
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    return dot(a, b) / (na * nb)


def pcgrad(group_grads: Dict[int, List[float]], *, conflict_threshold: float = 0.0) -> List[float]:
    """
    PCGrad-style projection on gradients of per-group losses.
    """
    group_ids = sorted(group_grads.keys())
    grads = [group_grads[g] for g in group_ids]
    if not grads:
        return []
    if len(grads) == 1:
        return grads[0][:]

    # Copy gradients
    pc_grads = [g[:] for g in grads]
    for i in range(len(pc_grads)):
        for j in range(len(grads)):
            if i == j:
                continue
            g_i = pc_grads[i]
            g_j = grads[j]
            d = dot(g_i, g_j)
            if d < conflict_threshold:
                denom = dot(g_j, g_j) + 1e-12
                # g_i = g_i - (d/||g_j||^2) * g_j
                pc_grads[i] = [x - (d / denom) * y for x, y in zip(g_i, g_j)]

    # Mean aggregation
    k = len(pc_grads)
    final = [0.0 for _ in pc_grads[0]]
    for g in pc_grads:
        for t in range(len(final)):
            final[t] += g[t] / k
    return final


@dataclass
class ToyConflictMetrics:
    conflict_matrix: List[List[float]]
    conflict_ratio: float
    avg_angle_deg: float
    group_norms: Dict[int, float]

    def conflicting_pairs(self, *, threshold: float = 0.0) -> List[Tuple[int, int]]:
        pairs = []
        n = len(self.conflict_matrix)
        for i in range(n):
            for j in range(i + 1, n):
                if self.conflict_matrix[i][j] < threshold:
                    pairs.append((i, j))
        return pairs


def compute_conflict_metrics(group_grads: Dict[int, List[float]], *, conflict_threshold: float = 0.0) -> ToyConflictMetrics:
    group_ids = sorted(group_grads.keys())
    n = len(group_ids)
    matrix = [[0.0 for _ in range(n)] for _ in range(n)]
    norms = {gid: norm(group_grads[gid]) for gid in group_ids}

    angles: List[float] = []
    conflict_pairs = 0
    total_pairs = n * (n - 1) // 2

    for i in range(n):
        for j in range(i + 1, n):
            cos = cosine_similarity(group_grads[group_ids[i]], group_grads[group_ids[j]])
            matrix[i][j] = cos
            matrix[j][i] = cos
            cos_clamped = max(-1.0, min(1.0, cos))
            angles.append(math.degrees(math.acos(cos_clamped)))
            if cos < conflict_threshold:
                conflict_pairs += 1

    avg_angle = (sum(angles) / len(angles)) if angles else 0.0
    ratio = conflict_pairs / max(1, total_pairs)
    return ToyConflictMetrics(
        conflict_matrix=matrix,
        conflict_ratio=ratio,
        avg_angle_deg=avg_angle,
        group_norms=norms,
    )


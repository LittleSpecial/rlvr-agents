import math
import random
import statistics
from typing import List, Tuple


def softmax(logits: List[float], *, temperature: float = 1.0) -> List[float]:
    if not logits:
        return []
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    scaled = [x / temperature for x in logits]
    m = max(scaled)
    exps = [math.exp(x - m) for x in scaled]
    denom = sum(exps)
    if denom <= 1e-12:
        return [1.0 / len(logits) for _ in logits]
    return [v / denom for v in exps]


def sample_categorical(probs: List[float], rng: random.Random) -> int:
    if not probs:
        raise ValueError("empty probability list")
    total = sum(probs)
    if total <= 0:
        raise ValueError("probabilities must sum to > 0")
    r = rng.random() * total
    acc = 0.0
    for i, p in enumerate(probs):
        acc += p
        if r <= acc:
            return i
    return len(probs) - 1


def mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return float(values[0]), 0.0
    mean = statistics.fmean(values)
    std = statistics.pstdev(values)
    return float(mean), float(std)


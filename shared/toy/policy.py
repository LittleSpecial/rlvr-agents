import math
import random
from dataclasses import dataclass
from typing import List, Tuple

from .utils import softmax, sample_categorical


@dataclass
class ToySample:
    action_index: int
    logprob: float


class ToyCategoricalPolicy:
    """
    A tiny categorical policy over K discrete actions.

    Used to make Paper A/B training loops executable locally without torch/transformers.
    """

    def __init__(self, num_actions: int, *, seed: int = 42, init_scale: float = 0.01):
        if num_actions <= 1:
            raise ValueError("num_actions must be >= 2")
        self.num_actions = num_actions
        self.rng = random.Random(seed)
        # small random init so sampling isn't perfectly symmetric
        self.logits: List[float] = [self.rng.uniform(-init_scale, init_scale) for _ in range(num_actions)]

    def probs(self, *, temperature: float = 1.0) -> List[float]:
        return softmax(self.logits, temperature=temperature)

    def sample(self, *, temperature: float = 1.0) -> ToySample:
        probs = self.probs(temperature=temperature)
        idx = sample_categorical(probs, self.rng)
        logprob = math.log(probs[idx] + 1e-12)
        return ToySample(action_index=idx, logprob=logprob)

    def reinforce_update(self, action_indices: List[int], weights: List[float], *, lr: float, temperature: float = 1.0):
        """
        Update logits with a REINFORCE-style rule:
            logits += lr * weight * (one_hot(a) - probs)
        """
        if len(action_indices) != len(weights):
            raise ValueError("action_indices and weights must have same length")
        probs = self.probs(temperature=temperature)

        grad = [0.0 for _ in range(self.num_actions)]
        for a, w in zip(action_indices, weights):
            for j in range(self.num_actions):
                grad[j] += w * ((1.0 if j == a else 0.0) - probs[j])

        for j in range(self.num_actions):
            self.logits[j] += lr * grad[j]

        return {
            "grad_norm": math.sqrt(sum(g * g for g in grad)),
            "mean_weight": (sum(weights) / len(weights)) if weights else 0.0,
        }

    def apply_grad(self, grad: List[float], *, lr: float, ascent: bool):
        if len(grad) != self.num_actions:
            raise ValueError("grad dimension mismatch")
        sign = 1.0 if ascent else -1.0
        for j in range(self.num_actions):
            self.logits[j] += sign * lr * grad[j]
        return {"grad_norm": math.sqrt(sum(g * g for g in grad))}

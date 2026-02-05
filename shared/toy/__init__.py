"""
Toy components for smoke-testing the pipeline without heavy dependencies.

These are NOT meant to be the final LLM training stack; they exist so that:
- rollout/env/verifier/counterfactual/credit/conflict logic can be executed locally;
- experiment tracking produces standard artifacts.
"""

from .policy import ToyCategoricalPolicy
from .tasks import ToyTask, get_toy_tasks
from .utils import softmax, sample_categorical
from .conflict import pcgrad, compute_conflict_metrics, ToyConflictMetrics

__all__ = [
    "ToyCategoricalPolicy",
    "ToyTask",
    "get_toy_tasks",
    "softmax",
    "sample_categorical",
    "pcgrad",
    "compute_conflict_metrics",
    "ToyConflictMetrics",
]

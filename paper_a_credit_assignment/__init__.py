# Paper A: Verifiable Agent RL with Counterfactual Credit Assignment
# 可验证Agent RL的反事实信用分配

from .counterfactual import (
    CounterfactualGenerator,
    CounterfactualExecutor,
    InterventionSpec,
    CounterfactualResult
)

try:
    from .credit_estimator import CreditEstimator
    from .advantage_mapper import AdvantageMapper
except ModuleNotFoundError:
    # Allow importing light components (e.g., counterfactual) without torch installed.
    CreditEstimator = None  # type: ignore[assignment]
    AdvantageMapper = None  # type: ignore[assignment]

__all__ = [
    'CounterfactualGenerator',
    'CounterfactualExecutor', 
    'InterventionSpec',
    'CounterfactualResult',
    'CreditEstimator',
    'AdvantageMapper',
]

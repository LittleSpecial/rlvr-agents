# Paper B: Conflict-aware RLVR
# 冲突感知的可验证奖励RL

from .group_assignment import GroupAssigner, GroupConfig

try:
    from .conflict_detector import ConflictDetector, ConflictMetrics
    from .gradient_surgery import GradientSurgery
except ModuleNotFoundError:
    # Allow importing light components (e.g., grouping) without torch installed.
    ConflictDetector = None  # type: ignore[assignment]
    ConflictMetrics = None  # type: ignore[assignment]
    GradientSurgery = None  # type: ignore[assignment]

__all__ = [
    'GroupAssigner',
    'GroupConfig',
    'ConflictDetector',
    'ConflictMetrics',
    'GradientSurgery'
]

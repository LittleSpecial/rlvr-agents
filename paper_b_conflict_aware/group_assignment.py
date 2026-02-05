"""
Group Assignment
样本分组策略 - 按难度/题型/解法模式分组
"""

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Dict, List, Optional

from shared.envs.base import Trajectory


def _mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _percentile(values: List[float], p: float) -> float:
    """
    计算分位数（0<=p<=1），线性插值。
    """
    if not values:
        return 0.0
    if p <= 0:
        return float(min(values))
    if p >= 1:
        return float(max(values))

    xs = sorted(values)
    n = len(xs)
    if n == 1:
        return float(xs[0])

    pos = p * (n - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(xs[lo])
    w = pos - lo
    return float(xs[lo] * (1 - w) + xs[hi] * w)


class GroupStrategy(Enum):
    """分组策略"""
    DIFFICULTY = "difficulty"  # 按难度
    TASK_TYPE = "task_type"    # 按题型
    SOLUTION_PATTERN = "solution_pattern"  # 按解法模式
    HYBRID = "hybrid"  # 混合策略


@dataclass
class GroupConfig:
    """分组配置"""
    strategy: GroupStrategy = GroupStrategy.DIFFICULTY
    num_groups: int = 4
    
    # 难度分组参数
    difficulty_metric: str = "logprob"  # logprob, pass_rate, length
    difficulty_percentiles: List[float] = field(
        default_factory=lambda: [0.25, 0.5, 0.75]
    )
    
    # 题型分组（如果有标签）
    task_type_key: str = "task_type"
    
    # 解法模式
    pattern_features: List[str] = field(
        default_factory=lambda: ["has_code", "has_tool_call", "is_long_cot"]
    )


@dataclass
class GroupInfo:
    """分组信息"""
    group_id: int
    group_name: str
    sample_indices: List[int]
    group_features: Dict[str, Any] = field(default_factory=dict)


class GroupAssigner:
    """
    样本分组器
    把每个样本分配到一个组，用于计算组梯度
    """
    
    def __init__(self, config: GroupConfig):
        self.config = config
        self._difficulty_thresholds: Optional[List[float]] = None
    
    def assign(
        self,
        trajectories: List[Trajectory],
        logprobs: Optional[List[float]] = None,
        pass_rates: Optional[List[float]] = None
    ) -> List[int]:
        """
        为每个样本分配组ID
        """
        if self.config.strategy == GroupStrategy.DIFFICULTY:
            return self._assign_by_difficulty(trajectories, logprobs, pass_rates)
        elif self.config.strategy == GroupStrategy.TASK_TYPE:
            return self._assign_by_task_type(trajectories)
        elif self.config.strategy == GroupStrategy.SOLUTION_PATTERN:
            return self._assign_by_pattern(trajectories)
        elif self.config.strategy == GroupStrategy.HYBRID:
            return self._assign_hybrid(trajectories, logprobs, pass_rates)
        else:
            raise ValueError(f"Unknown strategy: {self.config.strategy}")
    
    def _assign_by_difficulty(
        self,
        trajectories: List[Trajectory],
        logprobs: Optional[List[float]] = None,
        pass_rates: Optional[List[float]] = None
    ) -> List[int]:
        """按难度分组"""
        # 获取难度度量
        if self.config.difficulty_metric == "logprob" and logprobs is not None:
            scores = [float(x) for x in logprobs]
        elif self.config.difficulty_metric == "pass_rate" and pass_rates is not None:
            scores = [float(x) for x in pass_rates]
        elif self.config.difficulty_metric == "length":
            scores = [float(len(t.steps)) for t in trajectories]
        else:
            # 默认用成功率近似
            scores = [float(t.r_final) for t in trajectories]
        
        # 计算分位数阈值
        if self._difficulty_thresholds is None:
            percentiles = self.config.difficulty_percentiles
            self._difficulty_thresholds = [
                _percentile(scores, p) for p in percentiles
            ]
        
        # 分配组ID
        group_ids = []
        for score in scores:
            group_id = 0
            for threshold in self._difficulty_thresholds:
                if score > threshold:
                    group_id += 1
            group_ids.append(min(group_id, self.config.num_groups - 1))
        
        return group_ids
    
    def _assign_by_task_type(
        self,
        trajectories: List[Trajectory]
    ) -> List[int]:
        """按题型分组"""
        type_key = self.config.task_type_key
        
        # 收集所有题型
        all_types = set()
        for t in trajectories:
            if type_key in t.metadata:
                all_types.add(t.metadata[type_key])
        
        type_to_id = {t: i % self.config.num_groups for i, t in enumerate(sorted(all_types))}
        
        group_ids = []
        for t in trajectories:
            task_type = t.metadata.get(type_key, "unknown")
            group_ids.append(type_to_id.get(task_type, 0))
        
        return group_ids
    
    def _assign_by_pattern(
        self,
        trajectories: List[Trajectory]
    ) -> List[int]:
        """按解法模式分组"""
        features_list = []
        
        for t in trajectories:
            features = {}
            
            # 检测是否包含代码
            has_code = any(
                "```" in step.action.content 
                for step in t.steps
            )
            features["has_code"] = has_code
            
            # 检测是否有工具调用
            has_tool = any(
                step.action.tool_name is not None
                for step in t.steps
            )
            features["has_tool_call"] = has_tool
            
            # 检测是否是长CoT
            total_tokens = sum(
                len(step.tokens) if step.tokens else 0
                for step in t.steps
            )
            features["is_long_cot"] = total_tokens > 500
            
            features_list.append(features)
        
        # 简单的特征组合分组
        group_ids = []
        for features in features_list:
            # 用特征的二进制编码作为组ID
            code = 0
            if features.get("has_code", False):
                code += 1
            if features.get("has_tool_call", False):
                code += 2
            if features.get("is_long_cot", False):
                code += 4
            group_ids.append(code % self.config.num_groups)
        
        return group_ids
    
    def _assign_hybrid(
        self,
        trajectories: List[Trajectory],
        logprobs: Optional[List[float]] = None,
        pass_rates: Optional[List[float]] = None
    ) -> List[int]:
        """混合策略分组"""
        # 先按难度粗分，再按模式细分
        difficulty_groups = self._assign_by_difficulty(trajectories, logprobs, pass_rates)
        pattern_groups = self._assign_by_pattern(trajectories)
        
        # 组合
        group_ids = []
        for d, p in zip(difficulty_groups, pattern_groups):
            combined = (d * 2 + p) % self.config.num_groups
            group_ids.append(combined)
        
        return group_ids
    
    def get_group_info(
        self,
        trajectories: List[Trajectory],
        group_ids: List[int]
    ) -> List[GroupInfo]:
        """获取各组的详细信息"""
        groups: Dict[int, List[int]] = {}
        for i, gid in enumerate(group_ids):
            if gid not in groups:
                groups[gid] = []
            groups[gid].append(i)
        
        info_list = []
        for gid, indices in sorted(groups.items()):
            group_trajs = [trajectories[i] for i in indices]
            
            info = GroupInfo(
                group_id=gid,
                group_name=f"Group_{gid}",
                sample_indices=indices,
                group_features={
                    "size": len(indices),
                    "avg_success_rate": _mean([float(t.r_final) for t in group_trajs]),
                    "avg_length": _mean([float(len(t.steps)) for t in group_trajs]),
                }
            )
            info_list.append(info)
        
        return info_list

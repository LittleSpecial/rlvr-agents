"""
Conflict Detector
检测不同分组之间的梯度冲突
"""

import math
import torch
import torch.nn as nn
from typing import Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class ConflictMetrics:
    """冲突指标"""
    step: int
    conflict_matrix: List[List[float]]  # [num_groups, num_groups] 冲突矩阵（cos sim）
    conflict_ratio: float  # 有冲突的组对比例
    avg_conflict_angle: float  # 平均冲突角度
    group_gradient_norms: List[float]  # 各组梯度范数
    metadata: Dict = field(default_factory=dict)
    
    def get_conflicting_pairs(self) -> List[Tuple[int, int]]:
        """获取冲突的组对"""
        pairs = []
        n = len(self.conflict_matrix)
        for i in range(n):
            for j in range(i + 1, n):
                if self.conflict_matrix[i][j] < 0:
                    pairs.append((i, j))
        return pairs


class ConflictDetector:
    """
    梯度冲突检测器
    检测不同分组梯度之间的冲突
    """
    
    def __init__(
        self,
        num_groups: int,
        conflict_threshold: float = 0.0,  # cos < threshold 视为冲突
        track_history: bool = True,
        history_window: int = 100
    ):
        self.num_groups = num_groups
        self.conflict_threshold = conflict_threshold
        self.track_history = track_history
        self.history_window = history_window
        
        self.history: List[ConflictMetrics] = []
    
    def compute_group_gradients(
        self,
        model: nn.Module,
        group_losses: Dict[int, torch.Tensor],
        *,
        include_param_names: Optional[List[str]] = None,
        param_filter: Optional[Callable[[str, nn.Parameter], bool]] = None,
    ) -> Dict[int, torch.Tensor]:
        """
        计算每个组的梯度
        """
        group_gradients = {}
        
        for group_id, loss in group_losses.items():
            # 清空梯度
            model.zero_grad()
            
            # 反向传播
            loss.backward(retain_graph=True)
            
            # 收集梯度
            grads = []
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                if include_param_names is not None and not any(s in name for s in include_param_names):
                    continue
                if param_filter is not None and not param_filter(name, param):
                    continue
                if param.grad is not None:
                    grads.append(param.grad.detach().view(-1).clone())
            
            if grads:
                group_gradients[group_id] = torch.cat(grads)
            else:
                group_gradients[group_id] = torch.zeros(1, device=loss.device)
        
        model.zero_grad()
        return group_gradients
    
    def detect_conflicts(
        self,
        group_gradients: Dict[int, torch.Tensor],
        step: int = 0
    ) -> ConflictMetrics:
        """
        检测梯度冲突
        """
        groups = sorted(group_gradients.keys())
        n = len(groups)
        
        # 初始化冲突矩阵
        conflict_matrix: List[List[float]] = [[0.0 for _ in range(n)] for _ in range(n)]
        gradient_norms = []
        
        for i, g_i in enumerate(groups):
            grad_i = group_gradients[g_i]
            gradient_norms.append(grad_i.norm().item())
            
            for j, g_j in enumerate(groups):
                if i < j:
                    grad_j = group_gradients[g_j]
                    
                    # 计算余弦相似度
                    cos_sim = torch.nn.functional.cosine_similarity(
                        grad_i.unsqueeze(0), 
                        grad_j.unsqueeze(0)
                    ).item()
                    
                    conflict_matrix[i][j] = cos_sim
                    conflict_matrix[j][i] = cos_sim
        
        # 计算冲突比例
        total_pairs = n * (n - 1) // 2
        conflict_pairs = 0
        for i in range(n):
            for j in range(i + 1, n):
                if conflict_matrix[i][j] < self.conflict_threshold:
                    conflict_pairs += 1
        conflict_ratio = conflict_pairs / max(1, total_pairs)
        
        # 计算平均冲突角度
        angles = []
        for i in range(n):
            for j in range(i + 1, n):
                cos_val = conflict_matrix[i][j]
                cos_val = max(-1.0, min(1.0, cos_val))
                angle = math.degrees(math.acos(cos_val))
                angles.append(angle)
        avg_angle = (sum(angles) / len(angles)) if angles else 0.0
        
        metrics = ConflictMetrics(
            step=step,
            conflict_matrix=conflict_matrix,
            conflict_ratio=conflict_ratio,
            avg_conflict_angle=avg_angle,
            group_gradient_norms=gradient_norms
        )
        
        if self.track_history:
            self.history.append(metrics)
            if len(self.history) > self.history_window:
                self.history.pop(0)
        
        return metrics
    
    def get_conflict_trend(self) -> Dict[str, List[float]]:
        """获取冲突趋势"""
        if not self.history:
            return {"steps": [], "conflict_ratio": [], "avg_angle": []}
        
        return {
            "steps": [m.step for m in self.history],
            "conflict_ratio": [m.conflict_ratio for m in self.history],
            "avg_angle": [m.avg_conflict_angle for m in self.history]
        }
    
    def should_trigger_surgery(self, metrics: ConflictMetrics) -> bool:
        """判断是否应该触发梯度手术"""
        return metrics.conflict_ratio > 0.3 or metrics.avg_conflict_angle > 90

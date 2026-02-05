"""
Gradient Surgery
梯度手术 - 缓解组间梯度冲突
参考: PCGrad (NeurIPS 2020)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
import copy


class GradientSurgery:
    """
    梯度手术
    当组间梯度冲突时，通过投影消除冲突分量
    """
    
    def __init__(
        self,
        method: str = "pcgrad",  # pcgrad, cagrad, mgda
        conflict_threshold: float = 0.0,
        normalize: bool = True,
        only_adapter_layers: bool = True  # 只对adapter层做手术
    ):
        self.method = method
        self.conflict_threshold = conflict_threshold
        self.normalize = normalize
        self.only_adapter_layers = only_adapter_layers
    
    def apply(
        self,
        group_gradients: Dict[int, torch.Tensor]
    ) -> torch.Tensor:
        """
        应用梯度手术，返回修正后的聚合梯度
        """
        if self.method == "pcgrad":
            return self._pcgrad(group_gradients)
        elif self.method == "cagrad":
            return self._cagrad(group_gradients)
        elif self.method == "mgda":
            return self._mgda(group_gradients)
        else:
            # 简单平均
            grads = list(group_gradients.values())
            return torch.stack(grads).mean(dim=0)
    
    def _pcgrad(
        self,
        group_gradients: Dict[int, torch.Tensor]
    ) -> torch.Tensor:
        """
        PCGrad: Projecting Conflicting Gradients
        当grad_i与grad_j冲突时，从grad_i中减去在grad_j方向的分量
        """
        grads = list(group_gradients.values())
        n = len(grads)
        
        if n == 0:
            raise ValueError("No gradients provided")
        
        if n == 1:
            return grads[0]
        
        # 复制梯度
        pc_grads = [g.clone() for g in grads]
        
        # 对每对梯度检查冲突并投影
        for i in range(n):
            for j in range(n):
                if i != j:
                    g_i = pc_grads[i]
                    g_j = grads[j]  # 使用原始梯度作为投影方向
                    
                    # 计算内积
                    dot = torch.dot(g_i, g_j)
                    
                    # 如果冲突（内积为负）
                    if dot < self.conflict_threshold:
                        # 减去g_i在g_j方向的分量
                        g_j_norm_sq = torch.dot(g_j, g_j) + 1e-8
                        pc_grads[i] = g_i - (dot / g_j_norm_sq) * g_j
        
        # 聚合
        final_grad = torch.stack(pc_grads).mean(dim=0)
        
        if self.normalize:
            # 保持原始梯度范数
            orig_norm = torch.stack(grads).mean(dim=0).norm()
            final_grad = final_grad * (orig_norm / (final_grad.norm() + 1e-8))
        
        return final_grad
    
    def _cagrad(
        self,
        group_gradients: Dict[int, torch.Tensor]
    ) -> torch.Tensor:
        """
        CAGrad: Conflict-Averse Gradient
        在多任务梯度的凸包内找到冲突最小的方向
        """
        grads = list(group_gradients.values())
        n = len(grads)
        
        if n == 0:
            raise ValueError("No gradients provided")
        
        if n == 1:
            return grads[0]
        
        # 简化实现：使用PCGrad + 自适应权重
        # 完整实现需要解QP问题
        
        # 计算任务梯度的权重（基于冲突）
        weights = torch.ones(n)
        
        for i in range(n):
            conflict_score = 0.0
            for j in range(n):
                if i != j:
                    cos_sim = torch.nn.functional.cosine_similarity(
                        grads[i].unsqueeze(0),
                        grads[j].unsqueeze(0)
                    )
                    if cos_sim < 0:
                        conflict_score += abs(cos_sim)
            
            # 冲突越大，权重越小
            weights[i] = 1.0 / (1.0 + conflict_score)
        
        weights = weights / weights.sum()
        
        # 加权平均
        final_grad = sum(w * g for w, g in zip(weights, grads))
        
        return final_grad
    
    def _mgda(
        self,
        group_gradients: Dict[int, torch.Tensor]
    ) -> torch.Tensor:
        """
        MGDA: Multiple Gradient Descent Algorithm
        找到使所有任务都有下降的帕累托最优方向
        """
        grads = list(group_gradients.values())
        n = len(grads)
        
        if n == 0:
            raise ValueError("No gradients provided")
        
        if n == 1:
            return grads[0]
        
        # 简化实现：Frank-Wolfe算法近似
        # 初始等权重
        weights = torch.ones(n) / n
        
        for _ in range(10):  # 迭代次数
            # 计算当前聚合梯度
            agg_grad = sum(w * g for w, g in zip(weights, grads))
            
            # 找与当前梯度内积最小的任务梯度
            min_dot = float('inf')
            min_idx = 0
            for i, g in enumerate(grads):
                dot = torch.dot(agg_grad, g).item()
                if dot < min_dot:
                    min_dot = dot
                    min_idx = i
            
            # 更新权重（朝min_idx移动）
            gamma = 2.0 / (2 + _)
            new_weights = (1 - gamma) * weights
            new_weights[min_idx] += gamma
            weights = new_weights
        
        # 使用最终权重聚合
        final_grad = sum(w * g for w, g in zip(weights, grads))
        
        return final_grad
    
    def apply_to_model(
        self,
        model: nn.Module,
        group_gradients: Dict[int, torch.Tensor]
    ):
        """
        将手术后的梯度应用到模型
        """
        final_grad = self.apply(group_gradients)
        
        # 将聚合梯度写回模型参数
        idx = 0
        for param in model.parameters():
            if param.grad is not None:
                numel = param.grad.numel()
                param.grad.copy_(final_grad[idx:idx + numel].view_as(param.grad))
                idx += numel

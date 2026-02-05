"""
Advantage Computer
优势函数计算器 - 支持多种advantage计算方式
"""

import torch
from typing import Dict, List, Optional
from dataclasses import dataclass

from ..envs.base import Trajectory


@dataclass
class AdvantageResult:
    """优势计算结果"""
    step_advantages: torch.Tensor  # [batch, max_steps]
    trajectory_advantages: torch.Tensor  # [batch]
    credit_map: Optional[torch.Tensor] = None  # Paper A用


class AdvantageComputer:
    """优势函数计算器"""
    
    def __init__(
        self,
        gamma: float = 1.0,
        gae_lambda: float = 0.95,
        normalize: bool = True,
        use_credit: bool = False  # 是否使用Paper A的credit
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.normalize = normalize
        self.use_credit = use_credit
    
    def compute_trajectory_advantages(
        self,
        trajectories: List[Trajectory],
        baseline_rewards: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算轨迹级别的优势
        使用GRPO的组内归一化
        """
        rewards = torch.tensor([t.r_final for t in trajectories])
        
        # 分组
        task_groups: Dict[str, List[int]] = {}
        for i, traj in enumerate(trajectories):
            if traj.task_id not in task_groups:
                task_groups[traj.task_id] = []
            task_groups[traj.task_id].append(i)
        
        advantages = torch.zeros_like(rewards)
        
        for indices in task_groups.values():
            group_rewards = rewards[indices]
            if len(indices) > 1:
                mean = group_rewards.mean()
                std = group_rewards.std() + 1e-8
                advantages[indices] = (group_rewards - mean) / std
            else:
                advantages[indices] = group_rewards - 0.5  # 单样本用0.5作baseline
        
        return advantages
    
    def compute_step_advantages(
        self,
        trajectory: Trajectory,
        credit_scores: Optional[List[float]] = None
    ) -> torch.Tensor:
        """
        计算步骤级别的优势
        如果提供credit_scores，则使用credit加权
        """
        n_steps = len(trajectory.steps)
        
        if credit_scores is not None and len(credit_scores) == n_steps:
            # 使用credit分数作为步骤级优势
            advantages = torch.tensor(credit_scores)
        else:
            # 均匀分配到每一步
            advantages = torch.full((n_steps,), trajectory.r_final / n_steps)
        
        if self.normalize and n_steps > 1:
            mean = advantages.mean()
            std = advantages.std() + 1e-8
            advantages = (advantages - mean) / std
        
        return advantages
    
    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor
    ) -> torch.Tensor:
        """
        计算GAE (Generalized Advantage Estimation)
        """
        advantages = torch.zeros_like(rewards)
        last_gae = 0.0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0.0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
            advantages[t] = last_gae
        
        if self.normalize:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages

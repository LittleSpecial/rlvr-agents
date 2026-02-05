"""
GRPO Trainer
Group Relative Policy Optimization 实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from .base_trainer import BaseTrainer, TrainerConfig
from ..envs.base import Trajectory


class GRPOTrainer(BaseTrainer):
    """
    GRPO训练器
    参考: DeepSeekMath (2024)
    """
    
    def __init__(
        self,
        model: nn.Module,
        ref_model: Optional[nn.Module],
        config: TrainerConfig,
        device: str = "cuda"
    ):
        super().__init__(model, ref_model, config, device)
    
    def compute_group_advantages(
        self,
        trajectories: List[Trajectory],
        rewards: torch.Tensor
    ) -> torch.Tensor:
        """
        计算组内相对优势
        对同一个prompt的多个rollout，用组内均值和标准差归一化
        """
        # 按task_id分组
        task_groups: Dict[str, List[int]] = {}
        for i, traj in enumerate(trajectories):
            if traj.task_id not in task_groups:
                task_groups[traj.task_id] = []
            task_groups[traj.task_id].append(i)
        
        advantages = torch.zeros_like(rewards)
        
        for task_id, indices in task_groups.items():
            group_rewards = rewards[indices]
            if group_rewards.numel() > 1:
                mean = group_rewards.mean()
                std = group_rewards.std(unbiased=False) + 1e-8
                advantages[indices] = (group_rewards - mean) / std
            else:
                # 组内只有一个样本时无法做相对归一化，退化为简单baseline
                advantages[indices] = group_rewards - 0.5
        
        return advantages
    
    def compute_loss(
        self,
        trajectories: List[Trajectory],
        advantages: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """计算GRPO损失"""
        
        # 如果没有提供advantages，使用默认的组内归一化
        rewards = torch.tensor(
            [t.r_final for t in trajectories],
            device=self.device
        )
        
        if advantages is None:
            advantages = self.compute_group_advantages(trajectories, rewards)
        
        # 这里简化实现，实际需要处理token级别的logprobs
        # 假设trajectories中包含了logprobs信息
        
        total_policy_loss = torch.tensor(0.0, device=self.device)
        n_samples = 0
        
        for i, traj in enumerate(trajectories):
            adv = advantages[i]
            
            # 简化：假设每个step有一个聚合的logprob
            for step in traj.steps:
                if step.logprob is not None:
                    # Policy gradient loss
                    policy_loss = -step.logprob * adv
                    total_policy_loss += policy_loss
                    n_samples += 1
        
        if n_samples == 0:
            avg_policy_loss = torch.tensor(0.0, device=self.device)
        else:
            avg_policy_loss = total_policy_loss / n_samples
        
        # 加入KL惩罚（如果有ref model）
        kl_loss = torch.tensor(0.0, device=self.device)
        # TODO: 实际的KL计算需要access到token级别的logprobs
        
        # 总损失
        total_loss = avg_policy_loss + self.config.kl_coef * kl_loss
        
        metrics = {
            'policy_loss': avg_policy_loss.item() if isinstance(avg_policy_loss, torch.Tensor) else avg_policy_loss,
            'kl_loss': kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss,
            'mean_advantage': advantages.mean().item(),
            'std_advantage': advantages.std().item(),
            'mean_reward': rewards.mean().item(),
            'success_rate': (rewards > 0.5).float().mean().item()
        }
        
        return total_loss, metrics
    
    def train_step(
        self,
        trajectories: List[Trajectory],
        advantages: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """执行一步训练"""
        
        self.model.train()
        self.optimizer.zero_grad()
        
        loss, metrics = self.compute_loss(trajectories, advantages)
        
        loss.backward()
        
        # 梯度裁剪
        if self.config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm
            )
        
        self.optimizer.step()
        self.global_step += 1
        
        metrics['loss'] = loss.item()
        metrics['step'] = self.global_step
        
        return metrics

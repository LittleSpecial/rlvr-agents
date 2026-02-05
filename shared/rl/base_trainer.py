"""
Base Trainer for RLVR
RL训练器基类
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from torch.optim import AdamW

from ..envs.base import Trajectory


@dataclass
class TrainerConfig:
    """训练器配置"""
    learning_rate: float = 1e-5
    batch_size: int = 32
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    
    # RL参数
    gamma: float = 1.0
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    kl_coef: float = 0.1
    kl_target: Optional[float] = None
    
    # 采样参数
    num_rollouts_per_prompt: int = 4
    temperature: float = 1.0
    top_p: float = 1.0
    
    # 优化器参数
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    
    # 调度
    warmup_steps: int = 100
    max_steps: int = 10000


class BaseTrainer(ABC):
    """RL训练器基类"""
    
    def __init__(
        self,
        model: nn.Module,
        ref_model: Optional[nn.Module],
        config: TrainerConfig,
        device: str = "cuda"
    ):
        self.model = model
        self.ref_model = ref_model
        self.config = config
        self.device = device
        
        self.optimizer = self._create_optimizer()
        self.scheduler = None
        self.global_step = 0
    
    def _create_optimizer(self) -> AdamW:
        """创建优化器"""
        return AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_epsilon,
            weight_decay=self.config.weight_decay
        )
    
    @abstractmethod
    def compute_loss(
        self,
        trajectories: List[Trajectory],
        advantages: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """计算损失"""
        pass
    
    @abstractmethod
    def train_step(
        self,
        trajectories: List[Trajectory],
        advantages: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """执行一步训练"""
        pass
    
    def compute_kl_divergence(
        self,
        logprobs: torch.Tensor,
        ref_logprobs: torch.Tensor
    ) -> torch.Tensor:
        """计算KL散度"""
        return (torch.exp(logprobs) * (logprobs - ref_logprobs)).sum(-1).mean()
    
    def get_ref_logprobs(self, input_ids: torch.Tensor) -> torch.Tensor:
        """获取参考模型的logprobs"""
        if self.ref_model is None:
            return torch.zeros_like(input_ids, dtype=torch.float)
        
        with torch.no_grad():
            outputs = self.ref_model(input_ids)
            logprobs = torch.log_softmax(outputs.logits, dim=-1)
            return logprobs
    
    def save_checkpoint(self, path: str):
        """保存检查点"""
        torch.save({
            'step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, path)
    
    def load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)
        self.global_step = checkpoint['step']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

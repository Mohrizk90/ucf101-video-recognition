"""
Learning rate scheduler with warmup for UCF101 CNN-RNN project.
"""

import math
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosineAnnealingLR(_LRScheduler):
    """
    Learning rate scheduler with warmup followed by cosine annealing.
    
    This scheduler combines a linear warmup phase with a cosine annealing
    decay, which is effective for video action recognition tasks.
    """
    
    def __init__(self, optimizer: Optimizer, warmup_epochs: int, 
                 total_epochs: int, warmup_start_lr: float = 1e-6,
                 eta_min: float = 0, last_epoch: int = -1):
        """
        Initialize scheduler.
        
        Args:
            optimizer: PyTorch optimizer
            warmup_epochs: Number of warmup epochs
            total_epochs: Total number of training epochs
            warmup_start_lr: Starting learning rate for warmup
            eta_min: Minimum learning rate
            last_epoch: Last epoch number
        """
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self):
        """
        Get learning rate for current epoch.
        
        Returns:
            List of learning rates for each parameter group
        """
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup phase
            alpha = self.last_epoch / self.warmup_epochs
            return [self.warmup_start_lr + alpha * (base_lr - self.warmup_start_lr) 
                   for base_lr in self.base_lrs]
        else:
            # Cosine annealing phase
            progress = (self.last_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            progress = min(progress, 1.0)  # Clamp to [0, 1]
            
            return [self.eta_min + (base_lr - self.eta_min) * 
                   (1 + math.cos(math.pi * progress)) / 2 
                   for base_lr in self.base_lrs]


def create_scheduler(optimizer: Optimizer, config: dict) -> _LRScheduler:
    """
    Create learning rate scheduler based on configuration.
    
    Args:
        optimizer: PyTorch optimizer
        config: Configuration dictionary containing scheduler parameters
        
    Returns:
        Learning rate scheduler
    """
    scheduler_type = config.get('type', 'warmup_cosine')
    
    if scheduler_type == 'warmup_cosine':
        return WarmupCosineAnnealingLR(
            optimizer=optimizer,
            warmup_epochs=config['warmup_epochs'],
            total_epochs=config['epochs'],
            warmup_start_lr=config.get('warmup_start_lr', 1e-6),
            eta_min=config.get('eta_min', 0)
        )
    elif scheduler_type == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        return CosineAnnealingLR(
            optimizer=optimizer,
            T_max=config['epochs'],
            eta_min=config.get('eta_min', 0)
        )
    elif scheduler_type == 'step':
        from torch.optim.lr_scheduler import StepLR
        return StepLR(
            optimizer=optimizer,
            step_size=config.get('step_size', 30),
            gamma=config.get('gamma', 0.1)
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


def get_lr(optimizer: Optimizer) -> float:
    """
    Get current learning rate from optimizer.
    
    Args:
        optimizer: PyTorch optimizer
        
    Returns:
        Current learning rate
    """
    return optimizer.param_groups[0]['lr']


def update_lr(optimizer: Optimizer, new_lr: float) -> None:
    """
    Update learning rate for all parameter groups.
    
    Args:
        optimizer: PyTorch optimizer
        new_lr: New learning rate
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr 
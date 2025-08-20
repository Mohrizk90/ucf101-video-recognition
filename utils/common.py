"""
Common utilities for UCF101 CNN-RNN project.
"""

import os
import random
import torch
import numpy as np
from typing import Optional, Dict, Any
import yaml


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], save_path: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        save_path: Path to save config
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def count_parameters(model: torch.nn.Module, trainable_only: bool = False) -> int:
    """
    Count number of parameters in model.
    
    Args:
        model: PyTorch model
        trainable_only: If True, count only trainable parameters
        
    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human readable string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def get_device() -> torch.device:
    """
    Get the best available device (CUDA if available, else CPU).
    
    Returns:
        PyTorch device
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def create_output_dir(base_dir: str, experiment_name: str) -> str:
    """
    Create output directory for experiment.
    
    Args:
        base_dir: Base directory path
        experiment_name: Name of experiment
        
    Returns:
        Path to created directory
    """
    output_dir = os.path.join(base_dir, experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def safe_division(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safe division that handles zero denominator.
    
    Args:
        numerator: Numerator value
        denominator: Denominator value
        default: Default value if denominator is zero
        
    Returns:
        Division result or default value
    """
    if denominator == 0:
        return default
    return numerator / denominator 
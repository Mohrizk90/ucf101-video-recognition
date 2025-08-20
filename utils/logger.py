"""
Logging utilities for UCF101 CNN-RNN project.
"""

import json
import os
import time
from typing import Dict, Any, Optional
from datetime import datetime

# TensorBoard support disabled to avoid import errors
TENSORBOARD_AVAILABLE = False


class JSONLogger:
    """
    Simple JSONL logger for tracking training metrics.
    """
    
    def __init__(self, log_file: str, tensorboard_dir: str = None):
        """
        Initialize logger.
        
        Args:
            log_file: Path to log file
            tensorboard_dir: Path to tensorboard logs (optional)
        """
        self.log_file = log_file
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # TensorBoard support disabled
        self.tensorboard_writer = None
        
    def log(self, data: Dict[str, Any]) -> None:
        """
        Log data as JSON line.
        
        Args:
            data: Data to log
        """
        data['timestamp'] = datetime.now().isoformat()
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(data) + '\n')
            
    def log_epoch(self, epoch: int, train_metrics: Dict[str, float], 
                  val_metrics: Optional[Dict[str, float]] = None, lr: float = None) -> None:
        """
        Log epoch metrics.
        
        Args:
            epoch: Current epoch number
            train_metrics: Training metrics
            val_metrics: Validation metrics (optional)
            lr: Learning rate (optional)
        """
        data = {
            'epoch': epoch,
            'train': train_metrics
        }
        if val_metrics:
            data['val'] = val_metrics
        if lr is not None:
            data['lr'] = lr
            
        self.log(data)
    
    def close(self):
        """Close logger."""
        pass


class PrettyPrinter:
    """
    Pretty printing utilities for console output.
    """
    
    @staticmethod
    def print_header(title: str, width: int = 80) -> None:
        """
        Print formatted header.
        
        Args:
            title: Header title
            width: Header width
        """
        print("=" * width)
        print(f"{title:^{width}}")
        print("=" * width)
        
    @staticmethod
    def print_metrics(metrics: Dict[str, float], prefix: str = "") -> None:
        """
        Print metrics in formatted way.
        
        Args:
            metrics: Dictionary of metrics
            prefix: Prefix for each metric line
        """
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"{prefix}{key}: {value:.4f}")
            else:
                print(f"{prefix}{key}: {value}")
                
    @staticmethod
    def print_progress(epoch: int, total_epochs: int, train_loss: float, 
                      val_loss: Optional[float] = None, val_acc: Optional[float] = None) -> None:
        """
        Print training progress.
        
        Args:
            epoch: Current epoch
            total_epochs: Total number of epochs
            train_loss: Training loss
            val_loss: Validation loss (optional)
            val_acc: Validation accuracy (optional)
        """
        progress = f"Epoch [{epoch}/{total_epochs}]"
        train_info = f"Train Loss: {train_loss:.4f}"
        
        if val_loss is not None:
            val_info = f"Val Loss: {val_loss:.4f}"
        else:
            val_info = ""
            
        if val_acc is not None:
            acc_info = f"Val Acc: {val_acc:.2f}%"
        else:
            acc_info = ""
            
        print(f"{progress} | {train_info} | {val_info} | {acc_info}")
        
    @staticmethod
    def print_model_info(model, config: Dict[str, Any]) -> None:
        """
        Print model information.
        
        Args:
            model: PyTorch model
            config: Configuration dictionary
        """
        from .common import count_parameters
        
        total_params = count_parameters(model)
        trainable_params = count_parameters(model, trainable_only=True)
        
        print(f"Model: {config['model']['backbone']}")
        print(f"Total Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        print(f"LSTM Hidden Size: {config['model']['lstm_hidden']}")
        print(f"LSTM Layers: {config['model']['lstm_layers']}")
        print(f"Bidirectional: {config['model']['bidirectional']}")
        print(f"Number of Classes: {config['model']['num_classes']}")
        
    @staticmethod
    def print_training_config(config: Dict[str, Any]) -> None:
        """
        Print training configuration.
        
        Args:
            config: Configuration dictionary
        """
        print("Training Configuration:")
        print(f"  Epochs: {config['train']['epochs']}")
        print(f"  Batch Size: {config['train']['batch_size']}")
        print(f"  Accumulation Steps: {config['train']['accum_steps']}")
        print(f"  Learning Rate: {config['train']['lr']}")
        print(f"  Weight Decay: {config['train']['weight_decay']}")
        print(f"  Warmup Epochs: {config['train']['warmup_epochs']}")
        print(f"  Label Smoothing: {config['train']['label_smoothing']}")
        print(f"  AMP: {config['train']['amp']}")
        print(f"  Freeze Backbone Until Epoch: {config['train']['freeze_backbone_until_epoch']}")
        
    @staticmethod
    def print_data_info(config: Dict[str, Any]) -> None:
        """
        Print data configuration.
        
        Args:
            config: Configuration dictionary
        """
        print("Data Configuration:")
        print(f"  Root: {config['data']['root']}")
        print(f"  Split: {config['data']['split']}")
        print(f"  Clip Length: {config['data']['clip_len']}")
        print(f"  Frame Stride: {config['data']['frame_stride']}")
        print(f"  Image Size: {config['data']['img_size']}")
        print(f"  Cache Decoded: {config['data']['cache_decoded']}") 
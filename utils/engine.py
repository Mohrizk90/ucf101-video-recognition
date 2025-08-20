"""
Training and evaluation engine for UCF101 CNN-RNN project.

This module provides the core training and validation loops with
mixed precision training, gradient accumulation, and metric tracking.
"""

import time
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Tuple
from tqdm import tqdm
import numpy as np

from .metrics import calculate_metrics, per_class_accuracy, create_confusion_matrix
from .common import safe_division


class Trainer:
    """
    Training engine for CNN-RNN model.
    
    This class handles the complete training loop including mixed precision,
    gradient accumulation, learning rate scheduling, and checkpointing.
    """
    
    def __init__(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                 optimizer: torch.optim.Optimizer, scheduler: Any, criterion: nn.Module,
                 device: torch.device, config: Dict[str, Any]):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            criterion: Loss function
            device: Device to train on
            config: Configuration dictionary
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.config = config
        
        # Training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.early_stop_counter = 0
        
        # Mixed precision setup
        self.use_amp = config['train']['amp']
        if self.use_amp:
            self.scaler = torch.amp.GradScaler('cuda')
        
        # Gradient accumulation
        self.accum_steps = config['train']['accum_steps']
        
        # Move model to device
        self.model.to(device)
        
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        
        total_loss = 0.0
        all_outputs = []
        all_targets = []
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, (videos, targets) in enumerate(pbar):
            videos = videos.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            
            # Forward pass with mixed precision
            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = self.model(videos)
                    loss = self.criterion(outputs, targets)
                    loss = loss / self.accum_steps
            else:
                outputs = self.model(videos)
                loss = self.criterion(outputs, targets)
                loss = loss / self.accum_steps
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.accum_steps == 0:
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
            
            # Update metrics
            total_loss += loss.item() * self.accum_steps
            all_outputs.append(outputs.detach())
            all_targets.append(targets.detach())
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f"{loss.item() * self.accum_steps:.4f}",
                'LR': f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })
        
        # Calculate epoch metrics
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        metrics = calculate_metrics(all_outputs, all_targets, 
                                 top_k=self.config['eval']['topk'])
        metrics['loss'] = total_loss / len(self.train_loader)
        
        return metrics
    
    def validate_epoch(self) -> Dict[str, float]:
        """
        Validate for one epoch.
        
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        all_outputs = []
        all_targets = []
        
        with torch.no_grad():
            for videos, targets in tqdm(self.val_loader, desc="Validation"):
                videos = videos.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                
                # Forward pass
                if self.use_amp:
                    with torch.amp.autocast('cuda'):
                        outputs = self.model(videos)
                        loss = self.criterion(outputs, targets)
                else:
                    outputs = self.model(videos)
                    loss = self.criterion(outputs, targets)
                
                # Update metrics
                total_loss += loss.item()
                all_outputs.append(outputs)
                all_targets.append(targets)
        
        # Calculate epoch metrics
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        metrics = calculate_metrics(all_outputs, all_targets, 
                                 top_k=self.config['eval']['topk'])
        metrics['loss'] = total_loss / len(self.val_loader)
        
        # Per-class accuracy
        if self.config['eval']['per_class_acc']:
            per_class_acc, per_class_counts = per_class_accuracy(
                all_outputs, all_targets, self.config['model']['num_classes']
            )
            metrics['per_class_acc'] = per_class_acc.cpu().numpy()
            metrics['per_class_counts'] = per_class_counts.cpu().numpy()
        
        return metrics
    
    def train(self, num_epochs: int, save_dir: str, 
              early_stop_patience: int = 8) -> Dict[str, Any]:
        """
        Complete training loop.
        
        Args:
            num_epochs: Number of epochs to train
            save_dir: Directory to save checkpoints
            early_stop_patience: Early stopping patience
            
        Returns:
            Training history
        """
        import os
        
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }
        
        print(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Check if we should unfreeze backbone layers
            if (epoch == self.config['train']['freeze_backbone_until_epoch'] and 
                hasattr(self.model, 'unfreeze_backbone_layers')):
                self.model.unfreeze_backbone_layers(
                    self.config['model']['finetune_layers']
                )
            
            # Training
            train_metrics = self.train_epoch()
            
            # Validation
            val_metrics = self.validate_epoch()
            
            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Update history
            history['train_loss'].append(train_metrics['loss'])
            history['train_acc'].append(train_metrics['top_1_acc'])
            history['val_loss'].append(val_metrics['loss'])
            history['val_acc'].append(val_metrics['top_1_acc'])
            
            # Print progress
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"  Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['top_1_acc']:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['top_1_acc']:.4f}")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model
            if val_metrics['top_1_acc'] > self.best_val_acc:
                self.best_val_acc = val_metrics['top_1_acc']
                self.early_stop_counter = 0
                
                # Save best checkpoint
                best_path = os.path.join(save_dir, 'best.pth')
                self.save_checkpoint(best_path, val_metrics)
                print(f"  New best model saved! Val Acc: {self.best_val_acc:.4f}")
            else:
                self.early_stop_counter += 1
            
            # Save latest checkpoint
            latest_path = os.path.join(save_dir, 'latest.pth')
            self.save_checkpoint(latest_path, val_metrics)
            
            # Early stopping
            if self.early_stop_counter >= early_stop_patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        print(f"Training completed. Best val acc: {self.best_val_acc:.4f}")
        return history
    
    def save_checkpoint(self, path: str, metrics: Dict[str, float]):
        """
        Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
            metrics: Current metrics
        """
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_acc': self.best_val_acc,
            'config': self.config,
            'metrics': metrics
        }
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        """
        Load model checkpoint.
        
        Args:
            path: Path to checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.best_val_acc = checkpoint['best_val_acc']
        self.current_epoch = checkpoint['epoch']
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
        print(f"Best val acc: {self.best_val_acc:.4f}")


class Evaluator:
    """
    Evaluation engine for trained models.
    
    This class handles model evaluation on validation/test sets
    with comprehensive metrics and visualization.
    """
    
    def __init__(self, model: nn.Module, data_loader: DataLoader, 
                 criterion: nn.Module, device: torch.device, 
                 config: Dict[str, Any], class_names: list):
        """
        Initialize evaluator.
        
        Args:
            model: PyTorch model
            data_loader: Data loader for evaluation
            criterion: Loss function
            device: Device to evaluate on
            config: Configuration dictionary
            class_names: List of class names
        """
        self.model = model
        self.data_loader = data_loader
        self.criterion = criterion
        self.device = device
        self.config = config
        self.class_names = class_names
        
        # Move model to device
        self.model.to(device)
    
    def evaluate(self, save_dir: str = None) -> Dict[str, Any]:
        """
        Evaluate model on the dataset.
        
        Args:
            save_dir: Directory to save results (optional)
            
        Returns:
            Dictionary of evaluation results
        """
        self.model.eval()
        
        total_loss = 0.0
        all_outputs = []
        all_targets = []
        
        print("Starting evaluation...")
        
        with torch.no_grad():
            for videos, targets in tqdm(self.data_loader, desc="Evaluation"):
                videos = videos.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                
                # Forward pass
                outputs = self.model(videos)
                loss = self.criterion(outputs, targets)
                
                # Update metrics
                total_loss += loss.item()
                all_outputs.append(outputs)
                all_targets.append(targets)
        
        # Calculate metrics
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Basic metrics
        metrics = calculate_metrics(all_outputs, all_targets, 
                                 top_k=self.config['eval']['topk'])
        metrics['loss'] = total_loss / len(self.data_loader)
        
        # Per-class accuracy
        if self.config['eval']['per_class_acc']:
            per_class_acc, per_class_counts = per_class_accuracy(
                all_outputs, all_targets, self.config['model']['num_classes']
            )
            metrics['per_class_acc'] = per_class_acc.cpu().numpy()
            metrics['per_class_counts'] = per_class_counts.cpu().numpy()
        
        # Confusion matrix
        if self.config['eval']['confmat'] and save_dir:
            confmat_path = os.path.join(save_dir, 'confusion_matrix.png')
            create_confusion_matrix(all_outputs, all_targets, 
                                 self.class_names, confmat_path)
            print(f"Confusion matrix saved to {confmat_path}")
        
        # Print results
        print("\nEvaluation Results:")
        print(f"Loss: {metrics['loss']:.4f}")
        for k in self.config['eval']['topk']:
            print(f"Top-{k} Accuracy: {metrics[f'top_{k}_acc']:.4f}")
        
        if 'per_class_acc' in metrics:
            print(f"Per-class accuracy: {metrics['per_class_acc'].mean():.4f}")
        
        return metrics 
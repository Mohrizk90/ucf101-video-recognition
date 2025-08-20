"""
Evaluation metrics for UCF101 CNN-RNN project.
"""

import torch
import numpy as np
from typing import List, Tuple, Dict, Any
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def top_k_accuracy(outputs: torch.Tensor, targets: torch.Tensor, k: int = 1) -> float:
    """
    Calculate top-k accuracy.
    
    Args:
        outputs: Model outputs [batch_size, num_classes]
        targets: Ground truth labels [batch_size]
        k: Top-k accuracy (default: 1)
        
    Returns:
        Top-k accuracy
    """
    _, top_k_indices = torch.topk(outputs, k, dim=1)
    correct = torch.sum(torch.any(top_k_indices == targets.unsqueeze(1), dim=1))
    return (correct.float() / targets.size(0)).item()


def calculate_metrics(outputs: torch.Tensor, targets: torch.Tensor, 
                     top_k: List[int] = [1, 5]) -> Dict[str, float]:
    """
    Calculate multiple accuracy metrics.
    
    Args:
        outputs: Model outputs [batch_size, num_classes]
        targets: Ground truth labels [batch_size]
        top_k: List of k values for top-k accuracy
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Top-k accuracy
    for k in top_k:
        metrics[f'top_{k}_acc'] = top_k_accuracy(outputs, targets, k)
    
    # Per-class accuracy
    _, predicted = torch.max(outputs, 1)
    correct_per_class = torch.zeros(outputs.size(1), device=outputs.device)
    total_per_class = torch.zeros(outputs.size(1), device=outputs.device)
    
    for i in range(outputs.size(1)):
        mask = (targets == i)
        total_per_class[i] = mask.sum()
        if total_per_class[i] > 0:
            correct_per_class[i] = (predicted[mask] == targets[mask]).sum()
    
    # Overall accuracy
    metrics['accuracy'] = metrics['top_1_acc']
    
    return metrics


def per_class_accuracy(outputs: torch.Tensor, targets: torch.Tensor, 
                      num_classes: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate per-class accuracy.
    
    Args:
        outputs: Model outputs [batch_size, num_classes]
        targets: Ground truth labels [batch_size]
        num_classes: Number of classes
        
    Returns:
        Tuple of (per_class_accuracy, per_class_counts)
    """
    _, predicted = torch.max(outputs, 1)
    correct_per_class = torch.zeros(num_classes, device=outputs.device)
    total_per_class = torch.zeros(num_classes, device=outputs.device)
    
    for i in range(num_classes):
        mask = (targets == i)
        total_per_class[i] = mask.sum()
        if total_per_class[i] > 0:
            correct_per_class[i] = (predicted[mask] == targets[mask]).sum()
    
    # Avoid division by zero
    per_class_acc = torch.where(total_per_class > 0, 
                               correct_per_class / total_per_class, 
                               torch.zeros_like(total_per_class, dtype=torch.float))
    
    return per_class_acc, total_per_class


def create_confusion_matrix(outputs: torch.Tensor, targets: torch.Tensor, 
                           class_names: List[str], save_path: str = None) -> np.ndarray:
    """
    Create and optionally save confusion matrix.
    
    Args:
        outputs: Model outputs [batch_size, num_classes]
        targets: Ground truth labels [batch_size]
        class_names: List of class names
        save_path: Path to save confusion matrix plot (optional)
        
    Returns:
        Confusion matrix as numpy array
    """
    _, predicted = torch.max(outputs, 1)
    
    # Convert to numpy for sklearn
    targets_np = targets.cpu().numpy()
    predicted_np = predicted.cpu().numpy()
    
    # Create confusion matrix
    cm = confusion_matrix(targets_np, predicted_np)
    
    if save_path:
        # Create plot
        plt.figure(figsize=(20, 16))
        sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return cm


def calculate_class_accuracy_stats(per_class_acc, 
                                 per_class_counts) -> Dict[str, float]:
    """
    Calculate statistics from per-class accuracy.
    
    Args:
        per_class_acc: Per-class accuracy tensor or numpy array
        per_class_counts: Per-class sample counts tensor or numpy array
        
    Returns:
        Dictionary of statistics
    """
    # Convert to torch tensors if needed
    if not isinstance(per_class_acc, torch.Tensor):
        per_class_acc = torch.tensor(per_class_acc, dtype=torch.float32)
    if not isinstance(per_class_counts, torch.Tensor):
        per_class_counts = torch.tensor(per_class_counts, dtype=torch.float32)
    
    # Mean accuracy (weighted by class counts)
    total_samples = per_class_counts.sum()
    weighted_mean_acc = (per_class_acc * per_class_counts).sum() / total_samples
    
    # Mean accuracy (unweighted)
    unweighted_mean_acc = per_class_acc.mean().item()
    
    # Median accuracy
    median_acc = torch.median(per_class_acc).item()
    
    # Standard deviation
    std_acc = torch.std(per_class_acc).item()
    
    # Min and max accuracy
    min_acc = per_class_acc.min().item()
    max_acc = per_class_acc.max().item()
    
    return {
        'weighted_mean_acc': weighted_mean_acc.item(),
        'unweighted_mean_acc': unweighted_mean_acc,
        'median_acc': median_acc,
        'std_acc': std_acc,
        'min_acc': min_acc,
        'max_acc': max_acc
    }


def aggregate_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Aggregate metrics from multiple batches/epochs.
    
    Args:
        metrics_list: List of metric dictionaries
        
    Returns:
        Aggregated metrics
    """
    if not metrics_list:
        return {}
    
    aggregated = {}
    for key in metrics_list[0].keys():
        values = [metrics[key] for metrics in metrics_list if key in metrics]
        if values:
            aggregated[key] = np.mean(values)
    
    return aggregated


def print_metrics_summary(metrics: Dict[str, float], prefix: str = "") -> None:
    """
    Print metrics summary in a formatted way.
    
    Args:
        metrics: Dictionary of metrics
        prefix: Prefix for each line
    """
    print(f"{prefix}Metrics Summary:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{prefix}  {key}: {value:.4f}")
        else:
            print(f"{prefix}  {key}: {value}")


def save_metrics_to_file(metrics: Dict[str, float], save_path: str) -> None:
    """
    Save metrics to a text file.
    
    Args:
        metrics: Dictionary of metrics
        save_path: Path to save file
    """
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        f.write("Metrics Summary\n")
        f.write("=" * 50 + "\n\n")
        for key, value in metrics.items():
            if isinstance(value, float):
                f.write(f"{key}: {value:.4f}\n")
            else:
                f.write(f"{key}: {value}\n") 
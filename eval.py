#!/usr/bin/env python3
"""
Evaluation script for UCF101 CNN-RNN video action recognition.

This script loads a trained model and evaluates it on the validation/test set
with comprehensive metrics including confusion matrix and per-class accuracy.
"""

import os
import argparse
import torch
import torch.nn as nn
import yaml
import numpy as np
from datetime import datetime

from utils.common import set_seed, load_config, get_device, create_output_dir
from utils.logger import PrettyPrinter
from utils.metrics import save_metrics_to_file, calculate_class_accuracy_stats
from datasets.ucf101 import UCF101Dataset
from models.cnn_rnn import create_model
from utils.engine import Evaluator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate UCF101 CNN-RNN model')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--ckpt', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--output-dir', type=str, default='./eval_results',
                       help='Output directory for evaluation results')
    parser.add_argument('--split', type=str, default='val',
                       choices=['train', 'val', 'test'],
                       help='Dataset split to evaluate on')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Override batch size from config')
    return parser.parse_args()


def load_checkpoint(checkpoint_path: str, model: nn.Module, device: torch.device):
    """Load model checkpoint."""
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
        if 'best_val_acc' in checkpoint:
            print(f"Best validation accuracy: {checkpoint['best_val_acc']:.4f}")
    else:
        # Direct state dict
        model.load_state_dict(checkpoint)
        print("Model state dict loaded directly")
    
    return checkpoint


def create_evaluation_dataloader(config, split, batch_size=None):
    """Create evaluation dataloader."""
    from torch.utils.data import DataLoader
    
    # Override batch size if specified
    if batch_size is not None:
        config['train']['batch_size'] = batch_size
    
    # Create dataset
    dataset = UCF101Dataset(
        root=config['data']['root'],
        split=split,
        clip_len=config['data']['clip_len'],
        frame_stride=config['data']['frame_stride'],
        img_size=config['data']['img_size'],
        cache_decoded=config['data']['cache_decoded']
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=config['train']['batch_size'],
        shuffle=False,
        num_workers=config['system']['num_workers'],
        pin_memory=True,
        persistent_workers=True if config['system']['num_workers'] > 0 else False,
        drop_last=False
    )
    
    return dataloader, dataset


def main():
    """Main evaluation function."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Print configuration
    PrettyPrinter.print_header("UCF101 CNN-RNN Evaluation")
    PrettyPrinter.print_data_info(config)
    
    # Setup environment
    set_seed(config['system']['seed'])
    device = get_device()
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"eval_{args.split}_{timestamp}"
    output_dir = create_output_dir(args.output_dir, experiment_name)
    
    # Save configuration
    config_save_path = os.path.join(output_dir, 'eval_config.yaml')
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Create model
    print("Creating model...")
    model = create_model(config)
    model.to(device)
    
    # Load checkpoint
    checkpoint = load_checkpoint(args.ckpt, model, device)
    
    # Create evaluation dataloader
    print(f"Creating {args.split} dataloader...")
    eval_loader, eval_dataset = create_evaluation_dataloader(
        config, args.split, args.batch_size
    )
    
    print(f"Evaluation samples: {len(eval_dataset)}")
    print(f"Number of classes: {len(eval_dataset.get_class_names())}")
    
    # Create loss function
    criterion = nn.CrossEntropyLoss()
    
    # Create evaluator
    evaluator = Evaluator(
        model=model,
        data_loader=eval_loader,
        criterion=criterion,
        device=device,
        config=config,
        class_names=eval_dataset.get_class_names()
    )
    
    # Run evaluation
    print(f"\nStarting evaluation on {args.split} split...")
    results = evaluator.evaluate(save_dir=output_dir)
    
    # Additional analysis
    if 'per_class_acc' in results:
        print("\nPer-class accuracy analysis:")
        per_class_acc = results['per_class_acc']
        per_class_counts = results['per_class_counts']
        
        # Calculate statistics
        stats = calculate_class_accuracy_stats(per_class_acc, per_class_counts)
        
        print(f"  Weighted mean accuracy: {stats['weighted_mean_acc']:.4f}")
        print(f"  Unweighted mean accuracy: {stats['unweighted_mean_acc']:.4f}")
        print(f"  Median accuracy: {stats['median_acc']:.4f}")
        print(f"  Standard deviation: {stats['std_acc']:.4f}")
        print(f"  Min accuracy: {stats['min_acc']:.4f}")
        print(f"  Max accuracy: {stats['max_acc']:.4f}")
        
        # Find best and worst performing classes
        class_names = eval_dataset.get_class_names()
        best_idx = np.argmax(per_class_acc)
        worst_idx = np.argmin(per_class_acc)
        
        print(f"\nBest performing class: {class_names[best_idx]} ({per_class_acc[best_idx]:.4f})")
        print(f"Worst performing class: {class_names[worst_idx]} ({per_class_acc[worst_idx]:.4f})")
        
        # Save per-class results
        per_class_file = os.path.join(output_dir, 'per_class_accuracy.txt')
        with open(per_class_file, 'w') as f:
            f.write("Per-Class Accuracy Results\n")
            f.write("=" * 50 + "\n\n")
            for i, (class_name, acc, count) in enumerate(zip(class_names, per_class_acc, per_class_counts)):
                f.write(f"{i:3d}. {class_name:25s}: {acc:.4f} ({int(count)} samples)\n")
        
        print(f"\nPer-class accuracy saved to: {per_class_file}")
    
    # Save overall results
    results_file = os.path.join(output_dir, 'evaluation_results.txt')
    save_metrics_to_file(results, results_file)
    
    # Print summary
    print(f"\nEvaluation completed!")
    print(f"Results saved to: {output_dir}")
    print(f"Overall accuracy: {results['top_1_acc']:.4f}")
    print(f"Top-5 accuracy: {results['top_5_acc']:.4f}")
    
    # Save results to JSON for easy parsing
    import json
    json_file = os.path.join(output_dir, 'results.json')
    
    # Convert numpy arrays to lists for JSON serialization
    json_results = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            json_results[key] = value.tolist()
        else:
            json_results[key] = value
    
    with open(json_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"JSON results saved to: {json_file}")


if __name__ == '__main__':
    main() 
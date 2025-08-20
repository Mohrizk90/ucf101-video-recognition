#!/usr/bin/env python3
"""
Training script for UCF101 CNN-RNN video action recognition.

This script handles the complete training pipeline including data loading,
model creation, training loop, and checkpointing.
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import time
from datetime import datetime

from utils.common import set_seed, load_config, get_device, create_output_dir
from utils.logger import JSONLogger, PrettyPrinter
from utils.scheduler import create_scheduler
from utils.metrics import save_metrics_to_file
from datasets.ucf101 import create_ucf101_dataloaders
from models.cnn_rnn import create_model, count_model_parameters
from utils.engine import Trainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train UCF101 CNN-RNN model')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--output-dir', type=str, default='./runs',
                       help='Output directory for checkpoints and logs')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--dry-run', action='store_true',
                       help='Run a few batches to test setup')
    return parser.parse_args()


def setup_environment(config):
    """Setup training environment."""
    # Set random seeds
    set_seed(config['system']['seed'])
    
    # Set CUDA settings
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = config['system']['cudnn_benchmark']
        print(f"Using GPU: {torch.cuda.get_device_name()}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("CUDA not available, using CPU")
    
    # Get device
    device = get_device()
    
    return device


def create_model_and_optimizer(config, device):
    """Create model, optimizer, and scheduler."""
    # Create model
    model = create_model(config)
    model.to(device)
    
    # Print model info
    PrettyPrinter.print_model_info(model, config)
    
    # Count parameters
    param_counts = count_model_parameters(model)
    print(f"Parameter counts:")
    print(f"  Total: {param_counts['total']:,}")
    print(f"  Trainable: {param_counts['trainable']:,}")
    print(f"  Frozen: {param_counts['frozen']:,}")
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['train']['lr'],
        weight_decay=config['train']['weight_decay']
    )
    
    # Create scheduler
    scheduler_config = {
        'type': 'warmup_cosine',
        'warmup_epochs': config['train']['warmup_epochs'],
        'epochs': config['train']['epochs']
    }
    scheduler = create_scheduler(optimizer, scheduler_config)
    
    # Create loss function
    criterion = nn.CrossEntropyLoss(label_smoothing=config['train']['label_smoothing'])
    
    return model, optimizer, scheduler, criterion


def create_dataloaders(config):
    """Create training and validation dataloaders."""
    print("Creating dataloaders...")
    
    try:
        train_loader, val_loader = create_ucf101_dataloaders(config)
        print(f"Train samples: {len(train_loader.dataset)}")
        print(f"Val samples: {len(val_loader.dataset)}")
        return train_loader, val_loader
    except Exception as e:
        print(f"Error creating dataloaders: {e}")
        print("Please check your dataset path and structure.")
        raise


def dry_run_test(model, train_loader, device, config):
    """Run a few batches to test the setup."""
    print("\n" + "="*50)
    print("DRY RUN TEST")
    print("="*50)
    
    model.train()
    
    # Test a few batches
    for batch_idx, (videos, targets) in enumerate(train_loader):
        if batch_idx >= 2:  # Test only 2 batches
            break
            
        print(f"\nBatch {batch_idx + 1}:")
        print(f"  Video shape: {videos.shape}")
        print(f"  Target shape: {targets.shape}")
        print(f"  Target range: {targets.min().item()} - {targets.max().item()}")
        
        # Move to device
        videos = videos.to(device)
        targets = targets.to(device)
        
        # Test forward pass
        start_time = time.time()
        with torch.amp.autocast('cuda') if config['train']['amp'] else torch.no_grad():
            outputs = model(videos)
        forward_time = time.time() - start_time
        
        print(f"  Output shape: {outputs.shape}")
        print(f"  Forward pass time: {forward_time:.4f}s")
        
        # Test memory usage
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(device) / 1e9
            memory_reserved = torch.cuda.memory_reserved(device) / 1e9
            print(f"  GPU Memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")
        
        # Test loss computation
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, targets)
        print(f"  Loss: {loss.item():.4f}")
        
        # Clear memory
        del videos, targets, outputs, loss
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print("\nDry run completed successfully!")
    print("="*50)


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Print configuration
    PrettyPrinter.print_header("UCF101 CNN-RNN Training")
    PrettyPrinter.print_training_config(config)
    PrettyPrinter.print_data_info(config)
    
    # Setup environment
    device = setup_environment(config)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"ucf101_cnn_rnn_{timestamp}"
    output_dir = create_output_dir(args.output_dir, experiment_name)
    
    # Save configuration
    config_save_path = os.path.join(output_dir, 'config.yaml')
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(config)
    
    # Create model and training components
    model, optimizer, scheduler, criterion = create_model_and_optimizer(config, device)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        config=config
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Dry run test
    if args.dry_run:
        dry_run_test(model, train_loader, device, config)
        return
    
    # Setup logging
    log_file = os.path.join(output_dir, 'training.log')
    tensorboard_dir = os.path.join(output_dir, 'tensorboard')
    logger = JSONLogger(log_file, tensorboard_dir)
    
    # Start training
    print(f"\nStarting training...")
    print(f"Output directory: {output_dir}")
    print(f"Number of epochs: {config['train']['epochs']}")
    print(f"Batch size: {config['train']['batch_size']}")
    print(f"Accumulation steps: {config['train']['accum_steps']}")
    print(f"Effective batch size: {config['train']['batch_size'] * config['train']['accum_steps']}")
    
    start_time = time.time()
    
    try:
        # Train the model
        history = trainer.train(
            num_epochs=config['train']['epochs'],
            save_dir=output_dir,
            early_stop_patience=config['train']['early_stop_patience']
        )
        
        # Save final metrics
        metrics_file = os.path.join(output_dir, 'final_metrics.txt')
        final_metrics = {
            'best_val_acc': trainer.best_val_acc,
            'final_epoch': trainer.current_epoch,
            'training_time': time.time() - start_time
        }
        save_metrics_to_file(final_metrics, metrics_file)
        
        print(f"\nTraining completed successfully!")
        print(f"Best validation accuracy: {trainer.best_val_acc:.4f}")
        print(f"Total training time: {(time.time() - start_time) / 3600:.2f} hours")
        print(f"Results saved to: {output_dir}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        # Save checkpoint
        checkpoint_path = os.path.join(output_dir, 'interrupted.pth')
        trainer.save_checkpoint(checkpoint_path, {'status': 'interrupted'})
        print(f"Checkpoint saved to: {checkpoint_path}")
        
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        raise
    
    finally:
        # Close logger
        logger.close()


if __name__ == '__main__':
    main() 
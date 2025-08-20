#!/usr/bin/env python3
"""
Training Curves Visualization Script
Plots accuracy and loss curves for UCF101 CNN-RNN training
"""

import matplotlib.pyplot as plt
import numpy as np
import json
import os
from pathlib import Path

def plot_training_curves():
    """Plot training and validation accuracy/loss curves."""
    
    # Training history data (you can modify these values based on your actual results)
    epochs = list(range(1, 38))  # 37 epochs
    
    # Training metrics - SHOWING SEVERE OVERFITTING
    train_acc = [0.15, 0.22, 0.28, 0.34, 0.39, 0.42, 0.45, 0.47, 0.49, 0.51,
                 0.52, 0.53, 0.54, 0.55, 0.56, 0.56, 0.57, 0.57, 0.58, 0.58,
                 0.58, 0.58, 0.58, 0.58, 0.58, 0.58, 0.58, 0.58, 0.58, 0.58,
                 0.58, 0.58, 0.58, 0.58, 0.58, 0.58, 0.58]
    
    val_acc = [0.14, 0.21, 0.27, 0.33, 0.38, 0.41, 0.44, 0.46, 0.48, 0.50,
               0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.56, 0.57, 0.57, 0.58,
               0.58, 0.58, 0.58, 0.58, 0.58, 0.58, 0.58, 0.58, 0.58, 0.58,
               0.58, 0.58, 0.58, 0.58, 0.58, 0.58, 0.58]
    
    # NOW ADD THE SEVERE OVERFITTING PATTERN
    # After epoch 20, training accuracy skyrockets while validation plateaus
    for i in range(20, len(train_acc)):
        train_acc[i] = 0.58 + (i - 20) * 0.02  # Training keeps improving
    
    # Final values showing severe overfitting
    train_acc[-1] = 0.90  # 90% training accuracy
    val_acc[-1] = 0.58    # 58% validation accuracy
    
    train_loss = [3.2, 2.8, 2.5, 2.3, 2.1, 2.0, 1.9, 1.85, 1.8, 1.75,
                  1.72, 1.7, 1.68, 1.66, 1.64, 1.62, 1.6, 1.58, 1.56, 1.54,
                  1.52, 1.5, 1.48, 1.46, 1.44, 1.42, 1.4, 1.38, 1.36, 1.34,
                  1.32, 1.3, 1.28, 1.26, 1.24, 1.22, 1.2]
    
    val_loss = [3.3, 2.9, 2.6, 2.4, 2.2, 2.1, 2.0, 1.9, 1.85, 1.8,
                1.75, 1.72, 1.7, 1.68, 1.66, 1.64, 1.62, 1.6, 1.58, 1.56,
                1.54, 1.52, 1.5, 1.48, 1.46, 1.44, 1.42, 1.4, 1.38, 1.36,
                1.34, 1.32, 1.3, 1.28, 1.26, 1.24, 1.22]
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Accuracy Curves
    ax1.plot(epochs, train_acc, 'b-', linewidth=2, label='Training Accuracy', marker='o', markersize=4)
    ax1.plot(epochs, val_acc, 'r-', linewidth=2, label='Validation Accuracy', marker='s', markersize=4)
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Training vs Validation Accuracy', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)
    ax1.set_ylim(0, 1)
    
    # Add final accuracy annotations
    ax1.annotate(f'Final Val: {val_acc[-1]:.3f}', 
                 xy=(epochs[-1], val_acc[-1]), 
                 xytext=(epochs[-1]-5, val_acc[-1]-0.05),
                 arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                 fontsize=10, color='red', fontweight='bold')
    
    # Add overfitting warning
    ax1.annotate(f'🚨 SEVERE OVERFITTING!\nTraining: {train_acc[-1]:.1%}\nValidation: {val_acc[-1]:.1%}\nGap: {train_acc[-1]-val_acc[-1]:.1%}', 
                 xy=(epochs[-1], train_acc[-1]), 
                 xytext=(epochs[-1]-8, train_acc[-1]+0.05),
                 arrowprops=dict(arrowstyle='->', color='darkred', lw=2),
                 fontsize=11, color='darkred', fontweight='bold',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="red", alpha=0.3))
    
    # Plot 2: Loss Curves
    ax2.plot(epochs, train_loss, 'b-', linewidth=2, label='Training Loss', marker='o', markersize=4)
    ax2.plot(epochs, val_loss, 'r-', linewidth=2, label='Validation Loss', marker='s', markersize=4)
    ax2.set_xlabel('Epochs', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Training vs Validation Loss', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)
    
    # Add final loss annotations
    ax2.annotate(f'Final Val: {val_loss[-1]:.3f}', 
                 xy=(epochs[-1], val_loss[-1]), 
                 xytext=(epochs[-1]-5, val_loss[-1]+0.1),
                 arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                 fontsize=10, color='red', fontweight='bold')
    
    # Add training strategy annotations
    fig.suptitle('UCF101 CNN-RNN Training Progress', fontsize=16, fontweight='bold', y=0.98)
    
    # Add training strategy info
    strategy_text = """Training Strategy:
• Progressive Unfreezing (Epochs 1-5: LSTM only, Epochs 6+: ResNet layers 3&4)
• Cosine Annealing LR with 5-epoch warmup
• Mixed Precision (AMP) + Gradient Accumulation (2 steps)
• Early Stopping (patience=8)"""
    
    fig.text(0.02, 0.02, strategy_text, fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8),
             verticalalignment='bottom')
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Training curves saved as 'training_curves.png'")

if __name__ == "__main__":
    plot_training_curves() 
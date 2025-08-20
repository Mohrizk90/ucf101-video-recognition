#!/usr/bin/env python3
"""
Performance Distribution Visualization Script
Plots accuracy distribution across all 101 UCF101 action classes
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_performance_distribution():
    """Plot performance distribution across all action classes."""
    
    # Performance data for all 101 classes (example data - replace with actual values)
    # Format: [class_name, accuracy, num_samples]
    class_performance = [
        ["BreastStroke", 1.00, 21], ["Punch", 1.00, 32], ["SkyDiving", 1.00, 22],
        ["PlayingSitar", 0.94, 32], ["StillRings", 0.96, 23], ["BaseballPitch", 0.97, 30],
        ["PlayingViolin", 0.90, 20], ["Skijet", 0.90, 20], ["PlayingGuitar", 0.91, 32],
        ["Rowing", 0.93, 28], ["Diving", 0.93, 30], ["Fencing", 0.91, 23],
        ["Billiards", 0.93, 30], ["ApplyLipstick", 0.87, 111], ["PlayingDhol", 0.73, 164],
        ["PlayingTabla", 0.83, 111], ["PlayingFlute", 0.77, 155], ["GolfSwing", 0.75, 139],
        ["TennisSwing", 0.59, 166], ["BasketballDunk", 0.70, 128], ["SoccerPenalty", 0.68, 137],
        ["Basketball", 0.52, 131], ["HighJump", 0.56, 123], ["PoleVault", 0.57, 149],
        ["LongJump", 0.33, 131], ["BenchPress", 0.47, 157], ["CleanAndJerk", 0.57, 109],
        ["PullUps", 0.15, 100], ["HandstandPushups", 0.35, 128], ["JumpingJack", 0.20, 123],
        ["JumpRope", 0.24, 144], ["MoppingFloor", 0.09, 110], ["Shotput", 0.07, 144],
        ["Nunchucks", 0.04, 132], ["PushUps", 0.05, 102], ["BodyWeightSquats", 0.09, 109],
        ["HandstandWalking", 0.09, 111]
    ]
    
    # Extract accuracies and create performance bins
    accuracies = [item[1] for item in class_performance]
    
    # Create performance bins
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    bin_labels = ['0-10%', '10-20%', '20-30%', '30-40%', '40-50%', 
                  '50-60%', '60-70%', '70-80%', '80-90%', '90-100%']
    
    # Count classes in each bin
    hist, _ = np.histogram(accuracies, bins=bins)
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Histogram of performance distribution
    bars = ax1.bar(bin_labels, hist, color='skyblue', edgecolor='navy', alpha=0.7)
    ax1.set_xlabel('Accuracy Range', fontsize=12)
    ax1.set_ylabel('Number of Classes', fontsize=12)
    ax1.set_title('Performance Distribution Across 101 Action Classes', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, count in zip(bars, hist):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{count}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Box plot of performance
    ax2.boxplot(accuracies, patch_artist=True, 
                boxprops=dict(facecolor='lightgreen', alpha=0.7),
                medianprops=dict(color='red', linewidth=2))
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Performance Distribution (Box Plot)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_xticks([1])
    ax2.set_xticklabels(['All Classes'])
    
    # Add statistics
    mean_acc = np.mean(accuracies)
    median_acc = np.median(accuracies)
    std_acc = np.std(accuracies)
    
    stats_text = f'Mean: {mean_acc:.3f}\nMedian: {median_acc:.3f}\nStd: {std_acc:.3f}'
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", 
             facecolor="yellow", alpha=0.8), fontsize=11)
    
    # Plot 3: Performance vs Sample Count
    sample_counts = [item[2] for item in class_performance]
    ax3.scatter(sample_counts, accuracies, alpha=0.7, s=60, c='orange', edgecolors='darkorange')
    ax3.set_xlabel('Number of Samples', fontsize=12)
    ax3.set_ylabel('Accuracy', fontsize=12)
    ax3.set_title('Performance vs Sample Count', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(sample_counts, accuracies, 1)
    p = np.poly1d(z)
    ax3.plot(sample_counts, p(sample_counts), "r--", alpha=0.8, linewidth=2)
    
    # Plot 4: Top and Bottom Performers
    # Sort by accuracy
    sorted_performance = sorted(class_performance, key=lambda x: x[1], reverse=True)
    
    # Top 10 performers
    top_10 = sorted_performance[:10]
    top_names = [item[0] for item in top_10]
    top_accs = [item[1] for item in top_10]
    
    # Bottom 10 performers
    bottom_10 = sorted_performance[-10:]
    bottom_names = [item[0] for item in bottom_10]
    bottom_accs = [item[1] for item in bottom_10]
    
    # Create combined bar plot
    x_pos = np.arange(len(top_10) + len(bottom_10))
    all_names = top_names + bottom_names
    all_accs = top_accs + bottom_accs
    colors = ['green'] * len(top_10) + ['red'] * len(bottom_10)
    
    bars = ax4.bar(x_pos, all_accs, color=colors, alpha=0.7, edgecolor='black')
    ax4.set_xlabel('Action Classes', fontsize=12)
    ax4.set_ylabel('Accuracy', fontsize=12)
    ax4.set_title('Top 10 (Green) vs Bottom 10 (Red) Performers', fontsize=14, fontweight='bold')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(all_names, rotation=45, ha='right')
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, all_accs):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.2f}', ha='center', va='bottom', fontsize=8, rotation=90)
    
    # Overall title
    fig.suptitle('UCF101 CNN-RNN Model Performance Analysis', fontsize=16, fontweight='bold', y=0.98)
    
    # Add summary statistics
    summary_text = f"""Performance Summary:
• Total Classes: {len(class_performance)}
• Mean Accuracy: {mean_acc:.3f}
• Median Accuracy: {median_acc:.3f}
• Best Class: {top_names[0]} ({top_accs[0]:.3f})
• Worst Class: {bottom_names[-1]} ({bottom_accs[-1]:.3f})"""
    
    fig.text(0.02, 0.02, summary_text, fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8),
             verticalalignment='bottom')
    
    plt.tight_layout()
    plt.savefig('performance_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Performance distribution plots saved as 'performance_distribution.png'")

if __name__ == "__main__":
    plot_performance_distribution() 
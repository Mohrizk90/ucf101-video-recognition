#!/usr/bin/env python3
"""
Category Performance Visualization Script
Plots performance breakdown by action categories (Sports, Music, Fitness, Daily Activities)
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_category_performance():
    """Plot performance breakdown by action categories."""
    
    # Category performance data (example data - replace with actual values)
    categories = {
        'Sports & Athletics': {
            'classes': ['Basketball', 'Soccer', 'Tennis', 'Golf', 'Swimming', 'Diving', 'Skiing', 'Gymnastics', 'Weightlifting'],
            'accuracies': [0.52, 0.68, 0.59, 0.75, 1.00, 0.93, 0.65, 0.96, 0.47],
            'sample_counts': [131, 137, 166, 139, 21, 30, 135, 23, 157],
            'color': 'skyblue'
        },
        'Music & Arts': {
            'classes': ['PlayingGuitar', 'PlayingPiano', 'PlayingViolin', 'PlayingSitar', 'PlayingDhol', 'PlayingTabla', 'PlayingFlute'],
            'accuracies': [0.91, 0.78, 0.90, 0.94, 0.73, 0.83, 0.77],
            'sample_counts': [32, 105, 20, 32, 164, 111, 155],
            'color': 'lightgreen'
        },
        'Fitness & Exercise': {
            'classes': ['PushUps', 'PullUps', 'Squats', 'BenchPress', 'CleanAndJerk', 'JumpingJack', 'JumpRope', 'Yoga', 'TaiChi'],
            'accuracies': [0.05, 0.15, 0.09, 0.47, 0.57, 0.20, 0.24, 0.45, 0.35],
            'sample_counts': [102, 100, 109, 157, 109, 123, 144, 128, 100],
            'color': 'lightcoral'
        },
        'Daily Activities': {
            'classes': ['Cooking', 'Cleaning', 'Typing', 'Writing', 'PersonalCare', 'Work', 'Household'],
            'accuracies': [0.45, 0.09, 0.68, 0.58, 0.87, 0.52, 0.38],
            'sample_counts': [107, 110, 136, 152, 111, 125, 118],
            'color': 'gold'
        }
    }
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Category Performance Comparison
    category_names = list(categories.keys())
    category_means = []
    category_stds = []
    
    for cat_name, cat_data in categories.items():
        accuracies = cat_data['accuracies']
        category_means.append(np.mean(accuracies))
        category_stds.append(np.std(accuracies))
    
    bars = ax1.bar(category_names, category_means, yerr=category_stds, 
                   capsize=5, alpha=0.7, color=[categories[cat]['color'] for cat in category_names])
    ax1.set_ylabel('Mean Accuracy', fontsize=12)
    ax1.set_title('Performance by Action Category', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, mean_acc in zip(bars, category_means):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{mean_acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Category Sample Count vs Performance
    for cat_name, cat_data in categories.items():
        sample_counts = cat_data['sample_counts']
        accuracies = cat_data['accuracies']
        ax2.scatter(sample_counts, accuracies, 
                   c=cat_data['color'], s=80, alpha=0.7, 
                   label=cat_name, edgecolors='black')
    
    ax2.set_xlabel('Number of Samples', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Sample Count vs Performance by Category', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    
    # Plot 3: Detailed Class Performance within Categories
    # Create horizontal bar chart for each category
    y_pos = 0
    all_classes = []
    all_accuracies = []
    all_colors = []
    
    for cat_name, cat_data in categories.items():
        for i, (class_name, accuracy) in enumerate(zip(cat_data['classes'], cat_data['accuracies'])):
            all_classes.append(f"{class_name}")
            all_accuracies.append(accuracy)
            all_colors.append(cat_data['color'])
            y_pos += 1
    
    # Sort by accuracy for better visualization
    sorted_indices = np.argsort(all_accuracies)
    sorted_classes = [all_classes[i] for i in sorted_indices]
    sorted_accuracies = [all_accuracies[i] for i in sorted_indices]
    sorted_colors = [all_colors[i] for i in sorted_indices]
    
    bars = ax3.barh(range(len(sorted_classes)), sorted_accuracies, 
                    color=sorted_colors, alpha=0.7, edgecolor='black')
    ax3.set_yticks(range(len(sorted_classes)))
    ax3.set_yticklabels(sorted_classes, fontsize=9)
    ax3.set_xlabel('Accuracy', fontsize=12)
    ax3.set_title('Individual Class Performance (Sorted)', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')
    ax3.set_xlim(0, 1)
    
    # Add value labels on bars
    for i, (bar, acc) in enumerate(zip(bars, sorted_accuracies)):
        width = bar.get_width()
        ax3.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                f'{acc:.2f}', ha='left', va='center', fontsize=8)
    
    # Plot 4: Category Statistics Summary
    # Create a summary table
    summary_data = []
    for cat_name, cat_data in categories.items():
        accuracies = cat_data['accuracies']
        sample_counts = cat_data['sample_counts']
        
        summary_data.append({
            'Category': cat_name,
            'Classes': len(accuracies),
            'Mean Acc': f"{np.mean(accuracies):.3f}",
            'Median Acc': f"{np.median(accuracies):.3f}",
            'Std Acc': f"{np.std(accuracies):.3f}",
            'Total Samples': sum(sample_counts),
            'Best Class': cat_data['classes'][np.argmax(accuracies)],
            'Best Acc': f"{np.max(accuracies):.3f}"
        })
    
    # Create table
    ax4.axis('tight')
    ax4.axis('off')
    
    table_data = [[col for col in summary_data[0].keys()]]
    for row in summary_data:
        table_data.append([row[col] for col in row.keys()])
    
    table = ax4.table(cellText=table_data[1:], colLabels=table_data[0],
                      cellLoc='center', loc='center', colWidths=[0.15, 0.08, 0.1, 0.1, 0.08, 0.12, 0.15, 0.1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style the table
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('lightblue')
        table[(0, i)].set_text_props(weight='bold')
    
    ax4.set_title('Category Performance Summary', fontsize=14, fontweight='bold', pad=20)
    
    # Overall title
    fig.suptitle('UCF101 CNN-RNN Performance by Action Categories', fontsize=16, fontweight='bold', y=0.98)
    
    # Add insights
    insights_text = """Key Insights:
• Sports & Music: Highest performing categories
• Fitness: Most challenging due to subtle movements
• Daily Activities: Variable performance based on complexity
• Sample count doesn't strongly correlate with performance"""
    
    fig.text(0.02, 0.02, insights_text, fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8),
             verticalalignment='bottom')
    
    plt.tight_layout()
    plt.savefig('category_performance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Category performance plots saved as 'category_performance.png'")

if __name__ == "__main__":
    plot_category_performance() 
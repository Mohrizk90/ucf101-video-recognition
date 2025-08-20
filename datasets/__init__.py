"""
Datasets package for UCF101 CNN-RNN project.
"""

from .ucf101 import UCF101Dataset, create_ucf101_dataloaders
 
__all__ = ['UCF101Dataset', 'create_ucf101_dataloaders'] 
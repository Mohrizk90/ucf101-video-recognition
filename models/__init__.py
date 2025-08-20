"""
Models package for UCF101 CNN-RNN project.
"""

from .cnn_rnn import CNNRNN, Backbone2D, TemporalHead, create_model, count_model_parameters
 
__all__ = ['CNNRNN', 'Backbone2D', 'TemporalHead', 'create_model', 'count_model_parameters'] 
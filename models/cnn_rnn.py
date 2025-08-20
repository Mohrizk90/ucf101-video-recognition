"""
CNN-RNN model architecture for UCF101 video action recognition.

This model combines a 2D CNN backbone (ResNet) with a bidirectional LSTM
for temporal modeling, providing an efficient alternative to 3D CNNs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import List, Optional, Tuple, Dict


class Backbone2D(nn.Module):
    """
    2D CNN backbone wrapper for video frames.
    
    This module applies a 2D CNN to each frame independently and
    outputs temporal features for the LSTM.
    """
    
    def __init__(self, backbone_name: str = "resnet18", 
                 finetune_layers: List[str] = None,
                 train_bn: bool = True):
        """
        Initialize backbone.
        
        Args:
            backbone_name: Name of backbone architecture
            finetune_layers: List of layer names to finetune
            train_bn: Whether to train batch normalization layers
        """
        super().__init__()
        
        # Load pre-trained backbone
        if backbone_name == "resnet18":
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            self.feature_dim = 512
        elif backbone_name == "resnet34":
            self.backbone = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
            self.feature_dim = 512
        elif backbone_name == "resnet50":
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            self.feature_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        
        # Remove final classification layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Set finetuning policy
        self.finetune_layers = finetune_layers or []
        self._set_finetuning_policy(train_bn)
        
    def _set_finetuning_policy(self, train_bn: bool):
        """
        Set which layers to finetune.
        
        Args:
            train_bn: Whether to train batch normalization layers
        """
        # Freeze all parameters initially
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Unfreeze specified layers
        for name, module in self.backbone.named_modules():
            if any(layer_name in name for layer_name in self.finetune_layers):
                for param in module.parameters():
                    param.requires_grad = True
                    
        # Handle batch normalization layers
        if not train_bn:
            for module in self.backbone.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
                    module.requires_grad_(False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through backbone.
        
        Args:
            x: Input tensor [B, C, T, H, W]
            
        Returns:
            Feature tensor [B, T, feature_dim]
        """
        batch_size, channels, time_steps, height, width = x.size()
        
        # Reshape to process all frames at once
        x = x.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W]
        x = x.contiguous().view(batch_size * time_steps, channels, height, width)
        
        # Extract features
        features = self.backbone(x)  # [B*T, feature_dim, 1, 1]
        features = features.squeeze(-1).squeeze(-1)  # [B*T, feature_dim]
        
        # Reshape back to temporal format
        features = features.view(batch_size, time_steps, self.feature_dim)
        
        return features


class TemporalAttention(nn.Module):
    """
    Temporal attention mechanism for video sequences.
    
    This module implements a simple dot-product attention over
    temporal features to weight the importance of each frame.
    """
    
    def __init__(self, feature_dim: int, attention_dim: int = 256):
        """
        Initialize attention module.
        
        Args:
            feature_dim: Dimension of input features
            attention_dim: Dimension of attention space
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.attention_dim = attention_dim
        
        # Attention projection layers
        self.query_proj = nn.Linear(feature_dim, attention_dim)
        self.key_proj = nn.Linear(feature_dim, attention_dim)
        self.value_proj = nn.Linear(feature_dim, feature_dim)
        
        # Output projection
        self.output_proj = nn.Linear(feature_dim, feature_dim)
        
        # Scale factor for dot product attention
        self.scale = attention_dim ** -0.5
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply temporal attention.
        
        Args:
            x: Input features [B, T, feature_dim]
            
        Returns:
            Tuple of (attended_features, attention_weights)
        """
        batch_size, time_steps, feature_dim = x.size()
        
        # Project to query, key, value
        queries = self.query_proj(x)  # [B, T, attention_dim]
        keys = self.key_proj(x)       # [B, T, attention_dim]
        values = self.value_proj(x)   # [B, T, feature_dim]
        
        # Compute attention scores
        attention_scores = torch.bmm(queries, keys.transpose(1, 2)) * self.scale  # [B, T, T]
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Apply attention to values
        attended_values = torch.bmm(attention_weights, values)  # [B, T, feature_dim]
        
        # Residual connection and output projection
        output = self.output_proj(attended_values + x)
        
        return output, attention_weights


class TemporalHead(nn.Module):
    """
    Temporal modeling head with LSTM and attention.
    
    This module processes the temporal sequence of features from
    the backbone using a bidirectional LSTM and attention.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1,
                 bidirectional: bool = True, dropout: float = 0.3,
                 use_attention: bool = True):
        """
        Initialize temporal head.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            bidirectional: Whether to use bidirectional LSTM
            dropout: Dropout rate
            use_attention: Whether to use attention mechanism
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism
        if use_attention:
            lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
            self.attention = TemporalAttention(lstm_output_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through temporal head.
        
        Args:
            x: Input features [B, T, input_dim]
            
        Returns:
            Temporal features [B, T, hidden_dim*2] or [B, hidden_dim*2]
        """
        # LSTM processing
        lstm_out, _ = self.lstm(x)  # [B, T, hidden_dim*2]
        
        if self.use_attention:
            # Apply attention
            attended_features, _ = self.attention(lstm_out)
            
            # Global average pooling over time
            output = attended_features.mean(dim=1)  # [B, hidden_dim*2]
        else:
            # Simple mean pooling over time
            output = lstm_out.mean(dim=1)  # [B, hidden_dim*2]
        
        # Apply dropout
        output = self.dropout(output)
        
        return output


class CNNRNN(nn.Module):
    """
    Complete CNN-RNN model for video action recognition.
    
    This model combines a 2D CNN backbone with a temporal LSTM head
    for efficient video understanding.
    """
    
    def __init__(self, num_classes: int = 101, backbone_name: str = "resnet18",
                 finetune_layers: List[str] = None, lstm_hidden: int = 256,
                 lstm_layers: int = 1, bidirectional: bool = True,
                 dropout: float = 0.3, use_attention: bool = True):
        """
        Initialize CNN-RNN model.
        
        Args:
            num_classes: Number of action classes
            backbone_name: Name of backbone architecture
            finetune_layers: List of layer names to finetune
            lstm_hidden: LSTM hidden dimension
            lstm_layers: Number of LSTM layers
            bidirectional: Whether to use bidirectional LSTM
            dropout: Dropout rate
            use_attention: Whether to use attention mechanism
        """
        super().__init__()
        
        # Backbone
        self.backbone = Backbone2D(
            backbone_name=backbone_name,
            finetune_layers=finetune_layers,
            train_bn=True
        )
        
        # Temporal head
        self.temporal_head = TemporalHead(
            input_dim=self.backbone.feature_dim,
            hidden_dim=lstm_hidden,
            num_layers=lstm_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            use_attention=use_attention
        )
        
        # Classification head
        lstm_output_dim = lstm_hidden * 2 if bidirectional else lstm_hidden
        self.classifier = nn.Linear(lstm_output_dim, num_classes)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize model weights."""
        # Initialize LSTM and attention weights
        for module in [self.temporal_head, self.classifier]:
            if hasattr(module, 'weight'):
                nn.init.xavier_uniform_(module.weight)
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input video tensor [B, C, T, H, W]
            
        Returns:
            Classification logits [B, num_classes]
        """
        # Extract spatial features
        spatial_features = self.backbone(x)  # [B, T, feature_dim]
        
        # Temporal modeling
        temporal_features = self.temporal_head(spatial_features)  # [B, hidden_dim*2]
        
        # Classification
        logits = self.classifier(temporal_features)  # [B, num_classes]
        
        return logits
    
    def get_feature_maps(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get intermediate feature maps for visualization.
        
        Args:
            x: Input video tensor [B, C, T, H, W]
            
        Returns:
            Dictionary of feature maps
        """
        features = {}
        
        # Spatial features
        spatial_features = self.backbone(x)
        features['spatial'] = spatial_features
        
        # Temporal features
        temporal_features = self.temporal_head(spatial_features)
        features['temporal'] = temporal_features
        
        return features
    
    def unfreeze_backbone_layers(self, layer_names: List[str]):
        """
        Unfreeze specific backbone layers for finetuning.
        
        Args:
            layer_names: List of layer names to unfreeze
        """
        self.backbone.finetune_layers = layer_names
        self.backbone._set_finetuning_policy(train_bn=True)
        
        print(f"Unfrozen backbone layers: {layer_names}")
    
    def freeze_backbone(self):
        """Freeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("Frozen all backbone parameters")


def create_model(config: dict) -> CNNRNN:
    """
    Create CNN-RNN model from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        CNN-RNN model
    """
    model = CNNRNN(
        num_classes=config['model']['num_classes'],
        backbone_name=config['model']['backbone'],
        finetune_layers=config['model']['finetune_layers'],
        lstm_hidden=config['model']['lstm_hidden'],
        lstm_layers=config['model']['lstm_layers'],
        bidirectional=config['model']['bidirectional'],
        dropout=config['model']['dropout'],
        use_attention=True
    )
    
    return model


def count_model_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count model parameters by component.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts
    """
    total_params = 0
    trainable_params = 0
    
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        if param.requires_grad:
            trainable_params += param_count
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': total_params - trainable_params
    } 
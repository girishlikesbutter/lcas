#!/usr/bin/env python3
"""
Surrogate Neural Network Model
==============================

PyTorch implementation of Multi-Layer Perceptron for satellite face illumination prediction.
Provides fast neural network inference to replace computationally expensive ray tracing.
"""

import logging
import torch
import torch.nn as nn
from typing import Optional

logger = logging.getLogger(__name__)


class SurrogateNet(nn.Module):
    """
    Multi-Layer Perceptron for satellite face illumination prediction.
    
    Predicts lit fractions for satellite conceptual faces given sun vector direction.
    Architecture: 3 hidden layers (64 neurons each) with ReLU, sigmoid output.
    
    The network is designed to learn the mapping from 3D sun vectors to illumination
    fractions, replacing expensive ray tracing computations with fast neural network
    inference during simulation and analysis.
    
    Args:
        input_size: Number of input features (default: 3 for sun vector x,y,z)
        hidden_size: Number of neurons in hidden layers (default: 64)
        output_size: Number of output features (number of conceptual faces)
        
    Raises:
        ValueError: If any size parameter is not a positive integer
        
    Example:
        >>> model = SurrogateNet(input_size=3, output_size=5)
        >>> sun_vectors = torch.randn(10, 3)  # Batch of 10 sun vectors
        >>> predictions = model(sun_vectors)  # Shape: (10, 5)
        >>> assert torch.all(predictions >= 0.0) and torch.all(predictions <= 1.0)
    """
    
    def __init__(self, input_size: int = 3, hidden_size: int = 64, output_size: int = 1):
        """Initialize the SurrogateNet with specified architecture."""
        # PATTERN: Always call parent constructor first
        super().__init__()
        
        # CRITICAL: Validate parameters
        if not isinstance(input_size, int) or input_size <= 0:
            raise ValueError(f"input_size must be positive integer, got {input_size}")
        if not isinstance(hidden_size, int) or hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive integer, got {hidden_size}")
        if not isinstance(output_size, int) or output_size <= 0:
            raise ValueError(f"output_size must be positive integer, got {output_size}")
        
        # PATTERN: Define layers as module attributes for parameter registration
        self.hidden1 = nn.Linear(input_size, hidden_size)
        self.hidden2 = nn.Linear(hidden_size, hidden_size)
        self.hidden3 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        
        # PATTERN: Store architecture info for debugging and compatibility
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # PATTERN: Initialize logger like other LCAS modules
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"SurrogateNet initialized: {input_size}→{hidden_size}→{hidden_size}→{hidden_size}→{output_size}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Processes input sun vectors through the MLP architecture to predict
        illumination fractions for satellite conceptual faces.
        
        Args:
            x: Input tensor of shape (batch_size, input_size) containing sun vectors.
               Sun vectors should be normalized 3D direction vectors.
               
        Returns:
            Output tensor of shape (batch_size, output_size) with lit fractions in [0, 1].
            Each value represents the predicted illumination fraction for a conceptual face.
            
        Raises:
            ValueError: If input tensor has incorrect type or shape
            
        Example:
            >>> model = SurrogateNet(input_size=3, output_size=2)
            >>> x = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
            >>> output = model(x)
            >>> assert output.shape == (2, 2)
            >>> assert torch.all(output >= 0.0) and torch.all(output <= 1.0)
        """
        # CRITICAL: Validate input
        if not isinstance(x, torch.Tensor):
            raise ValueError(f"Input must be torch.Tensor, got {type(x)}")
        
        if x.dim() != 2:
            raise ValueError(f"Input must be 2D tensor (batch_size, input_size), got {x.dim()}D tensor")
        
        if x.size(1) != self.input_size:
            raise ValueError(f"Input must have shape (batch_size, {self.input_size}), got {x.shape}")
        
        # PATTERN: Chain operations through hidden layers with ReLU activations
        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        x = torch.relu(self.hidden3(x))
        
        # CRITICAL: Sigmoid activation for [0, 1] physical constraint
        x = torch.sigmoid(self.output(x))
        
        return x
    
    def get_device(self) -> torch.device:
        """
        Get the device where model parameters are located.
        
        Returns:
            torch.device: The device (CPU or CUDA) where model parameters reside.
            
        Example:
            >>> model = SurrogateNet()
            >>> device = model.get_device()
            >>> print(f"Model is on: {device}")
        """
        return next(self.parameters()).device
    
    def get_architecture_info(self) -> dict:
        """
        Get detailed information about the network architecture.
        
        Returns:
            dict: Architecture details including layer sizes, parameter count, and device.
            
        Example:
            >>> model = SurrogateNet(input_size=3, output_size=5)
            >>> info = model.get_architecture_info()
            >>> print(f"Total parameters: {info['total_parameters']}")
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'output_size': self.output_size,
            'architecture': f"{self.input_size}→{self.hidden_size}→{self.hidden_size}→{self.hidden_size}→{self.output_size}",
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'device': str(self.get_device()),
            'layer_details': {
                'hidden1': f"Linear({self.hidden1.in_features}, {self.hidden1.out_features})",
                'hidden2': f"Linear({self.hidden2.in_features}, {self.hidden2.out_features})",
                'hidden3': f"Linear({self.hidden3.in_features}, {self.hidden3.out_features})",
                'output': f"Linear({self.output.in_features}, {self.output.out_features})"
            }
        }
    
    def __repr__(self) -> str:
        """String representation of the model."""
        info = self.get_architecture_info()
        return (f"SurrogateNet(architecture={info['architecture']}, "
                f"parameters={info['total_parameters']}, device={info['device']})")


def create_surrogate_model(num_faces: int, input_size: int = 3, hidden_size: int = 64) -> SurrogateNet:
    """
    Factory function to create a SurrogateNet model for a specific satellite configuration.
    
    Args:
        num_faces: Number of conceptual faces in the satellite model
        input_size: Number of input features (default: 3 for sun vector)
        hidden_size: Number of neurons in hidden layers (default: 64)
        
    Returns:
        SurrogateNet: Configured model ready for training or inference
        
    Example:
        >>> model = create_surrogate_model(num_faces=8)
        >>> print(model.get_architecture_info()['architecture'])
        3→64→64→64→8
    """
    logger.info(f"Creating surrogate model for {num_faces} faces")
    return SurrogateNet(input_size=input_size, hidden_size=hidden_size, output_size=num_faces)


# Global convenience function for model creation with device handling
def create_model_with_device(num_faces: int, device: Optional[torch.device] = None) -> SurrogateNet:
    """
    Create a SurrogateNet model and move it to the specified device.
    
    Follows LCAS patterns for device handling with automatic GPU detection
    and fallback to CPU if GPU is not available.
    
    Args:
        num_faces: Number of conceptual faces in the satellite model
        device: Target device (if None, auto-detect GPU/CPU)
        
    Returns:
        SurrogateNet: Model moved to the appropriate device
        
    Example:
        >>> model = create_model_with_device(num_faces=5)
        >>> print(f"Model device: {model.get_device()}")
    """
    # PATTERN: Follow pytorch_shadow_engine.py device detection
    if device is None:
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            try:
                device = torch.device('cuda')
                # Test GPU allocation
                test_tensor = torch.zeros(1, device=device)
                del test_tensor
                logger.info("Using CUDA device for SurrogateNet")
            except Exception as e:
                logger.warning(f"Failed to use CUDA device: {e}, falling back to CPU")
                device = torch.device('cpu')
        else:
            device = torch.device('cpu')
            logger.info("CUDA not available, using CPU for SurrogateNet")
    
    model = create_surrogate_model(num_faces)
    model = model.to(device)
    
    logger.info(f"SurrogateNet created and moved to {device}")
    return model
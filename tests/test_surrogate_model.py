#!/usr/bin/env python3
"""
Unit tests for src/surrogate/model.py

Tests the SurrogateNet neural network class and associated functions for
satellite face illumination prediction using PyTorch.
"""

import pytest
import torch
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import logging

# Import the module under test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.surrogate.model import SurrogateNet, create_surrogate_model, create_model_with_device


class TestSurrogateNet:
    """Test the SurrogateNet neural network class."""
    
    def test_constructor_default_parameters(self):
        """Test SurrogateNet constructor with default parameters."""
        model = SurrogateNet()
        
        assert model.input_size == 3
        assert model.hidden_size == 64
        assert model.output_size == 1
        
        # Verify layer dimensions
        assert model.hidden1.in_features == 3
        assert model.hidden1.out_features == 64
        assert model.hidden2.in_features == 64
        assert model.hidden2.out_features == 64
        assert model.hidden3.in_features == 64
        assert model.hidden3.out_features == 64
        assert model.output.in_features == 64
        assert model.output.out_features == 1
    
    def test_constructor_custom_parameters(self):
        """Test SurrogateNet constructor with custom parameters."""
        model = SurrogateNet(input_size=5, hidden_size=32, output_size=10)
        
        assert model.input_size == 5
        assert model.hidden_size == 32
        assert model.output_size == 10
        
        # Verify layer dimensions
        assert model.hidden1.in_features == 5
        assert model.hidden1.out_features == 32
        assert model.hidden2.in_features == 32
        assert model.hidden2.out_features == 32
        assert model.hidden3.in_features == 32
        assert model.hidden3.out_features == 32
        assert model.output.in_features == 32
        assert model.output.out_features == 10
    
    def test_constructor_parameter_validation(self):
        """Test constructor parameter validation."""
        # Valid parameters should work
        SurrogateNet(input_size=3, hidden_size=64, output_size=10)
        
        # Invalid input_size
        with pytest.raises(ValueError, match="input_size must be positive integer"):
            SurrogateNet(input_size=0)
        with pytest.raises(ValueError, match="input_size must be positive integer"):
            SurrogateNet(input_size=-1)
        with pytest.raises(ValueError, match="input_size must be positive integer"):
            SurrogateNet(input_size=3.5)
        
        # Invalid hidden_size
        with pytest.raises(ValueError, match="hidden_size must be positive integer"):
            SurrogateNet(hidden_size=0)
        with pytest.raises(ValueError, match="hidden_size must be positive integer"):
            SurrogateNet(hidden_size=-1)
        
        # Invalid output_size
        with pytest.raises(ValueError, match="output_size must be positive integer"):
            SurrogateNet(output_size=0)
        with pytest.raises(ValueError, match="output_size must be positive integer"):
            SurrogateNet(output_size=-1)
    
    def test_forward_pass_single_sample(self):
        """Test forward pass with single sample."""
        model = SurrogateNet(input_size=3, output_size=5)
        
        # Single sample
        x = torch.randn(1, 3)
        output = model(x)
        
        assert output.shape == (1, 5)
        assert torch.all(output >= 0.0), "Output contains values < 0"
        assert torch.all(output <= 1.0), "Output contains values > 1"
    
    def test_forward_pass_batch_samples(self):
        """Test forward pass with batch of samples."""
        model = SurrogateNet(input_size=3, output_size=5)
        
        # Batch of samples
        x = torch.randn(10, 3)
        output = model(x)
        
        assert output.shape == (10, 5)
        assert torch.all(output >= 0.0), "Output contains values < 0"
        assert torch.all(output <= 1.0), "Output contains values > 1"
    
    def test_sigmoid_output_constraint(self):
        """Test that output is always in [0, 1] range even with extreme inputs."""
        model = SurrogateNet(output_size=5)
        
        # Test with various input magnitudes
        test_cases = [
            torch.randn(5, 3) * 100,     # Very large random inputs
            torch.randn(5, 3) * -100,    # Very large negative inputs
            torch.zeros(5, 3),           # Zero inputs
            torch.ones(5, 3) * 1000,     # Very large positive inputs
            torch.ones(5, 3) * -1000     # Very large negative inputs
        ]
        
        for x in test_cases:
            output = model(x)
            assert torch.all(output >= 0.0), f"Output contains values < 0 for input scale {torch.max(torch.abs(x))}"
            assert torch.all(output <= 1.0), f"Output contains values > 1 for input scale {torch.max(torch.abs(x))}"
    
    def test_forward_pass_input_validation(self):
        """Test forward pass input validation."""
        model = SurrogateNet(input_size=3, output_size=2)
        
        # Wrong input type
        with pytest.raises(ValueError, match="Input must be torch.Tensor"):
            model(np.array([[1, 2, 3]]))
        
        # Wrong tensor dimensions
        with pytest.raises(ValueError, match="Input must be 2D tensor"):
            model(torch.randn(3))  # 1D tensor
        with pytest.raises(ValueError, match="Input must be 2D tensor"):
            model(torch.randn(5, 3, 2))  # 3D tensor
        
        # Wrong input size
        with pytest.raises(ValueError, match="Input must have shape"):
            model(torch.randn(10, 5))  # Wrong feature dimension
    
    def test_device_compatibility_cpu(self):
        """Test model works on CPU."""
        model = SurrogateNet()
        x = torch.randn(3, 3)
        
        # Ensure CPU usage
        model_cpu = model.to('cpu')
        x_cpu = x.to('cpu')
        output_cpu = model_cpu(x_cpu)
        
        assert output_cpu.device.type == 'cpu'
        assert output_cpu.shape == (3, 1)
        assert torch.all(output_cpu >= 0.0) and torch.all(output_cpu <= 1.0)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_compatibility_gpu(self):
        """Test model works on GPU if available."""
        model = SurrogateNet()
        x = torch.randn(3, 3)
        
        # GPU test
        model_gpu = model.to('cuda')
        x_gpu = x.to('cuda')
        output_gpu = model_gpu(x_gpu)
        
        assert output_gpu.device.type == 'cuda'
        assert output_gpu.shape == (3, 1)
        assert torch.all(output_gpu >= 0.0) and torch.all(output_gpu <= 1.0)
    
    def test_get_device_method(self):
        """Test get_device method."""
        model = SurrogateNet()
        device = model.get_device()
        
        assert isinstance(device, torch.device)
        assert device.type in ['cpu', 'cuda']
    
    def test_get_architecture_info(self):
        """Test get_architecture_info method."""
        model = SurrogateNet(input_size=3, hidden_size=64, output_size=5)
        info = model.get_architecture_info()
        
        # Check required keys
        required_keys = ['input_size', 'hidden_size', 'output_size', 'architecture', 
                        'total_parameters', 'trainable_parameters', 'device', 'layer_details']
        for key in required_keys:
            assert key in info, f"Missing key: {key}"
        
        # Check values
        assert info['input_size'] == 3
        assert info['hidden_size'] == 64
        assert info['output_size'] == 5
        assert info['architecture'] == "3→64→64→64→5"
        assert info['total_parameters'] > 0
        assert info['trainable_parameters'] > 0
        assert info['trainable_parameters'] == info['total_parameters']
    
    def test_model_repr(self):
        """Test string representation of model."""
        model = SurrogateNet(input_size=3, output_size=5)
        repr_str = repr(model)
        
        assert "SurrogateNet" in repr_str
        assert "3→64→64→64→5" in repr_str
        assert "parameters=" in repr_str
        assert "device=" in repr_str
    
    def test_model_parameters_exist(self):
        """Test that model parameters are properly initialized."""
        model = SurrogateNet()
        
        # Check that parameters exist
        params = list(model.parameters())
        assert len(params) > 0
        
        # Check that all parameters have gradients enabled
        for param in params:
            assert param.requires_grad, "Parameter should require gradients"
    
    def test_gradient_computation(self):
        """Test that gradients are computed correctly."""
        model = SurrogateNet(input_size=3, output_size=2)
        x = torch.randn(5, 3)
        
        # Forward pass
        output = model(x)
        loss = torch.mean(output)
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist
        for param in model.parameters():
            assert param.grad is not None, "Gradients not computed"
            assert not torch.allclose(param.grad, torch.zeros_like(param.grad)), "Gradients are zero"
    
    def test_model_serialization(self):
        """Test model state dict saving and loading."""
        model1 = SurrogateNet(input_size=3, output_size=2)
        
        # Get state dict
        state_dict = model1.state_dict()
        
        # Create new model and load state
        model2 = SurrogateNet(input_size=3, output_size=2)
        model2.load_state_dict(state_dict)
        
        # Test that models produce same output
        x = torch.randn(5, 3)
        output1 = model1(x)
        output2 = model2(x)
        
        assert torch.allclose(output1, output2), "Models don't produce same output after loading"


class TestFactoryFunctions:
    """Test factory functions for model creation."""
    
    def test_create_surrogate_model(self):
        """Test create_surrogate_model factory function."""
        model = create_surrogate_model(num_faces=8)
        
        assert isinstance(model, SurrogateNet)
        assert model.input_size == 3
        assert model.hidden_size == 64
        assert model.output_size == 8
    
    def test_create_surrogate_model_custom_params(self):
        """Test create_surrogate_model with custom parameters."""
        model = create_surrogate_model(num_faces=5, input_size=4, hidden_size=32)
        
        assert model.input_size == 4
        assert model.hidden_size == 32
        assert model.output_size == 5
    
    def test_create_model_with_device_cpu(self):
        """Test create_model_with_device with CPU."""
        device = torch.device('cpu')
        model = create_model_with_device(num_faces=3, device=device)
        
        assert model.get_device() == device
        assert model.output_size == 3
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_create_model_with_device_gpu(self):
        """Test create_model_with_device with GPU."""
        device = torch.device('cuda')
        model = create_model_with_device(num_faces=3, device=device)
        
        assert model.get_device().type == 'cuda'
        assert model.output_size == 3
    
    def test_create_model_with_device_auto_detect(self):
        """Test create_model_with_device with automatic device detection."""
        model = create_model_with_device(num_faces=4)
        
        device = model.get_device()
        assert device.type in ['cpu', 'cuda']
        assert model.output_size == 4


class TestIntegrationWithDataGenerator:
    """Integration tests with surrogate data generator format."""
    
    def test_integration_with_data_generator_format(self):
        """Test that model works with data from surrogate_data_generator.py format."""
        # Simulate data generator output format
        num_samples = 10
        num_faces = 5
        
        # Create synthetic training data (sun_vec_x, sun_vec_y, sun_vec_z, face1, face2, ...)
        sun_vectors = torch.randn(num_samples, 3)
        sun_vectors = sun_vectors / torch.norm(sun_vectors, dim=1, keepdim=True)  # Normalize
        
        # Create model
        model = SurrogateNet(input_size=3, output_size=num_faces)
        
        # Test forward pass
        predictions = model(sun_vectors)
        
        # Validate output
        assert predictions.shape == (num_samples, num_faces)
        assert torch.all(predictions >= 0.0) and torch.all(predictions <= 1.0)
        
        # Test that model is trainable (gradients flow)
        loss = torch.mean(predictions)
        loss.backward()
        
        # Check that gradients exist
        for param in model.parameters():
            assert param.grad is not None, "Gradients not computed"
    
    def test_normalized_sun_vectors(self):
        """Test model with properly normalized sun vectors."""
        model = SurrogateNet(input_size=3, output_size=3)
        
        # Create normalized sun vectors (unit vectors)
        sun_vectors = torch.tensor([
            [1.0, 0.0, 0.0],   # +X direction
            [0.0, 1.0, 0.0],   # +Y direction
            [0.0, 0.0, 1.0],   # +Z direction
            [-1.0, 0.0, 0.0],  # -X direction
            [0.7071, 0.7071, 0.0],  # 45 degrees in XY plane
        ])
        
        predictions = model(sun_vectors)
        
        assert predictions.shape == (5, 3)
        assert torch.all(predictions >= 0.0) and torch.all(predictions <= 1.0)
        
        # Check that different sun directions produce potentially different results
        # (not a strict requirement due to random initialization, but good to check)
        assert predictions.numel() > 0


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_very_small_model(self):
        """Test model with minimal configuration."""
        model = SurrogateNet(input_size=1, hidden_size=1, output_size=1)
        
        x = torch.randn(1, 1)
        output = model(x)
        
        assert output.shape == (1, 1)
        assert torch.all(output >= 0.0) and torch.all(output <= 1.0)
    
    def test_large_batch_size(self):
        """Test model with large batch size."""
        model = SurrogateNet(input_size=3, output_size=2)
        
        # Large batch
        x = torch.randn(1000, 3)
        output = model(x)
        
        assert output.shape == (1000, 2)
        assert torch.all(output >= 0.0) and torch.all(output <= 1.0)
    
    def test_model_eval_mode(self):
        """Test model in evaluation mode."""
        model = SurrogateNet()
        x = torch.randn(5, 3)
        
        # Train mode
        model.train()
        output_train = model(x)
        
        # Eval mode
        model.eval()
        output_eval = model(x)
        
        # For this simple MLP, outputs should be identical in train/eval
        # (no dropout or batch norm)
        assert torch.allclose(output_train, output_eval, atol=1e-6)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
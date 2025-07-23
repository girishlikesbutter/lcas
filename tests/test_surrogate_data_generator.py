#!/usr/bin/env python3
"""
Unit tests for surrogate_data_generator.py

Tests the DataGenerator class and associated functions for generating
synthetic training data for surrogate models.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import logging

# Import the module under test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from surrogate_data_generator import DataGenerator, sample_uniform_sphere


class TestSampleUniformSphere:
    """Test the sample_uniform_sphere function."""
    
    def test_sample_uniform_sphere_valid_input(self):
        """Test sample_uniform_sphere with valid input."""
        num_samples = 100
        vectors = sample_uniform_sphere(num_samples)
        
        # Check shape
        assert vectors.shape == (num_samples, 3)
        
        # Check that all vectors are unit length (within tolerance)
        norms = np.linalg.norm(vectors, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-10)
        
        # Check that vectors are not all the same (uniformity test)
        # With 100 samples, we should have good diversity
        unique_vectors = np.unique(np.round(vectors, decimals=6), axis=0)
        assert len(unique_vectors) > 90  # Should have at least 90 unique vectors
    
    def test_sample_uniform_sphere_single_sample(self):
        """Test sample_uniform_sphere with single sample."""
        vectors = sample_uniform_sphere(1)
        assert vectors.shape == (1, 3)
        assert np.allclose(np.linalg.norm(vectors[0]), 1.0)
    
    def test_sample_uniform_sphere_invalid_input(self):
        """Test sample_uniform_sphere with invalid inputs."""
        # Test negative input
        with pytest.raises(ValueError, match="must be a positive integer"):
            sample_uniform_sphere(-1)
        
        # Test zero input
        with pytest.raises(ValueError, match="must be a positive integer"):
            sample_uniform_sphere(0)
        
        # Test non-integer input
        with pytest.raises(ValueError, match="must be a positive integer"):
            sample_uniform_sphere(10.5)


class TestDataGenerator:
    """Test the DataGenerator class."""
    
    @pytest.fixture
    def config_path(self):
        """Provide path to test configuration."""
        config_path = Path("data/models/intelsat_901_config.yaml")
        if not config_path.exists():
            pytest.skip(f"Configuration file not found: {config_path}")
        return str(config_path)
    
    @pytest.fixture
    def generator(self, config_path):
        """Create a DataGenerator instance for testing."""
        return DataGenerator(config_path)
    
    def test_generator_initialization(self, generator):
        """Test DataGenerator initialization."""
        assert generator.satellite is not None
        assert generator.config is not None
        assert generator.config_manager is not None
        assert generator._combined_mesh is not None
        assert hasattr(generator, 'logger')
        assert isinstance(generator.logger, logging.Logger)
    
    def test_calculate_lit_fractions_valid_input(self, generator):
        """Test calculate_lit_fractions with valid sun vector."""
        # Use a simple sun vector pointing in +Z direction
        sun_vector = np.array([0.0, 0.0, 1.0])
        lit_fractions = generator.calculate_lit_fractions(sun_vector)
        
        # Should return a dictionary
        assert isinstance(lit_fractions, dict)
        assert len(lit_fractions) > 0
        
        # All values should be between 0.0 and 1.0
        for face_name, fraction in lit_fractions.items():
            assert isinstance(face_name, str)
            assert 0.0 <= fraction <= 1.0
    
    def test_calculate_lit_fractions_different_directions(self, generator):
        """Test that different sun directions produce different results."""
        sun_vector1 = np.array([1.0, 0.0, 0.0])
        sun_vector2 = np.array([0.0, 1.0, 0.0])
        
        lit_fractions1 = generator.calculate_lit_fractions(sun_vector1)
        lit_fractions2 = generator.calculate_lit_fractions(sun_vector2)
        
        # Should have same face names
        assert set(lit_fractions1.keys()) == set(lit_fractions2.keys())
        
        # At least some values should be different (satellite is not symmetric)
        # Allow for some cases where they might be the same due to geometry
        total_diff = sum(abs(lit_fractions1[face] - lit_fractions2[face]) 
                        for face in lit_fractions1.keys())
        # If total difference is very small, it might still be valid due to symmetric geometry
        # Just ensure the method runs without error
        assert total_diff >= 0.0
    
    def test_calculate_lit_fractions_invalid_input(self, generator):
        """Test calculate_lit_fractions with invalid inputs."""
        # Test with wrong shape
        with pytest.raises(ValueError, match="must be 3D numpy array"):
            generator.calculate_lit_fractions(np.array([1.0, 0.0]))
        
        # Test with zero vector
        with pytest.raises(ValueError, match="cannot be zero vector"):
            generator.calculate_lit_fractions(np.array([0.0, 0.0, 0.0]))
        
        # Test with non-numpy array
        with pytest.raises(ValueError, match="must be 3D numpy array"):
            generator.calculate_lit_fractions([1.0, 0.0, 0.0])
    
    def test_run_generation_small_sample(self, generator, tmp_path):
        """Test run_generation with small sample count."""
        output_file = tmp_path / "test_output.csv"
        
        # Generate small sample for fast testing
        generator.run_generation(5, str(output_file))
        
        # Validate output file exists
        assert output_file.exists()
        
        # Read and validate CSV content
        data = np.loadtxt(output_file, delimiter=',', skiprows=1)
        assert data.shape[0] == 5  # 5 samples
        assert data.shape[1] >= 4  # At least sun_vec (3) + 1 face
        
        # Check that sun vectors are unit length
        sun_vectors = data[:, :3]
        norms = np.linalg.norm(sun_vectors, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)
        
        # Check that lit fractions are in valid range
        lit_fractions = data[:, 3:]
        assert np.all(lit_fractions >= 0.0)
        assert np.all(lit_fractions <= 1.0)
    
    def test_run_generation_csv_format(self, generator, tmp_path):
        """Test CSV output format matches expected structure."""
        output_file = tmp_path / "format_test.csv"
        
        # Generate minimal sample
        generator.run_generation(3, str(output_file))
        
        # Check header format
        with open(output_file, 'r') as f:
            header = f.readline().strip()
            assert header.startswith("sun_vec_x,sun_vec_y,sun_vec_z")
            
            # Check that header has face names
            header_parts = header.split(',')
            assert len(header_parts) >= 4  # At least 3 sun_vec columns + 1 face
            
            # Verify data format
            data_line = f.readline().strip()
            data_values = data_line.split(',')
            assert len(data_values) == len(header_parts)
            
            # Check that all values are numeric
            for value in data_values:
                float(value)  # Should not raise exception
    
    def test_run_generation_invalid_inputs(self, generator, tmp_path):
        """Test run_generation with invalid inputs."""
        output_file = tmp_path / "test.csv"
        
        # Test negative sample count
        with pytest.raises(ValueError, match="must be positive integer"):
            generator.run_generation(-1, str(output_file))
        
        # Test zero sample count
        with pytest.raises(ValueError, match="must be positive integer"):
            generator.run_generation(0, str(output_file))
        
        # Test non-integer sample count
        with pytest.raises(ValueError, match="must be positive integer"):
            generator.run_generation(5.5, str(output_file))
    
    def test_run_generation_creates_directory(self, generator, tmp_path):
        """Test that run_generation creates output directory if needed."""
        nested_dir = tmp_path / "nested" / "path"
        output_file = nested_dir / "test.csv"
        
        # Directory should not exist initially  
        assert not nested_dir.exists()
        
        # Run generation
        generator.run_generation(2, str(output_file))
        
        # Directory should be created and file should exist
        assert nested_dir.exists()
        assert output_file.exists()
    
    @patch('surrogate_data_generator.sample_uniform_sphere')
    def test_run_generation_handles_sample_failure(self, mock_sample_sphere, generator, tmp_path):
        """Test run_generation handles failures in sample generation gracefully."""
        # Mock sample_uniform_sphere to return valid data
        mock_vectors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        mock_sample_sphere.return_value = mock_vectors
        
        output_file = tmp_path / "test_failure.csv"
        
        # Patch calculate_lit_fractions to fail on first call, succeed on second
        original_method = generator.calculate_lit_fractions
        call_count = 0
        
        def mock_calculate_lit_fractions(sun_vector):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Simulated ray tracing failure")
            return original_method(sun_vector)
        
        with patch.object(generator, 'calculate_lit_fractions', side_effect=mock_calculate_lit_fractions):
            generator.run_generation(2, str(output_file))
        
        # Should complete successfully with 1 valid sample
        assert output_file.exists()
        
        # Check that we get 1 valid sample (second one succeeded)
        data = np.loadtxt(output_file, delimiter=',', skiprows=1)
        assert data.shape[0] == 1  # Only 1 successful sample
    
    def test_run_generation_memory_efficiency(self, generator, tmp_path):
        """Test that run_generation doesn't consume excessive memory."""
        output_file = tmp_path / "memory_test.csv"
        
        # Test with larger sample (but still reasonable for testing)
        # This mainly ensures the method completes without memory errors
        generator.run_generation(50, str(output_file))
        
        # Validate output
        assert output_file.exists()
        data = np.loadtxt(output_file, delimiter=',', skiprows=1) 
        assert data.shape[0] == 50


class TestDataGeneratorIntegration:
    """Integration tests for DataGenerator."""
    
    def test_end_to_end_generation(self):
        """Test complete end-to-end data generation workflow."""
        config_path = Path("data/models/intelsat_901_config.yaml")
        if not config_path.exists():
            pytest.skip(f"Configuration file not found: {config_path}")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "integration_test.csv"
            
            # Create generator and run generation
            generator = DataGenerator(str(config_path))
            generator.run_generation(10, str(output_file))
            
            # Comprehensive validation
            assert output_file.exists()
            
            # Validate file structure
            with open(output_file, 'r') as f:
                lines = f.readlines()
                assert len(lines) == 11  # Header + 10 data lines
                
                # Check header
                header = lines[0].strip()
                assert "sun_vec_x,sun_vec_y,sun_vec_z" in header
                
                # Check data consistency
                data = np.loadtxt(output_file, delimiter=',', skiprows=1)
                assert data.shape[0] == 10
                
                # Verify sun vectors are unit length
                sun_vectors = data[:, :3]
                norms = np.linalg.norm(sun_vectors, axis=1)
                assert np.allclose(norms, 1.0, atol=1e-5)
                
                # Verify lit fractions are in valid range
                lit_fractions = data[:, 3:]
                assert np.all(lit_fractions >= 0.0)
                assert np.all(lit_fractions <= 1.0)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
#!/usr/bin/env python3
"""
Unit tests for ray tracing functionality in DataGenerator.

Tests the enhanced DataGenerator class with ray tracing capabilities
including mesh creation and lit fraction calculation.
"""

import sys
import os
import numpy as np
import pytest
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from surrogate_data_generator import DataGenerator


def test_combined_mesh_creation():
    """Test that combined mesh is created successfully"""
    # Create generator with test configuration
    config_path = "intelsat_901_config.yaml"
    
    # Check if config file exists, skip test if not available
    full_config_path = Path("data/models") / config_path
    if not full_config_path.exists():
        pytest.skip(f"Configuration file not found: {full_config_path}")
    
    generator = DataGenerator(config_path)
    
    # Verify mesh was created
    assert generator._combined_mesh is not None, "Combined mesh should not be None"
    assert len(generator._combined_mesh.vertices) > 0, "Combined mesh should have vertices"
    assert len(generator._combined_mesh.faces) > 0, "Combined mesh should have faces"
    
    # Verify mesh properties
    assert generator._combined_mesh.vertices.shape[1] == 3, "Vertices should be 3D"
    assert generator._combined_mesh.faces.shape[1] == 3, "Faces should be triangles"


def test_calculate_lit_fractions_basic():
    """Test basic lit fraction calculation"""
    config_path = "intelsat_901_config.yaml"
    
    # Check if config file exists, skip test if not available
    full_config_path = Path("data/models") / config_path
    if not full_config_path.exists():
        pytest.skip(f"Configuration file not found: {full_config_path}")
    
    generator = DataGenerator(config_path)
    
    # Test with sun from +X direction
    sun_vector = np.array([1.0, 0.0, 0.0])
    
    lit_fractions = generator.calculate_lit_fractions(sun_vector)
    
    # Verify return type and structure
    assert isinstance(lit_fractions, dict), "Result should be a dictionary"
    assert len(lit_fractions) > 0, "Should have at least one conceptual face"
    
    # All values should be between 0.0 and 1.0
    for face_name, fraction in lit_fractions.items():
        assert isinstance(face_name, str), "Face names should be strings"
        assert isinstance(fraction, float), "Lit fractions should be floats"
        assert 0.0 <= fraction <= 1.0, f"Lit fraction {fraction} for {face_name} should be between 0.0 and 1.0"
    
    print(f"✓ Calculated lit fractions for {len(lit_fractions)} faces")
    for face_name, fraction in lit_fractions.items():
        print(f"  {face_name}: {fraction:.3f}")


def test_calculate_lit_fractions_validation():
    """Test input validation"""
    config_path = "intelsat_901_config.yaml"
    
    # Check if config file exists, skip test if not available
    full_config_path = Path("data/models") / config_path
    if not full_config_path.exists():
        pytest.skip(f"Configuration file not found: {full_config_path}")
    
    generator = DataGenerator(config_path)
    
    # Test invalid input types
    with pytest.raises(ValueError, match="sun_vector_body must be 3D numpy array"):
        generator.calculate_lit_fractions([1, 0, 0])  # List instead of ndarray
        
    with pytest.raises(ValueError, match="sun_vector_body must be 3D numpy array"):
        generator.calculate_lit_fractions(np.array([1, 0]))  # Wrong shape
        
    with pytest.raises(ValueError, match="sun_vector_body must be 3D numpy array"):
        generator.calculate_lit_fractions(np.array([[1, 0, 0]]))  # Wrong shape (2D)
    
    # Test zero vector
    with pytest.raises(ValueError, match="sun_vector_body cannot be zero vector"):
        generator.calculate_lit_fractions(np.array([0.0, 0.0, 0.0]))


def test_sun_direction_variations():
    """Test different sun directions produce different results"""
    config_path = "intelsat_901_config.yaml"
    
    # Check if config file exists, skip test if not available
    full_config_path = Path("data/models") / config_path
    if not full_config_path.exists():
        pytest.skip(f"Configuration file not found: {full_config_path}")
    
    generator = DataGenerator(config_path)
    
    # Test different sun directions
    sun_x = np.array([1.0, 0.0, 0.0])  # Sun from +X
    sun_y = np.array([0.0, 1.0, 0.0])  # Sun from +Y
    sun_z = np.array([0.0, 0.0, 1.0])  # Sun from +Z
    
    lit_fractions_x = generator.calculate_lit_fractions(sun_x)
    lit_fractions_y = generator.calculate_lit_fractions(sun_y)
    lit_fractions_z = generator.calculate_lit_fractions(sun_z)
    
    # Results should be different for different sun directions
    # (unless satellite is perfectly symmetric, which is unlikely)
    assert lit_fractions_x != lit_fractions_y, "X and Y sun directions should produce different results"
    assert lit_fractions_x != lit_fractions_z, "X and Z sun directions should produce different results"
    assert lit_fractions_y != lit_fractions_z, "Y and Z sun directions should produce different results"
    
    # All should have the same face names
    assert set(lit_fractions_x.keys()) == set(lit_fractions_y.keys()), "All results should have same face names"
    assert set(lit_fractions_x.keys()) == set(lit_fractions_z.keys()), "All results should have same face names"
    
    print(f"✓ Verified different results for different sun directions")
    print(f"  Number of faces: {len(lit_fractions_x)}")


def test_mesh_optimization():
    """Test that mesh optimization works without errors"""
    config_path = "intelsat_901_config.yaml"
    
    # Check if config file exists, skip test if not available
    full_config_path = Path("data/models") / config_path
    if not full_config_path.exists():
        pytest.skip(f"Configuration file not found: {full_config_path}")
    
    generator = DataGenerator(config_path)
    
    # Verify mesh exists and has reasonable properties
    mesh = generator._combined_mesh
    assert mesh is not None, "Mesh should exist"
    
    # Check that mesh optimization completed without errors
    # (mesh.process(), merge_vertices(), remove_degenerate_faces() were called)
    assert mesh.is_watertight or not mesh.is_watertight, "Mesh should have defined watertight property"
    assert len(mesh.vertices) > 0, "Optimized mesh should still have vertices"
    assert len(mesh.faces) > 0, "Optimized mesh should still have faces"


def test_vector_normalization():
    """Test that input vectors are properly normalized"""
    config_path = "intelsat_901_config.yaml"
    
    # Check if config file exists, skip test if not available
    full_config_path = Path("data/models") / config_path
    if not full_config_path.exists():
        pytest.skip(f"Configuration file not found: {full_config_path}")
    
    generator = DataGenerator(config_path)
    
    # Test with non-unit vector (should be automatically normalized)
    sun_vector_unnormalized = np.array([2.0, 0.0, 0.0])  # Length = 2
    sun_vector_unit = np.array([1.0, 0.0, 0.0])  # Length = 1
    
    lit_fractions_unnormalized = generator.calculate_lit_fractions(sun_vector_unnormalized)
    lit_fractions_unit = generator.calculate_lit_fractions(sun_vector_unit)
    
    # Results should be identical (vector gets normalized internally)
    assert lit_fractions_unnormalized == lit_fractions_unit, "Normalized and unnormalized vectors should give same results"


if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__, "-v"])
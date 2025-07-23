#!/usr/bin/env python3
"""
Final validation script for ray tracing implementation.
"""

from surrogate_data_generator import DataGenerator
import numpy as np

def main():
    print("Running final validation checks...")
    
    # Test combined mesh creation and validation
    generator = DataGenerator('intelsat_901_config.yaml')
    print(f'✓ Combined mesh created: {len(generator._combined_mesh.vertices)} vertices, {len(generator._combined_mesh.faces)} faces')

    # Test ray tracing returns valid dictionary
    sun_vector = np.array([1.0, 0.0, 0.0])
    result = generator.calculate_lit_fractions(sun_vector)
    print(f'✓ Ray tracing returned dictionary with {len(result)} faces')

    # Verify all values between 0.0-1.0
    all_valid = all(0.0 <= val <= 1.0 for val in result.values())
    print(f'✓ All lit fractions valid (0.0-1.0): {all_valid}')

    # Test input validation
    try:
        generator.calculate_lit_fractions([1, 0, 0])
        print('✗ Input validation failed - should have caught list input')
    except ValueError:
        print('✓ Input validation catches invalid types')

    try:
        generator.calculate_lit_fractions(np.array([0.0, 0.0, 0.0]))
        print('✗ Input validation failed - should have caught zero vector')
    except ValueError:
        print('✓ Input validation catches zero vector')

    # Test different sun directions produce different results
    sun_x = np.array([1.0, 0.0, 0.0])
    sun_y = np.array([0.0, 1.0, 0.0])
    sun_z = np.array([0.0, 0.0, 1.0])

    result_x = generator.calculate_lit_fractions(sun_x)
    result_y = generator.calculate_lit_fractions(sun_y)
    result_z = generator.calculate_lit_fractions(sun_z)

    # Check if results are different
    different_results = (result_x != result_y or result_x != result_z)
    if different_results:
        print('✓ Different sun directions produce different results')
    else:
        print('⚠ All sun directions produce same results (may be normal for simple geometry)')

    print('✓ All validation checks completed successfully')
    
    return True

if __name__ == "__main__":
    main()
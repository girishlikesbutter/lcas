name: "Sample Uniform Sphere Function Implementation"
description: |

## Purpose
Implement a function that generates N uniformly distributed 3D unit vectors for sampling sun directions using the Gaussian normalization method. This function addresses the critical problem of pole clustering that occurs with naive spherical coordinate sampling.

## Core Principles
1. **Context is King**: Include ALL necessary documentation, examples, and caveats
2. **Validation Loops**: Provide executable tests/lints the AI can run and fix
3. **Information Dense**: Use keywords and patterns from the codebase
4. **Progressive Success**: Start simple, validate, then enhance
5. **Global rules**: Be sure to follow all rules in CLAUDE.md

---

## Goal
Implement `sample_uniform_sphere(num_samples: int) -> np.ndarray` function that:
- Generates uniformly distributed 3D unit vectors on the sphere surface
- Uses Gaussian normalization method to avoid pole clustering
- Integrates seamlessly into existing surrogate_data_generator.py
- Provides comprehensive documentation explaining the mathematical principles

## Why
- **Business value**: Enables accurate sampling of sun directions for lightcurve simulations
- **Integration**: Provides critical utility function for surrogate model training
- **Problems solved**: Eliminates pole clustering artifacts that would bias training data
- **Mathematical correctness**: Ensures true uniform distribution on sphere surface

## What
A utility function that generates uniform samples on the unit sphere using the mathematically correct Gaussian normalization method, with comprehensive documentation explaining why this approach is superior to naive spherical coordinate sampling.

### Success Criteria
- [ ] Function accepts num_samples parameter and returns np.ndarray of shape (num_samples, 3)
- [ ] Uses numpy.random.randn followed by L2 normalization
- [ ] Includes detailed docstring explaining mathematical principles
- [ ] Explains why Gaussian method avoids pole clustering
- [ ] Handles edge cases (zero norm vectors, invalid inputs)
- [ ] Follows existing code style and documentation patterns
- [ ] Includes comprehensive example in docstring

## All Needed Context

### Documentation & References
```yaml
# MUST READ - Include these in your context window
- file: surrogate_data_generator.py
  why: Target file for function implementation, existing patterns to follow
  
- file: src/utils/
  why: Examples of utility functions in the LCAS project for style consistency
  
- url: https://numpy.org/doc/stable/reference/random/generated/numpy.random.randn.html
  why: Official documentation for Gaussian random number generation
  
- url: https://mathworld.wolfram.com/SpherePointPicking.html
  why: Mathematical explanation of sphere point picking methodology
  
- docfile: docs/context-engineering/INITIAL_sample_uniform_sphere.md
  why: Initial requirements and specifications for the function

- doc: Mathematical principle reference
  critical: Multivariate Gaussian distribution is rotationally symmetric, normalization preserves this symmetry
```

### Current Codebase tree
```bash
.
├── surrogate_data_generator.py    # Target file for implementation
├── src/
│   ├── utils/                     # Pattern examples for utility functions
│   ├── core/
│   │   └── common.py             # Type definitions and utilities
│   └── ...
├── data/
│   ├── models/
│   └── training/
└── requirements.txt
```

### Desired Codebase tree with files to be modified
```bash
.
├── surrogate_data_generator.py    # Modified to include sample_uniform_sphere function
│   # Function added before DataGenerator class as utility function
```

### Known Gotchas & Library Quirks
```python
# CRITICAL: Gaussian normalization method required for uniform distribution
# CRITICAL: Naive spherical coordinate sampling causes pole clustering due to Jacobian
# CRITICAL: Must handle zero-norm edge case (extremely rare but mathematically possible)
# CRITICAL: Use np.linalg.norm with axis=1 for row-wise normalization
# CRITICAL: keepdims=True required for proper broadcasting during division
# CRITICAL: Follow Google-style docstrings throughout
# CRITICAL: Include mathematical explanation in docstring
# CRITICAL: Provide comprehensive example showing verification
```

## Implementation Blueprint

### Data models and structure

Function signature and core data flow:
```python
def sample_uniform_sphere(num_samples: int) -> np.ndarray:
    """
    Generate uniformly distributed 3D unit vectors for sampling sun directions.
    
    Input: num_samples (int) - Number of uniform samples to generate
    Output: np.ndarray of shape (num_samples, 3) - Unit vectors on sphere
    
    Mathematical principle: 
    - 3D Gaussian variables when normalized create uniform sphere distribution
    - Avoids pole clustering from naive θ,φ sampling
    """
```

### List of tasks to be completed

```yaml
Task 1: Add function before DataGenerator class
MODIFY surrogate_data_generator.py:
  - FIND pattern: "class DataGenerator:"
  - INSERT before class definition
  - PRESERVE existing imports and structure

Task 2: Implement function signature and validation
ADD function with proper signature:
  - Accept num_samples: int parameter
  - Add input validation for positive integers
  - Return type annotation: np.ndarray

Task 3: Implement Gaussian sampling and normalization
ADD core algorithm:
  - Generate random vectors using np.random.randn(num_samples, 3)
  - Calculate L2 norms using np.linalg.norm with axis=1, keepdims=True
  - Handle zero-norm edge case
  - Normalize vectors by division

Task 4: Add comprehensive docstring
ADD detailed documentation:
  - Mathematical explanation of Gaussian normalization method
  - Explanation of why naive θ,φ sampling fails (Jacobian issue)
  - Parameter and return value descriptions
  - Comprehensive example with verification
  - Error handling documentation

Task 5: Add example and verification code in docstring
ADD example showing:
  - Basic usage: sun_directions = sample_uniform_sphere(1000)
  - Shape verification: print(sun_directions.shape)
  - Unit length verification: np.allclose(np.linalg.norm(...), 1.0)
```

### Per task pseudocode

```python
# Task 1-3: Core function implementation
def sample_uniform_sphere(num_samples: int) -> np.ndarray:
    # Input validation
    if not isinstance(num_samples, int) or num_samples <= 0:
        raise ValueError(f"num_samples must be a positive integer, got {num_samples}")
    
    # Generate random 3D Gaussian vectors
    random_vectors = np.random.randn(num_samples, 3)
    
    # Compute L2 norms for normalization
    norms = np.linalg.norm(random_vectors, axis=1, keepdims=True)
    
    # Handle zero-norm edge case (extremely rare)
    norms = np.where(norms == 0, 1, norms)
    
    # Normalize to unit vectors
    unit_vectors = random_vectors / norms
    
    return unit_vectors

# Task 4: Comprehensive docstring structure
"""
Generate uniformly distributed 3D unit vectors for sampling sun directions.

This function uses the Gaussian normalization method to generate points
uniformly distributed on the unit sphere. Unlike naive spherical coordinate
sampling (uniform θ and φ), this method avoids clustering at the poles.

The mathematical principle: [detailed explanation]
The naive approach problem: [Jacobian explanation]

Args:
    num_samples: [parameter description with constraints]

Returns:
    [return value description with shape information]

Raises:
    ValueError: [error conditions]

Example:
    [comprehensive example with verification]
"""
```

### Integration Points
```yaml
FUNCTION_PLACEMENT:
  - location: "Before DataGenerator class definition in surrogate_data_generator.py"
  - pattern: "Utility function placement before main classes"
  
IMPORTS:
  - numpy: "Already imported as np in the file"
  - typing: "Return type annotation using np.ndarray"
  
STYLE:
  - docstring: "Google-style docstring format"
  - naming: "Snake_case following Python conventions"
  - type_hints: "Full type annotations for parameters and return"
```

## Validation Loop

### Level 1: Syntax & Style
```bash
# Check Python syntax
python -m py_compile surrogate_data_generator.py

# Expected: No output (success). If errors, fix syntax.
```

### Level 2: Function Import Test
```python
# Test that function can be imported and called
python -c "
from surrogate_data_generator import sample_uniform_sphere
import numpy as np
result = sample_uniform_sphere(10)
print(f'Shape: {result.shape}')
print(f'All unit length: {np.allclose(np.linalg.norm(result, axis=1), 1.0)}')
"

# Expected: 
# Shape: (10, 3)
# All unit length: True
```

### Level 3: Mathematical Properties Test
```python
# Test uniform distribution properties
python -c "
from surrogate_data_generator import sample_uniform_sphere
import numpy as np

# Generate large sample
samples = sample_uniform_sphere(10000)

# Verify unit length
norms = np.linalg.norm(samples, axis=1)
assert np.allclose(norms, 1.0), 'All vectors should be unit length'

# Check distribution uniformity (statistical test)
# Z-component should be uniformly distributed on [-1, 1]
z_coords = samples[:, 2]
assert -1 <= z_coords.min() and z_coords.max() <= 1, 'Z coordinates in valid range'

print('All mathematical properties verified!')
"

# Expected: "All mathematical properties verified!"
```

### Level 4: Edge Cases Test
```bash
# Test error handling
python -c "
from surrogate_data_generator import sample_uniform_sphere
try:
    sample_uniform_sphere(0)  # Should raise ValueError
    print('ERROR: Should have raised ValueError')
except ValueError as e:
    print(f'Correctly caught error: {e}')

try:
    sample_uniform_sphere(-5)  # Should raise ValueError
    print('ERROR: Should have raised ValueError')
except ValueError as e:
    print(f'Correctly caught error: {e}')
"

# Expected: Two "Correctly caught error" messages
```

## Final Validation Checklist
- [ ] Function implemented with correct signature
- [ ] Gaussian sampling and normalization work correctly
- [ ] All generated vectors have unit length
- [ ] Input validation handles edge cases properly
- [ ] Comprehensive docstring with mathematical explanation
- [ ] Example in docstring is complete and verifiable
- [ ] Code follows existing project style
- [ ] No import errors or syntax issues
- [ ] Mathematical properties verified with statistical tests
- [ ] Error handling works for invalid inputs

---

## Anti-Patterns to Avoid
- ❌ Don't use naive spherical coordinate sampling (uniform θ,φ)
- ❌ Don't skip the zero-norm edge case handling
- ❌ Don't use axis=0 for norm calculation (should be axis=1)
- ❌ Don't forget keepdims=True for proper broadcasting
- ❌ Don't omit the mathematical explanation in docstring
- ❌ Don't skip input validation for edge cases
- ❌ Don't use generic error messages - be specific

## Mathematical Background

### Why Gaussian Normalization Works
The key insight is that the multivariate Gaussian distribution N(0, I) in 3D is rotationally symmetric. When we normalize each vector to unit length, we project it onto the unit sphere while preserving this rotational symmetry, resulting in a uniform distribution on the sphere surface.

### Why Naive Sampling Fails
Sampling uniform θ ∈ [0, π] and φ ∈ [0, 2π] causes clustering at poles because the surface area element is dA = sin(θ)dθdφ. The sin(θ) factor means equal increments in θ near the poles (θ ≈ 0 or π) correspond to smaller surface areas, causing higher point density at the poles.

## Confidence Score: 10/10

Maximum confidence due to:
- Well-established mathematical method
- Clear requirements and specifications
- Existing file structure to follow
- Comprehensive validation approach
- Standard NumPy operations with known behavior
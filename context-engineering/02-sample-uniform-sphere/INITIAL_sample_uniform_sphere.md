## FEATURE:

Implement a function to generate N uniformly distributed 3D vectors for sampling sun directions in the existing `surrogate_data_generator.py` script. The function should use the Gaussian normalization method to ensure uniform distribution on the unit sphere without clustering at the poles.

The function should:
- Accept the number of samples as input
- Generate random vectors using numpy.random.randn(N, 3)
- Normalize each vector to unit length
- Return a NumPy array of shape (num_samples, 3)
- Include comprehensive documentation explaining the mathematical method

## EXAMPLES:

The following files provide patterns to follow:
- `surrogate_data_generator.py`: The target file where the function will be added
- `src/utils/` directory: Examples of utility functions in the LCAS project
- NumPy usage patterns throughout the codebase

Key patterns to observe:
- Function documentation style (Google docstrings)
- Type hints usage
- NumPy array operations
- Mathematical function implementations

## DOCUMENTATION:

- NumPy random sampling documentation: https://numpy.org/doc/stable/reference/random/generated/numpy.random.randn.html
- Sphere point picking methodology: https://mathworld.wolfram.com/SpherePointPicking.html
- The existing surrogate_data_generator.py file structure

## OTHER CONSIDERATIONS:

- The function name should be `sample_uniform_sphere(num_samples: int) -> np.ndarray`
- Must explain why Gaussian normalization is superior to naive spherical coordinate sampling (avoids pole clustering)
- Should follow the existing code style in surrogate_data_generator.py
- Add appropriate imports if numpy isn't already imported
- The function should be placed logically within the file (likely before the DataGenerator class as a utility function)
- Include mathematical explanation in the docstring about why this method produces uniform distribution
- Mention that naive spherical coordinate sampling (uniform θ and φ) causes clustering at poles due to the Jacobian of the spherical coordinate transformation
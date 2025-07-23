## FEATURE:

Adapt the core ray-tracing logic from the existing pytorch_shadow_engine.py into a simpler function within the DataGenerator class that calculates lit fractions for a single, static sun vector. This will enable efficient shadow calculation for surrogate model training data generation.

The implementation should:
a. Create a single, combined trimesh.Trimesh object representing the entire satellite geometry in the __init__ method and store it as a class attribute for efficiency
b. Add a new method to calculate lit fractions for each conceptual face by ray tracing from the face's facets towards a provided sun_vector_body against the combined satellite mesh
c. Return a dictionary mapping each conceptual face name (string) to its calculated lit fraction (a float between 0.0 and 1.0)

## EXAMPLES:

The following files provide patterns to follow:
- `src/illumination/pytorch_shadow_engine.py`: Contains the existing ray-tracing logic and mesh creation patterns
- `surrogate_data_generator.py`: The target file where the new method will be added to the DataGenerator class
- `generate_lightcurves_pytorch.py`: Shows how pytorch_shadow_engine is used in practice

Key patterns to extract:
- How trimesh.Trimesh objects are created from satellite geometry
- Ray-tracing algorithms for shadow calculation
- Conceptual face handling and lit fraction calculations
- Integration with satellite model structure

## DOCUMENTATION:

- The existing pytorch_shadow_engine.py implementation
- Trimesh library documentation: https://trimesh.org/trimesh.html
- Ray-tracing concepts and shadow calculation methodology
- The satellite model structure and conceptual faces mapping

## OTHER CONSIDERATIONS:

- The method should be added to the DataGenerator class as `calculate_lit_fractions(self, sun_vector_body: np.ndarray) -> Dict[str, float]`
- Must create the combined satellite mesh once in __init__ for efficiency (don't recreate on every call)
- Should handle edge cases like no intersection or degenerate geometry
- Follow existing LCAS coding conventions and error handling patterns
- Use appropriate trimesh operations for ray-casting and intersection detection
- The sun_vector_body should be a 3D unit vector in the satellite body frame
- Return 1.0 for fully lit faces, 0.0 for fully shadowed faces, and values between for partial shadowing
- Include comprehensive logging for debugging shadow calculation issues
- Consider numerical precision issues with ray-mesh intersections
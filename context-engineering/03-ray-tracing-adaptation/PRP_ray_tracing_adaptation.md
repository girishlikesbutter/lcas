name: "Ray Tracing Adaptation for DataGenerator"
description: |

## Purpose
Adapt the core ray-tracing logic from the existing pytorch_shadow_engine.py into a simpler function within the DataGenerator class that calculates lit fractions for a single, static sun vector. This will enable efficient shadow calculation for surrogate model training data generation.

## Core Principles
1. **Context is King**: Include ALL necessary documentation, examples, and caveats
2. **Validation Loops**: Provide executable tests/lints the AI can run and fix
3. **Information Dense**: Use keywords and patterns from the codebase
4. **Progressive Success**: Start simple, validate, then enhance
5. **Global rules**: Be sure to follow all rules in CLAUDE.md

---

## Goal
Enhance the DataGenerator class in `surrogate_data_generator.py` with efficient ray-tracing capabilities by:
- Creating a single, combined trimesh.Trimesh object representing the entire satellite geometry in the __init__ method
- Adding a new method `calculate_lit_fractions` that performs ray tracing from facet centers towards a provided sun_vector_body
- Returning a dictionary mapping each conceptual face name to its calculated lit fraction (0.0 to 1.0)

## Why
- **Business value**: Enables accurate shadow calculation for surrogate model training data without requiring pre-computed database
- **Integration**: Leverages existing ray-tracing infrastructure while simplifying for single-sun-vector use case
- **Problems solved**: Provides real-time shadow computation for arbitrary sun directions during data generation

## What
A new method in the DataGenerator class that:
- Takes a single 3D unit vector representing sun direction in satellite body frame
- Performs ray tracing against a pre-built combined satellite mesh
- Returns lit fractions for all conceptual faces as a dictionary
- Handles edge cases like no intersection or degenerate geometry

### Success Criteria
- [ ] Combined satellite mesh created once in __init__ for efficiency
- [ ] calculate_lit_fractions method successfully performs ray tracing
- [ ] Returns dictionary mapping conceptual face names to lit fractions (0.0-1.0)
- [ ] Handles back-face culling correctly
- [ ] Includes comprehensive logging for debugging
- [ ] Follows existing LCAS coding conventions

## All Needed Context

### Documentation & References
```yaml
# MUST READ - Include these in your context window
- file: src/illumination/pytorch_shadow_engine.py
  why: Reference implementation for ray-tracing logic, mesh creation patterns, and shadow computation algorithms
  critical: Lines 178-256 show create_shadow_mesh pattern, lines 634-681 show single-epoch shadow computation
  
- file: surrogate_data_generator.py
  why: Target file where new method will be added to DataGenerator class
  critical: Current __init__ method structure, logging patterns, and class organization
  
- file: generate_lightcurves_pytorch.py
  why: Shows how pytorch_shadow_engine is used in practice, demonstrates lit fraction calculation patterns
  
- url: https://trimesh.org/trimesh.ray.ray_triangle.html
  why: Official documentation for RayMeshIntersector class and ray casting methods
  critical: intersects_location method signature and return values
  
- url: https://trimesh.org/trimesh.html#trimesh.util.concatenate
  why: Documentation for combining multiple meshes into single mesh object
  
- file: src/models/model_definitions.py
  why: Satellite, Component, and Facet class definitions and structure
  critical: Understanding of facet.vertices, conceptual_faces_map, relative_position/orientation
  
- file: docs/context-engineering/CLAUDE.md
  why: Project conventions for imports, logging, docstrings, and error handling
```

### Current Codebase tree
```bash
.
├── surrogate_data_generator.py        # Target file for enhancement
├── src/
│   ├── illumination/
│   │   └── pytorch_shadow_engine.py   # Source of ray-tracing patterns
│   ├── models/
│   │   └── model_definitions.py       # Satellite model structure
│   ├── io/
│   │   └── model_io.py                # Model loading functions
│   ├── config/
│   │   └── model_config.py            # Configuration management
│   └── utils/
│       └── geometry_utils.py          # Geometry utilities
├── generate_lightcurves_pytorch.py    # Usage example
└── requirements.txt                   # Dependencies (includes trimesh)
```

### Desired Codebase tree with files to be modified
```bash
.
├── surrogate_data_generator.py        # MODIFY: Add _create_combined_satellite_mesh and calculate_lit_fractions methods
```

### Known Gotchas & Library Quirks
```python
# CRITICAL: trimesh.Trimesh requires vertices as np.array with shape (N, 3) and faces as np.array with shape (M, 3)
# CRITICAL: RayMeshIntersector.intersects_location returns (locations, ray_indices, triangle_indices)
# CRITICAL: Back-face culling requires dot product check: np.dot(facet_normal, sun_vector) > 0
# CRITICAL: Ray origins need epsilon offset from facet surface to avoid self-intersection
# CRITICAL: trimesh.util.concatenate expects list of trimesh objects, not empty list
# CRITICAL: Component transformations use quaternion.as_rotation_matrix for orientation
# CRITICAL: conceptual_faces_map maps face names to lists of facet indices within component
# CRITICAL: Facet vertices are in component local frame, need transformation to body frame
# CRITICAL: Some components may have empty facets list - must check before processing
# GOTCHA: import quaternion is required for quaternion operations
# GOTCHA: epsilon value of 1e-2 works well for ray origin offset (from validation code)
```

## Implementation Blueprint

### Data models and structure

Using existing LCAS models - no new models needed:
```python
# Using existing models from LCAS:
# - Satellite with components list
# - Component with facets, conceptual_faces_map, relative_position, relative_orientation
# - Facet with vertices (List[np.ndarray]), normal (np.ndarray)
# - RayMeshIntersector from trimesh for ray-mesh intersection

# Method signature:
def calculate_lit_fractions(self, sun_vector_body: np.ndarray) -> Dict[str, float]:
    """
    Calculate lit fractions for each conceptual face using ray tracing.
    
    Args:
        sun_vector_body: 3D unit vector representing sun direction in satellite body frame
        
    Returns:
        Dictionary mapping conceptual face names to lit fractions (0.0 to 1.0)
    """
```

### List of tasks to be completed to fulfill the PRP in the order they should be completed

```yaml
Task 1: Add mesh creation method to DataGenerator class
MODIFY surrogate_data_generator.py:
  - ADD _create_combined_satellite_mesh method to DataGenerator class
  - PATTERN: Mirror create_shadow_mesh from pytorch_shadow_engine.py lines 178-256
  - MODIFY: Remove articulation logic, use static component transforms only
  - PRESERVE: Component transformation logic (position + orientation)
  - ADD: Error handling for empty components or failed mesh creation

Task 2: Modify DataGenerator.__init__ to create combined mesh
MODIFY surrogate_data_generator.py:
  - FIND: DataGenerator.__init__ method after satellite loading
  - ADD: Call to _create_combined_satellite_mesh and store result
  - ADD: Logging for mesh creation status and statistics
  - PRESERVE: Existing initialization logic and error handling

Task 3: Add ray tracing method to DataGenerator class  
MODIFY surrogate_data_generator.py:
  - ADD calculate_lit_fractions method to DataGenerator class
  - PATTERN: Mirror single-epoch computation from pytorch_shadow_engine.py lines 634-681
  - MODIFY: Simplify for single sun vector instead of multiple epochs
  - PRESERVE: Back-face culling, ray origin epsilon offset, error handling patterns

Task 4: Add comprehensive logging and error handling
MODIFY surrogate_data_generator.py:
  - ADD debug logging for ray tracing statistics (rays cast, hits, misses)
  - ADD warning logging for edge cases (no forward-facing facets, ray tracing errors)
  - PATTERN: Follow existing logging style in DataGenerator class
  - ADD: Input validation for sun_vector_body (unit vector check)

Task 5: Add import statements and dependencies
MODIFY surrogate_data_generator.py:
  - ADD: import quaternion (needed for quaternion operations)
  - ADD: from trimesh.ray.ray_triangle import RayMeshIntersector
  - VERIFY: trimesh import already exists
  - PRESERVE: Existing import organization and style
```

### Per task pseudocode

```python
# Task 1: _create_combined_satellite_mesh method
def _create_combined_satellite_mesh(self) -> Optional[trimesh.Trimesh]:
    """Create combined satellite mesh for shadow calculations."""
    import quaternion  # CRITICAL: Required for quaternion operations
    
    component_meshes = []
    
    for component in self.satellite.components:
        if not component.facets:  # GOTCHA: Some components may be empty
            continue
            
        # Extract vertices and faces from facets (PATTERN from pytorch_shadow_engine.py)
        vertices_list = []
        faces_list = []
        vertex_offset = 0
        
        for facet in component.facets:
            vertices_list.extend(facet.vertices)  # facet.vertices is List[np.ndarray]
            faces_list.append([vertex_offset, vertex_offset + 1, vertex_offset + 2])
            vertex_offset += 3
            
        if not faces_list:  # GOTCHA: Skip if no valid facets
            continue
            
        # Create local component mesh
        local_mesh = trimesh.Trimesh(
            vertices=np.array(vertices_list),
            faces=np.array(faces_list),
            process=False  # CRITICAL: Disable auto-processing for performance
        )
        
        # Apply component transformation to body frame
        transform = np.eye(4)
        transform[:3, :3] = quaternion.as_rotation_matrix(component.relative_orientation)
        transform[:3, 3] = component.relative_position
        
        body_mesh = local_mesh.copy()
        body_mesh.apply_transform(transform)
        component_meshes.append(body_mesh)
    
    if not component_meshes:  # GOTCHA: Handle case with no valid components
        return None
        
    # Combine all meshes (CRITICAL: trimesh.util.concatenate expects non-empty list)
    combined_mesh = trimesh.util.concatenate(component_meshes)
    combined_mesh.process()  # CRITICAL: Process after concatenation for optimization
    
    return combined_mesh

# Task 3: calculate_lit_fractions method
def calculate_lit_fractions(self, sun_vector_body: np.ndarray) -> Dict[str, float]:
    """Calculate lit fractions using ray tracing."""
    import quaternion
    
    # Input validation (PATTERN: Always validate inputs first)
    if self._combined_mesh is None:
        raise RuntimeError("Combined satellite mesh not available")
        
    if not isinstance(sun_vector_body, np.ndarray) or sun_vector_body.shape != (3,):
        raise ValueError("sun_vector_body must be 3D numpy array")
        
    # Normalize to unit vector (GOTCHA: Ensure unit length)
    sun_vector_body = sun_vector_body / np.linalg.norm(sun_vector_body)
    
    # Create ray intersector once for all queries
    ray_intersector = RayMeshIntersector(self._combined_mesh)
    
    lit_fractions = {}
    total_rays_cast = 0
    total_hits = 0
    
    # Process each component and its conceptual faces
    for component in self.satellite.components:
        if not component.facets or not component.conceptual_faces_map:
            continue
            
        # Get component transformation
        comp_rot_matrix = quaternion.as_rotation_matrix(component.relative_orientation)
        comp_pos = component.relative_position
        
        # Process each conceptual face
        for face_name, facet_indices in component.conceptual_faces_map.items():
            lit_count = 0
            forward_facing_count = 0
            
            for facet_idx in facet_indices:
                if facet_idx >= len(component.facets):  # GOTCHA: Index bounds check
                    continue
                    
                facet = component.facets[facet_idx]
                
                # Transform facet normal and centroid to body frame
                facet_normal_body = comp_rot_matrix @ facet.normal
                facet_centroid_local = np.mean(facet.vertices, axis=0)
                facet_center_body = comp_rot_matrix @ facet_centroid_local + comp_pos
                
                # Back-face culling (CRITICAL: Only forward-facing facets can be lit)
                dot_product = np.dot(facet_normal_body, sun_vector_body)
                if dot_product <= 0:
                    continue  # Back-facing facet
                    
                forward_facing_count += 1
                
                # Create ray with epsilon offset (PATTERN from validation code)
                epsilon = 1e-2
                ray_origin = facet_center_body + facet_normal_body * epsilon
                ray_direction = sun_vector_body
                
                # Perform ray tracing
                try:
                    locations, ray_indices, _ = ray_intersector.intersects_location(
                        ray_origin[np.newaxis, :], ray_direction[np.newaxis, :], 
                        multiple_hits=False
                    )
                    
                    total_rays_cast += 1
                    if len(ray_indices) == 0:
                        lit_count += 1  # No intersection, facet is lit
                    else:
                        total_hits += 1  # Intersection found, facet is shadowed
                        
                except Exception as e:
                    self.logger.warning(f"Ray tracing error for {face_name}: {e}")
                    lit_count += 1  # Assume lit on error
            
            # Calculate lit fraction for this conceptual face
            if forward_facing_count > 0:
                lit_fractions[face_name] = lit_count / forward_facing_count
            else:
                lit_fractions[face_name] = 1.0  # Default to fully lit if no forward-facing facets
    
    # Log statistics
    self.logger.info(f"Ray tracing complete: {total_rays_cast} rays cast, "
                    f"{total_hits} hits, {len(lit_fractions)} faces processed")
    
    return lit_fractions
```

### Integration Points
```yaml
MESH_CREATION:
  - pattern: Use existing trimesh.Trimesh and trimesh.util.concatenate
  - storage: Store combined mesh as self._combined_mesh in DataGenerator.__init__
  - optimization: Create once, reuse for all ray tracing calls
  
LOGGING:
  - pattern: Use self.logger from DataGenerator class
  - level: INFO for normal operations, WARNING for errors
  - format: Follow existing LCAS logging style
  
ERROR_HANDLING:
  - pattern: Raise specific exceptions (RuntimeError, ValueError)
  - graceful: Handle empty components, failed ray tracing
  - fallback: Default to lit on errors to prevent data corruption
```

## Validation Loop

### Level 1: Syntax & Style
```bash
# Run these FIRST - fix any errors before proceeding
python -m py_compile surrogate_data_generator.py
python -c "import surrogate_data_generator"

# Expected: No errors. If errors, READ the error and fix.
```

### Level 2: Unit Tests
```python
# CREATE test_ray_tracing.py with these test cases:
def test_combined_mesh_creation():
    """Test that combined mesh is created successfully"""
    generator = DataGenerator("data/models/intelsat_901_model.yaml")
    assert generator._combined_mesh is not None
    assert len(generator._combined_mesh.vertices) > 0
    assert len(generator._combined_mesh.faces) > 0

def test_calculate_lit_fractions_basic():
    """Test basic lit fraction calculation"""
    generator = DataGenerator("data/models/intelsat_901_model.yaml")
    sun_vector = np.array([1.0, 0.0, 0.0])  # Sun from +X direction
    
    lit_fractions = generator.calculate_lit_fractions(sun_vector)
    assert isinstance(lit_fractions, dict)
    assert len(lit_fractions) > 0
    
    # All values should be between 0.0 and 1.0
    for face_name, fraction in lit_fractions.items():
        assert 0.0 <= fraction <= 1.0

def test_calculate_lit_fractions_validation():
    """Test input validation"""
    generator = DataGenerator("data/models/intelsat_901_model.yaml")
    
    # Test invalid input types
    with pytest.raises(ValueError):
        generator.calculate_lit_fractions([1, 0, 0])  # List instead of ndarray
        
    with pytest.raises(ValueError):
        generator.calculate_lit_fractions(np.array([1, 0]))  # Wrong shape

def test_sun_direction_variations():
    """Test different sun directions produce different results"""
    generator = DataGenerator("data/models/intelsat_901_model.yaml")
    
    sun_x = np.array([1.0, 0.0, 0.0])
    sun_y = np.array([0.0, 1.0, 0.0])
    
    lit_fractions_x = generator.calculate_lit_fractions(sun_x)
    lit_fractions_y = generator.calculate_lit_fractions(sun_y)
    
    # Results should be different for different sun directions
    assert lit_fractions_x != lit_fractions_y
```

```bash
# Run and iterate until passing:
python -m pytest test_ray_tracing.py -v
# If failing: Read error, understand root cause, fix code, re-run
```

### Level 3: Integration Test
```bash
# Test the enhanced script with ray tracing
python surrogate_data_generator.py \
    --config data/models/intelsat_901_model.yaml \
    --samples 10 \
    --output data/training/test_ray_tracing.csv

# Expected output should include:
# - "Creating combined satellite mesh for shadow calculations..."
# - "Combined mesh created: X vertices, Y faces"
# - Successful completion without errors

# Test specific functionality in Python REPL:
python3 -c "
import numpy as np
from surrogate_data_generator import DataGenerator

generator = DataGenerator('data/models/intelsat_901_model.yaml')
sun_vector = np.array([1.0, 0.0, 0.0])
lit_fractions = generator.calculate_lit_fractions(sun_vector)
print(f'Calculated lit fractions for {len(lit_fractions)} faces')
for face, fraction in lit_fractions.items():
    print(f'{face}: {fraction:.3f}')
"
```

## Final Validation Checklist
- [ ] Script runs without syntax errors: `python -m py_compile surrogate_data_generator.py`
- [ ] All imports resolve correctly: `python -c "import surrogate_data_generator"`
- [ ] Combined mesh creation works: Check for positive vertex/face counts
- [ ] Ray tracing method returns valid dictionary: All values between 0.0-1.0
- [ ] Input validation catches invalid sun vectors
- [ ] Different sun directions produce different results
- [ ] Logging provides informative output
- [ ] Error cases handled gracefully
- [ ] Code follows LCAS conventions and CLAUDE.md rules

---

## Anti-Patterns to Avoid
- ❌ Don't recreate mesh on every calculate_lit_fractions call - build once in __init__
- ❌ Don't skip back-face culling - only forward-facing facets can be lit
- ❌ Don't forget epsilon offset for ray origins - causes self-intersection
- ❌ Don't assume all components have facets - check for empty lists
- ❌ Don't ignore ray tracing errors - handle gracefully with fallback
- ❌ Don't skip input validation - ensure sun_vector_body is valid
- ❌ Don't use sync operations in async context (not applicable here)
- ❌ Don't hardcode epsilon values without testing - use proven value 1e-2

## Confidence Score: 9/10

High confidence due to:
- Clear reference implementation in pytorch_shadow_engine.py
- Well-documented trimesh library with proven patterns
- Existing satellite model structure is well understood
- Comprehensive validation gates cover critical functionality
- Strong pattern matching from existing codebase

Minor uncertainty only on specific epsilon values and edge case handling, but the validation tests will catch any issues.
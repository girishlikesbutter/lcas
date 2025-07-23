name: "Data Generation Loop: Complete Surrogate Model Training Data Pipeline"
description: |

## Purpose
Complete the implementation of the `run_generation()` method in `surrogate_data_generator.py` to create a full data generation pipeline that produces CSV training data for surrogate models by combining uniform sphere sampling, ray tracing, and progress tracking.

## Core Principles
1. **Context is King**: Include ALL necessary documentation, examples, and caveats
2. **Validation Loops**: Provide executable tests/lints the AI can run and fix
3. **Information Dense**: Use keywords and patterns from the codebase
4. **Progressive Success**: Start simple, validate, then enhance
5. **Global rules**: Be sure to follow all rules in CLAUDE.md

---

## Goal
Complete the `run_generation()` method in `surrogate_data_generator.py` to generate CSV training data for surrogate models. The method should efficiently process thousands of sun vectors, calculate lit fractions using ray tracing, and output structured CSV data with progress tracking.

## Why
- **Business value**: Enables training of surrogate models for fast satellite illumination prediction
- **Integration**: Completes the 4-task Context Engineering workflow for LCAS project
- **Problems solved**: Automates generation of large-scale training datasets for machine learning models

## What
Complete implementation that:
- Uses existing `sample_uniform_sphere()` to generate N uniformly distributed sun vectors
- Iterates through vectors with tqdm progress tracking
- Calls existing `calculate_lit_fractions()` method for each sun vector
- Collects input sun vectors and output lit fractions into structured data
- Outputs CSV file using numpy.savetxt (following existing LCAS patterns)
- Handles memory efficiently for large sample counts
- Provides comprehensive error handling and logging

### Success Criteria
- [ ] `run_generation()` method generates specified number of samples
- [ ] CSV output contains sun vector columns (x,y,z) and all conceptual face lit fractions
- [ ] Progress tracking shows meaningful updates during generation
- [ ] Memory usage remains reasonable for large sample counts (10k+ samples)
- [ ] All validation gates pass (linting, type checking, tests)
- [ ] Integration with existing LCAS patterns and conventions

## All Needed Context

### Documentation & References
```yaml
# MUST READ - Include these in your context window
- url: https://tqdm.github.io/docs/tqdm/
  why: Progress bar implementation patterns and customization options
  
- url: https://numpy.org/doc/stable/reference/generated/numpy.savetxt.html
  why: CSV output formatting with headers and delimiters
  
- url: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.from_dict.html
  why: Alternative data structure approach (though numpy.savetxt preferred)
  
- file: generate_lightcurves_pytorch.py:538-539
  why: Existing CSV output pattern using numpy.savetxt with proper formatting
  
- file: surrogate_data_generator.py:1-200
  why: Current implementation with all foundation components already built
  
- file: src/illumination/pytorch_shadow_engine.py:170-171
  why: Batch processing and progress logging patterns in LCAS codebase

- docfile: context-engineering/04-data-generation-loop/INITIAL_data_generation_loop.md
  why: Specific requirements and considerations for this implementation
```

### Current Codebase tree
```bash
.
├── surrogate_data_generator.py           # Target file - mostly complete
├── generate_lightcurves_pytorch.py       # CSV output patterns
├── src/
│   ├── illumination/
│   │   └── pytorch_shadow_engine.py     # Progress logging patterns
│   ├── config/
│   │   └── model_config.py              # Configuration management
│   └── io/
│       └── model_io.py                  # Model loading utilities
├── requirements.txt                      # Needs tqdm addition
└── tests/
    ├── test_ray_tracing.py              # Testing patterns
    └── final_validation.py              # Validation examples
```

### Desired Implementation Status
```bash
# After implementation:
surrogate_data_generator.py:
├── ✅ Command-line argument parsing (already complete)
├── ✅ Model loading and mesh creation (already complete) 
├── ✅ sample_uniform_sphere() function (already complete)
├── ✅ calculate_lit_fractions() method (already complete)
├── 🎯 run_generation() method (TO IMPLEMENT)
├── 🎯 CSV output functionality (TO IMPLEMENT)
└── 🎯 Progress tracking with tqdm (TO IMPLEMENT)

requirements.txt:
└── 🎯 Add tqdm>=4.60.0 (TO ADD)
```

### Known Gotchas of our codebase & Library Quirks
```python
# CRITICAL: Use numpy.savetxt NOT pandas - follow existing LCAS patterns
# Evidence: generate_lightcurves_pytorch.py:538-539 uses np.savetxt(data_path, data_array, delimiter=',', header=header, fmt='%s', comments='')

# CRITICAL: surrogate_data_generator.py already has complete implementation
# - Model loading, mesh creation, ray tracing ALL done
# - Only missing: run_generation() method implementation

# CRITICAL: Conceptual faces structure is Dict[str, float]
# face_name -> lit_fraction (0.0 to 1.0)
# Must maintain consistent face ordering across all samples

# CRITICAL: Ray tracing method signature is:
# calculate_lit_fractions(self, sun_vector_body: np.ndarray) -> Dict[str, float]

# CRITICAL: Memory management for large datasets
# Don't store all data in memory - process in chunks or use generators

# CRITICAL: Progress tracking patterns from codebase
# Use tqdm for user-facing progress, logging for detailed milestones
# Pattern: if (i + 1) % 50 == 0: logger.info(f"Processed {i + 1}/{total}...")

# CRITICAL: CSV header format must match data columns exactly
# Pattern: "sun_vec_x,sun_vec_y,sun_vec_z,face_1_name,face_2_name,..."

# CRITICAL: Input validation and error handling
# Check for empty/invalid configurations, handle ray tracing failures gracefully
```

## Implementation Blueprint

### Data models and structure

The data collection process needs to handle:
```python
# Input data structure (already implemented)
sun_vectors: np.ndarray  # Shape (num_samples, 3) from sample_uniform_sphere()

# Output data structure (from existing calculate_lit_fractions)  
lit_fractions: Dict[str, float]  # {"face_name": 0.75, "another_face": 0.23, ...}

# Final CSV structure (following generate_lightcurves_pytorch.py pattern)
data_array: np.ndarray  # Shape (num_samples, 3 + num_faces)
# Columns: [sun_vec_x, sun_vec_y, sun_vec_z, face1_lit, face2_lit, ...]

# Header string (following existing pattern)
header: str = "sun_vec_x,sun_vec_y,sun_vec_z,face_1_name,face_2_name,..."
```

### List of tasks to be completed to fulfill the PRP in the order they should be completed

```yaml
Task 1: Add Required Dependencies
MODIFY requirements.txt:
  - FIND pattern: existing numpy/torch dependencies
  - ADD after existing packages: "tqdm>=4.60.0"
  - PRESERVE existing version specifications and formatting

Task 2: Implement Core Data Collection Logic
MODIFY surrogate_data_generator.py run_generation() method:
  - REPLACE placeholder logging with actual implementation
  - PATTERN: Follow existing method signatures and docstring style
  - COLLECT data from sample_uniform_sphere() and calculate_lit_fractions()
  - MAINTAIN memory efficiency for large sample counts

Task 3: Add Progress Tracking
MODIFY surrogate_data_generator.py:
  - IMPORT tqdm at top of file
  - INTEGRATE tqdm wrapper around main generation loop
  - PATTERN: Use descriptive progress messages like LCAS codebase
  - ADD logging milestones every 50-100 iterations

Task 4: Implement CSV Output
MODIFY surrogate_data_generator.py:
  - PATTERN: Follow generate_lightcurves_pytorch.py numpy.savetxt approach
  - CREATE structured data array with numpy.column_stack
  - GENERATE proper CSV header string with face names
  - HANDLE file path creation with pathlib.Path like existing code

Task 5: Add Input Validation and Error Handling
MODIFY surrogate_data_generator.py run_generation() method:
  - VALIDATE num_samples parameter (positive integer)
  - VALIDATE output_path parameter (writable directory)
  - HANDLE calculate_lit_fractions() failures gracefully
  - ADD comprehensive logging for debugging

Task 6: Create Unit Tests
CREATE tests/test_surrogate_data_generator.py:
  - PATTERN: Follow test_ray_tracing.py structure and conventions
  - TEST happy path with small sample count
  - TEST edge cases (zero samples, invalid paths)
  - TEST CSV output format and content validation
  - MOCK external dependencies where appropriate
```

### Per task pseudocode as needed added to each task

```python
# Task 2: Core Data Collection Logic
def run_generation(self, num_samples: int, output_path: str) -> None:
    """Generate training data for surrogate models."""
    # PATTERN: Follow existing LCAS logging style
    logger.info(f"Starting generation of {num_samples} samples...")
    
    # PATTERN: Use existing sample_uniform_sphere function
    sun_vectors = sample_uniform_sphere(num_samples)  # Shape: (num_samples, 3)
    
    # CRITICAL: Get face names for consistent ordering
    # First sample to determine face names structure
    sample_lit_fractions = self.calculate_lit_fractions(sun_vectors[0])
    face_names = list(sample_lit_fractions.keys())  # Consistent ordering
    
    # MEMORY EFFICIENCY: Process in chunks if needed
    data_rows = []
    
    # PATTERN: Use tqdm for progress tracking
    for i, sun_vector in tqdm(enumerate(sun_vectors), 
                              total=num_samples,
                              desc="Generating training data"):
        try:
            # PATTERN: Use existing calculate_lit_fractions
            lit_fractions = self.calculate_lit_fractions(sun_vector)
            
            # STRUCTURE: [sun_x, sun_y, sun_z, face1_lit, face2_lit, ...]
            face_values = [lit_fractions[face_name] for face_name in face_names]
            row = [sun_vector[0], sun_vector[1], sun_vector[2]] + face_values
            data_rows.append(row)
            
            # PATTERN: Logging milestones like pytorch_shadow_engine.py
            if (i + 1) % 50 == 0:
                logger.info(f"  Processed {i + 1}/{num_samples} samples...")
                
        except Exception as e:
            logger.error(f"Failed to process sample {i}: {e}")
            # DECISION: Skip failed samples or fail completely?
            continue
    
    # PATTERN: Create numpy array like generate_lightcurves_pytorch.py
    data_array = np.array(data_rows)
    
    # PATTERN: Create header string for CSV
    header = "sun_vec_x,sun_vec_y,sun_vec_z," + ",".join(face_names)
    
    # PATTERN: Use numpy.savetxt like existing LCAS code
    output_path = Path(output_path)
    np.savetxt(output_path, data_array, delimiter=',', 
               header=header, fmt='%.6f', comments='')
    
    logger.info(f"Generated {len(data_rows)} samples saved to {output_path}")

# Task 6: Unit Tests Structure
class TestSurrogateDataGenerator:
    def test_run_generation_happy_path(self, tmp_path):
        """Test successful data generation with small sample count."""
        # PATTERN: Follow test_ray_tracing.py setup
        config_path = "data/models/intelsat_901_config.yaml"
        if not Path(config_path).exists():
            pytest.skip(f"Config not found: {config_path}")
        
        generator = DataGenerator(config_path)
        output_file = tmp_path / "test_output.csv"
        
        # Test with small sample for speed
        generator.run_generation(10, str(output_file))
        
        # Validate output
        assert output_file.exists()
        data = np.loadtxt(output_file, delimiter=',', skiprows=1)
        assert data.shape[0] == 10  # 10 samples
        assert data.shape[1] >= 4  # At least sun_vec + 1 face
        
    def test_run_generation_csv_format(self, tmp_path):
        """Test CSV output format matches expected structure."""
        # ... implementation following LCAS test patterns
```

### Integration Points
```yaml
DEPENDENCIES:
  - add to: requirements.txt
  - package: "tqdm>=4.60.0"
  - reason: "Progress tracking for data generation loops"
  
IMPORTS:
  - add to: surrogate_data_generator.py
  - pattern: "from tqdm import tqdm"
  - location: "After existing numpy/pathlib imports"
  
CSV_OUTPUT:
  - follow: generate_lightcurves_pytorch.py:538-539
  - pattern: "np.savetxt(data_path, data_array, delimiter=',', header=header, fmt='%s', comments='')"
  - format: "sun_vec_x,sun_vec_y,sun_vec_z,face1,face2,..."
  
LOGGING:
  - follow: existing LCAS patterns with structured messages
  - milestone: Every 50 samples processed
  - levels: INFO for progress, ERROR for failures
```

## Validation Loop

### Level 1: Syntax & Style  
```bash
# Run these FIRST - fix any errors before proceeding
ruff check surrogate_data_generator.py --fix   # Auto-fix style issues
mypy surrogate_data_generator.py               # Type checking
ruff check requirements.txt                    # Check requirements format

# Expected: No errors. If errors, READ the error and fix.
```

### Level 2: Unit Tests for new functionality
```python
# CREATE test_surrogate_data_generator.py with comprehensive test cases:
def test_run_generation_small_sample():
    """Test data generation works with small sample count"""
    generator = DataGenerator("data/models/intelsat_901_config.yaml")
    output_file = "/tmp/test_output.csv"
    generator.run_generation(5, output_file)
    
    # Validate file exists and has correct structure
    assert Path(output_file).exists()
    data = np.loadtxt(output_file, delimiter=',', skiprows=1)
    assert data.shape[0] == 5
    assert data.shape[1] >= 4  # sun_vec (3) + at least 1 face

def test_csv_format_validation():
    """Test CSV output format matches specification"""
    generator = DataGenerator("data/models/intelsat_901_config.yaml")
    output_file = "/tmp/format_test.csv"
    generator.run_generation(3, output_file)
    
    # Check header format
    with open(output_file, 'r') as f:
        header = f.readline().strip()
        assert header.startswith("sun_vec_x,sun_vec_y,sun_vec_z")
        
def test_progress_tracking():
    """Test tqdm progress tracking works"""
    # Use mock or capture output to verify tqdm is called
    generator = DataGenerator("data/models/intelsat_901_config.yaml")
    # ... test implementation

def test_error_handling():
    """Test graceful handling of invalid inputs"""
    generator = DataGenerator("data/models/intelsat_901_config.yaml")
    
    # Test invalid sample count
    with pytest.raises(ValueError):
        generator.run_generation(-1, "/tmp/test.csv")
        
    # Test invalid output path
    with pytest.raises(PermissionError):
        generator.run_generation(10, "/root/protected.csv")
```

```bash
# Run tests iteratively until passing:
pytest tests/test_surrogate_data_generator.py -v
# If failing: Debug specific test, understand root cause, fix code, re-run
```

### Level 3: Integration Test
```bash
# Test full pipeline with realistic data
python surrogate_data_generator.py \
  --config data/models/intelsat_901_config.yaml \
  --samples 100 \
  --output data/test_output.csv

# Expected output:
# INFO - Starting generation of 100 samples...
# [Progress bar showing completion]
# INFO - Processed 50/100 samples...
# INFO - Processed 100/100 samples...
# INFO - Generated 100 samples saved to data/test_output.csv

# Validate output file
head -5 data/test_output.csv
# Expected format:
# sun_vec_x,sun_vec_y,sun_vec_z,face_name_1,face_name_2,...
# 0.123456,-0.789012,0.456789,0.750000,0.250000,...
# ...

# Check file size and content
wc -l data/test_output.csv  # Should be 101 lines (header + 100 data)
```

## Final Validation Checklist
- [ ] All tests pass: `pytest tests/test_surrogate_data_generator.py -v`
- [ ] No linting errors: `ruff check surrogate_data_generator.py`
- [ ] No type errors: `mypy surrogate_data_generator.py`
- [ ] Integration test produces valid CSV: `python surrogate_data_generator.py --samples 100`
- [ ] Progress tracking displays correctly during generation
- [ ] CSV format matches specification (header + data columns)
- [ ] Memory usage reasonable for large sample counts (test with 1000+ samples)
- [ ] Error cases handled gracefully (invalid inputs, file permission errors)
- [ ] Logging provides useful information for debugging
- [ ] Requirements.txt updated with tqdm dependency

---

## Anti-Patterns to Avoid
- ❌ Don't use pandas for CSV output - follow existing numpy.savetxt pattern
- ❌ Don't store all data in memory - process efficiently for large sample counts  
- ❌ Don't skip progress tracking - users need feedback for long-running operations
- ❌ Don't ignore existing calculate_lit_fractions() method - it's already complete
- ❌ Don't create new mesh creation logic - use existing _combined_mesh
- ❌ Don't hardcode face names - extract dynamically from lit_fractions dict
- ❌ Don't skip input validation - handle edge cases gracefully
- ❌ Don't forget logging patterns - follow existing LCAS conventions

## Confidence Score: 9/10

High confidence due to:
- **Solid foundation**: 90% of surrogate_data_generator.py already implemented
- **Clear patterns**: Existing CSV output and progress tracking examples in codebase
- **Well-defined scope**: Single method implementation with clear requirements
- **Comprehensive validation**: Multiple testing levels with specific success criteria
- **Established conventions**: Clear LCAS patterns to follow for consistency

Minor uncertainty around:
- Optimal memory management strategy for very large sample counts (10k+)
- Specific face naming conventions from conceptual_faces_map structure

The implementation should succeed in one pass due to complete context and clear validation gates.
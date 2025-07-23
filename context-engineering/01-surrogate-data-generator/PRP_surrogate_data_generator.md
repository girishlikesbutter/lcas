name: "Surrogate Data Generator Script"
description: |

## Purpose
Create a command-line script for generating synthetic training data for surrogate models using the LCAS framework. The script will provide a boilerplate DataGenerator class that loads satellite models and generates data samples.

## Core Principles
1. **Context is King**: Include ALL necessary documentation, examples, and caveats
2. **Validation Loops**: Provide executable tests/lints the AI can run and fix
3. **Information Dense**: Use keywords and patterns from the codebase
4. **Progressive Success**: Start simple, validate, then enhance
5. **Global rules**: Be sure to follow all rules in CLAUDE.md

---

## Goal
Create a production-ready script `surrogate_data_generator.py` that:
- Uses argparse for command-line interface
- Loads satellite models using ModelConfigManager
- Provides a DataGenerator class with placeholder generation logic
- Follows existing LCAS project patterns and conventions

## Why
- **Business value**: Enables generation of synthetic training data for machine learning surrogate models
- **Integration**: Leverages existing LCAS infrastructure for model loading and configuration
- **Problems solved**: Provides a standardized way to generate training datasets from satellite models

## What
A command-line script that:
- Accepts configuration file path, number of samples, and output path as arguments
- Loads satellite models using the existing configuration system
- Creates a DataGenerator class that encapsulates the generation logic
- Outputs status messages and error handling

### Success Criteria
- [ ] Script accepts command-line arguments: --config, --samples, --output
- [ ] Successfully loads satellite model using ModelConfigManager
- [ ] DataGenerator class properly initialized with loaded model
- [ ] Placeholder run_generation method prints appropriate message
- [ ] Proper error handling for missing files or invalid arguments
- [ ] Follows LCAS coding conventions and patterns

## All Needed Context

### Documentation & References
```yaml
# MUST READ - Include these in your context window
- file: generate_lightcurves_pytorch.py
  why: Reference implementation showing argparse usage, model loading, project structure
  
- file: src/config/model_config.py
  why: ModelConfigManager implementation and usage patterns
  
- file: src/io/model_io.py
  why: load_satellite_from_yaml function signature and behavior
  
- url: https://docs.python.org/3/library/argparse.html
  why: Official argparse documentation for command-line parsing
  
- file: src/core/common.py
  why: Core type definitions and common utilities

- doc: CLAUDE.md rules
  critical: Follow project conventions, use python_dotenv, proper imports
```

### Current Codebase tree
```bash
.
├── generate_lightcurves_pytorch.py
├── src/
│   ├── config/
│   │   ├── model_config.py
│   │   └── config_schemas.py
│   ├── io/
│   │   └── model_io.py
│   ├── core/
│   │   └── common.py
│   └── ...
├── data/
│   ├── models/
│   │   └── intelsat_901_model.yaml
│   └── training/
└── requirements.txt
```

### Desired Codebase tree with files to be added
```bash
.
├── surrogate_data_generator.py    # New script at project root
├── data/
│   └── training/                  # Output directory for generated data
│       └── is901_data.csv         # Example output file
```

### Known Gotchas & Library Quirks
```python
# CRITICAL: Project uses absolute imports from src
# CRITICAL: Must add PROJECT_ROOT to sys.path before imports
# CRITICAL: load_satellite_from_yaml returns None on failure, not exception
# CRITICAL: Use pathlib.Path for all path operations
# CRITICAL: ModelConfigManager expects paths relative to project root
# CRITICAL: Follow Google-style docstrings throughout
# CRITICAL: Use logging instead of print for production code
```

## Implementation Blueprint

### Data models and structure

The script will use existing LCAS models and add minimal new structures:
```python
# Using existing models from LCAS:
# - RSO_Config from src.config.config_schemas
# - Component/Facet from loaded satellite model
# - ModelConfigManager for configuration handling

# New class structure:
class DataGenerator:
    """Generates synthetic training data from satellite models."""
    
    def __init__(self, config_path: str):
        """Initialize with model configuration."""
        # Load config using ModelConfigManager
        # Load satellite model
        # Store as instance variables
    
    def run_generation(self, num_samples: int, output_path: str):
        """Generate training data samples."""
        # Placeholder implementation
```

### List of tasks to be completed

```yaml
Task 1: Create script boilerplate and imports
CREATE surrogate_data_generator.py:
  - Add shebang line and module docstring
  - Setup PROJECT_ROOT and sys.path
  - Import required modules (argparse, pathlib, logging, etc.)
  - Import LCAS modules (ModelConfigManager, load_satellite_from_yaml)

Task 2: Implement argument parsing
ADD to surrogate_data_generator.py:
  - Create ArgumentParser with description
  - Add --config argument (required, type=str)
  - Add --samples argument (required, type=int)
  - Add --output argument (required, type=str)
  - Add helpful argument descriptions

Task 3: Implement DataGenerator class
ADD to surrogate_data_generator.py:
  - Create DataGenerator class with docstring
  - Implement __init__ method:
    - Accept config_path parameter
    - Initialize ModelConfigManager
    - Load configuration
    - Get model path from config
    - Load satellite model
    - Store model as instance variable
    - Handle errors gracefully
  - Implement run_generation placeholder:
    - Accept num_samples and output_path
    - Validate parameters
    - Print status message
    - Create output directory if needed

Task 4: Implement main function
ADD to surrogate_data_generator.py:
  - Parse command-line arguments
  - Setup logging
  - Create DataGenerator instance
  - Call run_generation with arguments
  - Handle exceptions and provide user feedback

Task 5: Add script entry point
ADD to surrogate_data_generator.py:
  - if __name__ == "__main__": block
  - Call main() function
```

### Per task pseudocode

```python
# Task 1: Script boilerplate
#!/usr/bin/env python3
"""
Surrogate Data Generator
=======================

This script generates synthetic training data for surrogate models
using satellite models loaded through the LCAS framework.

Usage:
    python surrogate_data_generator.py --config path/to/config.yaml --samples 1000 --output output.csv
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add project root to path (PATTERN from generate_lightcurves_pytorch.py)
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

# LCAS imports
from src.io.model_io import load_satellite_from_yaml
from src.config.model_config import ModelConfigManager

# Task 2: Argument parsing
def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic training data from satellite models",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        required=True,
        help="Path to model configuration YAML file"
    )
    # ... add other arguments
    
    return parser.parse_args()

# Task 3: DataGenerator class
class DataGenerator:
    """Generates synthetic training data from satellite models."""
    
    def __init__(self, config_path: str):
        """
        Initialize the data generator with a model configuration.
        
        Args:
            config_path: Path to the model configuration YAML file.
        """
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # PATTERN: Use ModelConfigManager like in generate_lightcurves_pytorch.py
        self.logger.info(f"Loading configuration from: {config_path}")
        self.config_manager = ModelConfigManager()
        self.config = self.config_manager.load_config(config_path)
        
        # Get model path and load satellite
        model_path = self.config_manager.get_model_path(self.config)
        self.logger.info(f"Loading satellite model: {model_path}")
        
        # GOTCHA: load_satellite_from_yaml returns None on failure
        self.satellite = load_satellite_from_yaml(str(model_path))
        if self.satellite is None:
            raise ValueError(f"Failed to load satellite model from {model_path}")
            
        self.logger.info(f"Successfully loaded model: {self.satellite.name}")

# Task 4: Main function
def main():
    """Main entry point for the script."""
    # Setup logging (PATTERN from LCAS modules)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Parse arguments
    args = parse_arguments()
    
    try:
        # Create generator
        generator = DataGenerator(args.config)
        
        # Run generation
        generator.run_generation(args.samples, args.output)
        
    except Exception as e:
        logging.error(f"Error: {e}")
        sys.exit(1)
```

### Integration Points
```yaml
CONFIGURATION:
  - Uses existing ModelConfigManager
  - Loads models from data/models/ directory
  - Follows configuration schema from config_schemas.py
  
OUTPUT:
  - Creates directories with Path.mkdir(exist_ok=True)
  - Saves to data/training/ by default
  - Uses CSV format for compatibility
  
LOGGING:
  - Standard logging configuration
  - Module-level loggers
  - INFO level for user feedback
```

## Validation Loop

### Level 1: Syntax & Style
```bash
# Check Python syntax
python -m py_compile surrogate_data_generator.py

# Expected: No output (success). If errors, fix syntax.
```

### Level 2: Import Validation
```python
# Test that all imports work correctly
python -c "import surrogate_data_generator"

# Expected: No errors. If ImportError, check sys.path setup.
```

### Level 3: Argument Parsing Test
```bash
# Test help message
python surrogate_data_generator.py --help

# Test missing arguments
python surrogate_data_generator.py

# Expected: Should show usage and error for missing required arguments
```

### Level 4: Integration Test
```bash
# Test with example configuration
python surrogate_data_generator.py \
    --config data/models/intelsat_901_model.yaml \
    --samples 100 \
    --output data/training/test_output.csv

# Expected: 
# - "Loading configuration from: data/models/intelsat_901_model.yaml"
# - "Successfully loaded model: Intelsat 901"
# - "Would generate 100 samples to data/training/test_output.csv"
```

## Final Validation Checklist
- [ ] Script runs without syntax errors
- [ ] All imports resolve correctly
- [ ] Argument parsing works with all combinations
- [ ] Model loads successfully with valid config
- [ ] Error messages are helpful for invalid inputs
- [ ] Output directory is created if missing
- [ ] Logging provides clear feedback
- [ ] Code follows LCAS conventions
- [ ] Docstrings follow Google style

---

## Anti-Patterns to Avoid
- ❌ Don't use print() - use logging instead
- ❌ Don't use os.path - use pathlib.Path
- ❌ Don't raise generic exceptions - provide context
- ❌ Don't assume paths exist - check or create them
- ❌ Don't hardcode paths - use configuration
- ❌ Don't skip error handling - fail gracefully

## Confidence Score: 9/10

High confidence due to:
- Clear examples in generate_lightcurves_pytorch.py
- Well-documented ModelConfigManager usage
- Established project patterns to follow
- Simple requirements with placeholder implementation

Minor uncertainty only on exact error handling preferences for this specific use case.
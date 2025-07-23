name: "Surrogate MLP Model: PyTorch Neural Network for Satellite Illumination Prediction"
description: |

## Purpose
Create a PyTorch nn.Module class for a Multi-Layer Perceptron (MLP) that predicts satellite face illumination fractions from sun vectors. This foundational neural network will serve as the surrogate model for fast illumination prediction during training and inference.

## Core Principles
1. **Context is King**: Include ALL necessary documentation, examples, and caveats
2. **Validation Loops**: Provide executable tests/lints the AI can run and fix
3. **Information Dense**: Use keywords and patterns from the codebase
4. **Progressive Success**: Start simple, validate, then enhance
5. **Global rules**: Be sure to follow all rules in CLAUDE.md

---

## Goal
Create a production-ready PyTorch nn.Module class `SurrogateNet` in `src/surrogate/model.py` that implements a flexible MLP architecture for predicting satellite face illumination fractions. The network should accept 3D sun vectors as input and output M-dimensional lit fraction predictions.

## Why
- **Business value**: Enables real-time satellite illumination prediction for simulation and analysis
- **Integration**: Provides the core ML component for surrogate modeling in the LCAS framework
- **Problems solved**: Replaces computationally expensive ray tracing with fast neural network inference

## What
A PyTorch neural network module that:
- Accepts 3D sun vectors as input (normalized direction vectors)
- Processes through 3 hidden layers with 64 neurons each and ReLU activation
- Outputs M-dimensional predictions with sigmoid activation (M = number of conceptual faces)
- Supports flexible architecture configuration for different satellite models
- Integrates with existing LCAS PyTorch patterns and device handling
- Includes comprehensive parameter validation and initialization

### Success Criteria
- [ ] `SurrogateNet` class inherits from `torch.nn.Module` properly
- [ ] Constructor accepts `input_size`, `hidden_size`, and `output_size` parameters
- [ ] Forward pass implements the specified architecture correctly
- [ ] Sigmoid activation ensures outputs are in [0, 1] range
- [ ] Device handling follows LCAS PyTorch patterns
- [ ] All validation gates pass (linting, type checking, tests)
- [ ] Compatible with data format from `surrogate_data_generator.py`

## All Needed Context

### Documentation & References
```yaml
# MUST READ - Include these in your context window
- url: https://docs.pytorch.org/stable/generated/torch.nn.Module.html
  why: Core nn.Module patterns, constructor requirements, forward method implementation
  
- url: https://docs.pytorch.org/stable/generated/torch.nn.Linear.html
  why: Linear layer implementation and parameter initialization
  
- url: https://docs.pytorch.org/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity
  why: ReLU and Sigmoid activation functions
  
- file: src/illumination/pytorch_shadow_engine.py:37-81
  why: LCAS PyTorch device handling patterns, GPU detection, and tensor operations
  
- file: generate_lightcurves_pytorch.py:138-142
  why: PyTorch engine integration patterns in LCAS codebase
  
- file: tests/test_surrogate_data_generator.py:1-305
  why: LCAS testing patterns, pytest structure, and mocking approaches
  
- file: surrogate_data_generator.py
  why: Data format compatibility - output shape and structure for training
  
- docfile: context-engineering/05-surrogate-mlp-model/INITIAL_surrogate_mlp_model.md
  why: Specific architecture requirements and integration considerations
```

### Current Codebase tree
```bash
.
├── src/
│   ├── illumination/
│   │   └── pytorch_shadow_engine.py      # PyTorch patterns and device handling
│   ├── config/
│   │   ├── model_config.py               # Configuration management patterns
│   │   └── config_schemas.py             # Data validation patterns
│   ├── core/
│   │   └── common_types.py               # Type definitions
│   └── utils/
│       └── geometry_utils.py             # Mathematical utilities
├── generate_lightcurves_pytorch.py       # PyTorch integration examples
├── surrogate_data_generator.py           # Data format reference
├── tests/
│   └── test_surrogate_data_generator.py  # Testing patterns
└── requirements.txt                      # PyTorch>=1.9.0 available
```

### Desired Codebase tree with files to be added
```bash
.
├── src/
│   ├── surrogate/                        # NEW: Surrogate modeling package
│   │   ├── __init__.py                   # Package initialization
│   │   └── model.py                      # NEW: SurrogateNet implementation
├── tests/
│   └── test_surrogate_model.py           # NEW: Comprehensive unit tests
```

### Known Gotchas of our codebase & Library Quirks
```python
# CRITICAL: PyTorch device handling must follow LCAS patterns
# Pattern from pytorch_shadow_engine.py:40-62 for GPU detection and fallback
use_gpu = torch.cuda.is_available()
device = torch.device('cuda' if use_gpu else 'cpu')

# CRITICAL: All nn.Module classes must call super().__init__() first
# Pattern: super().__init__() before defining any layers

# CRITICAL: Sigmoid activation on output layer is essential
# Physical constraint: lit fractions must be in [0, 1] range

# CRITICAL: Parameter initialization follows PyTorch defaults
# Don't override unless specifically needed - let PyTorch handle it

# CRITICAL: Forward method signature must be exactly: forward(self, x)
# Input x is expected to be tensor of shape (batch_size, input_size)

# CRITICAL: Type hints required throughout (follow LCAS patterns)
# Use typing.Optional, torch.Tensor annotations

# CRITICAL: Google-style docstrings like existing modules
# Include Args:, Returns:, and Raises: sections

# CRITICAL: Use logging.getLogger(__name__) pattern like other modules
import logging
logger = logging.getLogger(__name__)

# CRITICAL: Model should be device-agnostic in constructor
# Device placement handled externally by calling code
```

## Implementation Blueprint

### Data models and structure

The neural network architecture and data flow:
```python
# Architecture specification:
# Input: (batch_size, 3) - Sun vector components (x, y, z)
# Hidden Layer 1: (batch_size, 64) + ReLU
# Hidden Layer 2: (batch_size, 64) + ReLU  
# Hidden Layer 3: (batch_size, 64) + ReLU
# Output: (batch_size, output_size) + Sigmoid

# Expected data compatibility:
# Input format: torch.Tensor of shape (N, 3) from sun vectors
# Output format: torch.Tensor of shape (N, M) where M = number of faces
# Training data: CSV from surrogate_data_generator.py with matching format
```

### List of tasks to be completed to fulfill the PRP in the order they should be completed

```yaml
Task 1: Create Package Structure
CREATE src/surrogate/__init__.py:
  - PATTERN: Follow existing LCAS package structure 
  - ADD package-level docstring describing surrogate modeling functionality
  - IMPORT main classes for convenient access

Task 2: Implement SurrogateNet Core Architecture
CREATE src/surrogate/model.py:
  - PATTERN: Follow pytorch_shadow_engine.py imports and structure
  - IMPLEMENT SurrogateNet class inheriting from nn.Module
  - DEFINE flexible constructor with input_size, hidden_size, output_size parameters
  - CREATE layer definitions: 3 hidden layers + output layer
  - PRESERVE PyTorch parameter initialization defaults

Task 3: Implement Forward Pass Method
MODIFY src/surrogate/model.py SurrogateNet class:
  - IMPLEMENT forward(self, x) method with correct signature
  - CHAIN: input -> hidden1+ReLU -> hidden2+ReLU -> hidden3+ReLU -> output+Sigmoid
  - VALIDATE input tensor shape and provide helpful error messages
  - FOLLOW tensor flow patterns from pytorch_shadow_engine.py

Task 4: Add Device Handling Support
MODIFY src/surrogate/model.py:
  - ADD device property method for compatibility
  - PATTERN: Follow pytorch_shadow_engine.py device detection approach
  - IMPLEMENT to() method override if needed for device transfer
  - ENSURE model parameters move with device changes

Task 5: Add Input Validation and Error Handling
MODIFY src/surrogate/model.py:
  - VALIDATE constructor parameters (positive integers, type checking)
  - VALIDATE forward pass input shape and type
  - ADD comprehensive docstrings with Google style
  - HANDLE edge cases gracefully with informative error messages

Task 6: Create Comprehensive Unit Tests
CREATE tests/test_surrogate_model.py:
  - PATTERN: Follow test_surrogate_data_generator.py structure and conventions
  - TEST constructor with valid and invalid parameters
  - TEST forward pass with various input shapes and types
  - TEST device handling and parameter movement
  - TEST output range validation (sigmoid constraint)
  - TEST model serialization/deserialization
  - MOCK external dependencies where appropriate
```

### Per task pseudocode as needed added to each task

```python
# Task 2: Core Architecture Implementation
class SurrogateNet(nn.Module):
    """
    Multi-Layer Perceptron for satellite face illumination prediction.
    
    Predicts lit fractions for satellite conceptual faces given sun vector direction.
    Architecture: 3 hidden layers (64 neurons each) with ReLU, sigmoid output.
    
    Args:
        input_size: Number of input features (default: 3 for sun vector)
        hidden_size: Number of neurons in hidden layers (default: 64)
        output_size: Number of output features (number of conceptual faces)
    """
    
    def __init__(self, input_size: int = 3, hidden_size: int = 64, output_size: int = 1):
        # PATTERN: Always call parent constructor first
        super().__init__()
        
        # CRITICAL: Validate parameters
        if not isinstance(input_size, int) or input_size <= 0:
            raise ValueError(f"input_size must be positive integer, got {input_size}")
        
        # PATTERN: Define layers as module attributes for parameter registration
        self.hidden1 = nn.Linear(input_size, hidden_size)
        self.hidden2 = nn.Linear(hidden_size, hidden_size)
        self.hidden3 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        
        # PATTERN: Store architecture info for debugging
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # PATTERN: Initialize logger like other LCAS modules
        self.logger = logging.getLogger(__name__)

# Task 3: Forward Pass Implementation  
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Forward pass through the network.
    
    Args:
        x: Input tensor of shape (batch_size, input_size) containing sun vectors
        
    Returns:
        Output tensor of shape (batch_size, output_size) with lit fractions [0, 1]
        
    Raises:
        ValueError: If input tensor has incorrect shape
    """
    # CRITICAL: Validate input
    if not isinstance(x, torch.Tensor):
        raise ValueError(f"Input must be torch.Tensor, got {type(x)}")
    
    if x.dim() != 2 or x.size(1) != self.input_size:
        raise ValueError(f"Input must have shape (batch_size, {self.input_size}), got {x.shape}")
    
    # PATTERN: Chain operations through hidden layers
    x = torch.relu(self.hidden1(x))
    x = torch.relu(self.hidden2(x))
    x = torch.relu(self.hidden3(x))
    
    # CRITICAL: Sigmoid activation for [0, 1] constraint
    x = torch.sigmoid(self.output(x))
    
    return x

# Task 6: Unit Tests Structure
class TestSurrogateNet:
    """Test the SurrogateNet neural network class."""
    
    def test_constructor_valid_parameters(self):
        """Test SurrogateNet constructor with valid parameters."""
        # PATTERN: Test default parameters
        model = SurrogateNet()
        assert model.input_size == 3
        assert model.hidden_size == 64
        assert model.output_size == 1
        
        # Test custom parameters
        model = SurrogateNet(input_size=5, hidden_size=32, output_size=10)
        assert model.input_size == 5
        assert model.hidden_size == 32
        assert model.output_size == 10
    
    def test_forward_pass_valid_input(self):
        """Test forward pass with valid input tensors."""
        model = SurrogateNet(input_size=3, output_size=5)
        
        # Single sample
        x = torch.randn(1, 3)
        output = model(x)
        assert output.shape == (1, 5)
        assert torch.all(output >= 0.0) and torch.all(output <= 1.0)
        
        # Batch of samples
        x = torch.randn(10, 3)
        output = model(x)
        assert output.shape == (10, 5)
        assert torch.all(output >= 0.0) and torch.all(output <= 1.0)
    
    def test_forward_pass_invalid_input(self):
        """Test forward pass error handling."""
        model = SurrogateNet()
        
        # Wrong input shape
        with pytest.raises(ValueError, match="Input must have shape"):
            model(torch.randn(10, 5))  # Wrong feature dimension
            
        # Wrong input type
        with pytest.raises(ValueError, match="Input must be torch.Tensor"):
            model(np.array([[1, 2, 3]]))
```

### Integration Points
```yaml
PACKAGE_STRUCTURE:
  - create: src/surrogate/ directory
  - pattern: Follow existing src/ package organization
  - imports: Support from src.surrogate.model import SurrogateNet
  
DEVICE_HANDLING:
  - follow: pytorch_shadow_engine.py device detection patterns
  - pattern: self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  - compatibility: Model should work on both CPU and GPU
  
DATA_COMPATIBILITY:
  - input_format: torch.Tensor shape (batch_size, 3) for sun vectors
  - output_format: torch.Tensor shape (batch_size, num_faces) with sigmoid [0,1]
  - training_data: Compatible with surrogate_data_generator.py CSV output
  
TESTING:
  - framework: pytest following test_surrogate_data_generator.py patterns
  - location: tests/test_surrogate_model.py
  - coverage: Constructor, forward pass, device handling, edge cases
```

## Validation Loop

### Level 1: Syntax & Style
```bash
# Run these FIRST - fix any errors before proceeding
ruff check src/surrogate/ --fix           # Auto-fix style issues
mypy src/surrogate/                       # Type checking
python -c "import src.surrogate.model"    # Import validation

# Expected: No errors. If errors, READ the error and fix.
```

### Level 2: Unit Tests for new functionality
```python
# CREATE tests/test_surrogate_model.py with comprehensive test cases:
def test_surrogate_net_architecture():
    """Test that network architecture matches specifications."""
    model = SurrogateNet(input_size=3, hidden_size=64, output_size=10)
    
    # Verify layer dimensions
    assert model.hidden1.in_features == 3
    assert model.hidden1.out_features == 64
    assert model.hidden2.in_features == 64
    assert model.hidden2.out_features == 64
    assert model.hidden3.in_features == 64
    assert model.hidden3.out_features == 64
    assert model.output.in_features == 64
    assert model.output.out_features == 10

def test_sigmoid_output_constraint():
    """Test that output is always in [0, 1] range."""
    model = SurrogateNet(output_size=5)
    
    # Test with various input magnitudes
    for _ in range(10):
        x = torch.randn(5, 3) * 10  # Large random inputs
        output = model(x)
        assert torch.all(output >= 0.0), "Output contains values < 0"
        assert torch.all(output <= 1.0), "Output contains values > 1"

def test_device_compatibility():
    """Test model works on CPU and GPU if available."""
    model = SurrogateNet()
    x = torch.randn(3, 3)
    
    # CPU test
    model_cpu = model.to('cpu')
    x_cpu = x.to('cpu')
    output_cpu = model_cpu(x_cpu)
    assert output_cpu.device.type == 'cpu'
    
    # GPU test (if available)
    if torch.cuda.is_available():
        model_gpu = model.to('cuda')
        x_gpu = x.to('cuda')
        output_gpu = model_gpu(x_gpu)
        assert output_gpu.device.type == 'cuda'

def test_parameter_validation():
    """Test constructor parameter validation."""
    # Valid parameters
    SurrogateNet(input_size=3, hidden_size=64, output_size=10)
    
    # Invalid parameters
    with pytest.raises(ValueError):
        SurrogateNet(input_size=0)  # Zero input size
    with pytest.raises(ValueError):
        SurrogateNet(input_size=-1)  # Negative input size
    with pytest.raises(ValueError):
        SurrogateNet(hidden_size=0)  # Zero hidden size
    with pytest.raises(ValueError):
        SurrogateNet(output_size=0)  # Zero output size
```

```bash
# Run tests iteratively until passing:
pytest tests/test_surrogate_model.py -v --cov=src/surrogate --cov-report=term-missing

# If failing: Debug specific test, understand root cause, fix code, re-run
```

### Level 3: Integration Test
```python
# Test integration with surrogate data generator format
def test_integration_with_data_generator():
    """Test that model works with data from surrogate_data_generator.py."""
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
```

## Final Validation Checklist
- [ ] All tests pass: `pytest tests/test_surrogate_model.py -v`
- [ ] No linting errors: `ruff check src/surrogate/`
- [ ] No type errors: `mypy src/surrogate/`
- [ ] Import works: `python -c "from src.surrogate.model import SurrogateNet"`
- [ ] Model architecture correct: 3→64→64→64→M with ReLU/Sigmoid
- [ ] Output range constraint verified: All outputs in [0, 1]
- [ ] Device handling works on CPU and GPU
- [ ] Parameter validation prevents invalid configurations
- [ ] Compatible with surrogate_data_generator.py data format
- [ ] Docstrings follow Google style like LCAS modules
- [ ] Logging integration follows LCAS patterns

---

## Anti-Patterns to Avoid
- ❌ Don't skip super().__init__() call - required for nn.Module
- ❌ Don't override parameter initialization unnecessarily - trust PyTorch defaults
- ❌ Don't forget sigmoid activation on output - critical for [0,1] constraint
- ❌ Don't hardcode layer sizes - use constructor parameters for flexibility
- ❌ Don't ignore input validation - provide helpful error messages
- ❌ Don't skip device compatibility testing - model must work on CPU/GPU
- ❌ Don't use print() for debugging - use logging like other LCAS modules
- ❌ Don't forget type hints - required by LCAS conventions

## Confidence Score: 9/10

High confidence due to:
- **Clear architecture specification**: Exact layer structure and activation functions defined
- **Strong codebase patterns**: Existing PyTorch usage in pytorch_shadow_engine.py provides clear templates
- **Comprehensive validation**: Multiple testing levels ensure correctness
- **Well-defined scope**: Single class implementation with clear requirements
- **Established conventions**: LCAS patterns for structure, documentation, and testing

Minor uncertainty around:
- Specific parameter initialization preferences (using PyTorch defaults)
- Integration testing with actual training pipeline (addressed in validation)

The implementation should succeed in one pass due to complete context and clear validation gates.
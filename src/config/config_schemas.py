"""
Author: Girish Narayanan
Configuration schemas for RSO model system.
Defines the structure and validation for configuration files.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from pathlib import Path
import numpy as np

@dataclass
class BRDFParameters:
    """BRDF material parameters."""
    r_d: float = 0.1
    r_s: float = 0.1
    n_phong: float = 10.0

@dataclass
class ArticulationParameters:
    """Parameters for articulation behavior."""
    rule_type: str = "none"
    rotation_center: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    rotation_axis: List[float] = field(default_factory=lambda: [0.0, 0.0, 1.0])
    reference_normal: List[float] = field(default_factory=lambda: [1.0, 0.0, 0.0])
    parameters: Dict[str, Any] = field(default_factory=dict)
    limits: Optional[Dict[str, float]] = None

@dataclass
class SpiceConfig:
    """SPICE configuration parameters."""
    satellite_id: int = -126824
    metakernel_path: str = ""
    body_frame: str = "IS901_BUS_FRAME"

@dataclass
class SimulationDefaults:
    """Default simulation parameters."""
    subdivision_level: int = 3
    start_time: str = "2020-02-05T10:00:00"
    end_time: str = "2020-02-05T16:00:00"
    output_dir: str = "lightcurve_results_pytorch"

@dataclass
class ModelInfo:
    """Model file information."""
    name: str = "Unknown Model"
    model_file: str = ""
    brdf_file: str = ""
    articulation_file: str = ""

@dataclass
class RSO_Config:
    """Complete RSO configuration."""
    model_info: ModelInfo = field(default_factory=ModelInfo)
    spice_config: SpiceConfig = field(default_factory=SpiceConfig)
    simulation_defaults: SimulationDefaults = field(default_factory=SimulationDefaults)
    brdf_mappings: Dict[str, BRDFParameters] = field(default_factory=dict)
    articulation_rules: Dict[str, ArticulationParameters] = field(default_factory=dict)
    default_brdf: BRDFParameters = field(default_factory=BRDFParameters)
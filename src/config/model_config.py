"""
Model configuration management for RSO light curve generation.
Handles loading and management of model-specific configurations.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import asdict

from .config_schemas import RSO_Config, ModelInfo, SpiceConfig, SimulationDefaults
from .config_schemas import BRDFParameters, ArticulationParameters


class ModelConfigManager:
    """Manages loading and validation of RSO model configurations."""
    
    def __init__(self, project_root: Optional[Path] = None):
        """
        Initialize the configuration manager.
        
        Args:
            project_root: Path to project root directory
        """
        if project_root is None:
            # Default to current project structure
            self.project_root = Path(__file__).parent.parent.parent
        else:
            self.project_root = Path(project_root)
            
        self.models_dir = self.project_root / "data" / "models"
    
    def load_config(self, config_path: Optional[Union[str, Path]] = None) -> RSO_Config:
        """
        Load RSO configuration from file or use default Intelsat 901 configuration.
        
        Args:
            config_path: Path to configuration file. If None, uses Intelsat 901 defaults.
            
        Returns:
            RSO_Config: Loaded configuration
        """
        if config_path is None:
            # Use default Intelsat 901 configuration for backward compatibility
            return self._get_default_intelsat901_config()
        
        config_path = Path(config_path)
        if not config_path.is_absolute():
            config_path = self.models_dir / config_path
            
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
            
        # Parse main configuration
        config = RSO_Config()
        
        # Load model info
        if 'model_info' in config_data:
            model_info = config_data['model_info']
            config.model_info = ModelInfo(
                name=model_info.get('name', 'Unknown Model'),
                model_file=model_info.get('model_file', ''),
                brdf_file=model_info.get('brdf_file', ''),
                articulation_file=model_info.get('articulation_file', '')
            )
        
        # Load SPICE config
        if 'spice_config' in config_data:
            spice_config = config_data['spice_config']
            config.spice_config = SpiceConfig(
                satellite_id=spice_config.get('satellite_id', -126824),
                metakernel_path=spice_config.get('metakernel_path', ''),
                body_frame=spice_config.get('body_frame', 'IS901_BUS_FRAME')
            )
        
        # Load simulation defaults
        if 'simulation_defaults' in config_data:
            sim_defaults = config_data['simulation_defaults']
            config.simulation_defaults = SimulationDefaults(
                subdivision_level=sim_defaults.get('subdivision_level', 3),
                start_time=sim_defaults.get('start_time', '2020-02-05T10:00:00'),
                end_time=sim_defaults.get('end_time', '2020-02-05T16:00:00'),
                output_dir=sim_defaults.get('output_dir', 'lightcurve_results_pytorch')
            )
        
        # Load BRDF and articulation from separate files if specified
        base_dir = config_path.parent
        
        # Load BRDF mappings
        if config.model_info.brdf_file:
            brdf_path = base_dir / config.model_info.brdf_file
            config.brdf_mappings, config.default_brdf = self._load_brdf_config(brdf_path)
            
        # Load articulation rules
        if config.model_info.articulation_file:
            articulation_path = base_dir / config.model_info.articulation_file
            config.articulation_rules = self._load_articulation_config(articulation_path)
            
        return config
    
    def _load_brdf_config(self, brdf_path: Path) -> tuple[Dict[str, BRDFParameters], BRDFParameters]:
        """Load BRDF configuration from file."""
        if not brdf_path.exists():
            return {}, BRDFParameters()
            
        with open(brdf_path, 'r') as f:
            brdf_data = yaml.safe_load(f)
            
        brdf_mappings = {}
        if 'brdf_mappings' in brdf_data:
            for component_name, params in brdf_data['brdf_mappings'].items():
                brdf_mappings[component_name] = BRDFParameters(
                    r_d=params.get('r_d', 0.1),
                    r_s=params.get('r_s', 0.1),
                    n_phong=params.get('n_phong', 10.0)
                )
        
        default_brdf = BRDFParameters()
        if 'default_brdf' in brdf_data:
            default_params = brdf_data['default_brdf']
            default_brdf = BRDFParameters(
                r_d=default_params.get('r_d', 0.1),
                r_s=default_params.get('r_s', 0.1),
                n_phong=default_params.get('n_phong', 10.0)
            )
            
        return brdf_mappings, default_brdf
    
    def _load_articulation_config(self, articulation_path: Path) -> Dict[str, ArticulationParameters]:
        """Load articulation configuration from file."""
        if not articulation_path.exists():
            return {}
            
        with open(articulation_path, 'r') as f:
            articulation_data = yaml.safe_load(f)
            
        articulation_rules = {}
        if 'articulation_rules' in articulation_data:
            for component_name, params in articulation_data['articulation_rules'].items():
                articulation_rules[component_name] = ArticulationParameters(
                    rule_type=params.get('rule_type', 'none'),
                    rotation_center=params.get('rotation_center', [0.0, 0.0, 0.0]),
                    rotation_axis=params.get('rotation_axis', [0.0, 0.0, 1.0]),
                    reference_normal=params.get('reference_normal', [1.0, 0.0, 0.0]),
                    parameters=params.get('parameters', {}),
                    limits=params.get('limits', None)
                )
                
        return articulation_rules
    
    def _get_default_intelsat901_config(self) -> RSO_Config:
        """Get default Intelsat 901 configuration for backward compatibility."""
        
        # Default BRDF parameters (matching current hardcoded values)
        default_brdf_mappings = {
            "Bus": BRDFParameters(r_d=0.02, r_s=0.5, n_phong=300.0),
            "Solar_Panel": BRDFParameters(r_d=0.026, r_s=0.3, n_phong=200.0),
            "Solar_Panel_North": BRDFParameters(r_d=0.026, r_s=0.3, n_phong=200.0),
            "Solar_Panel_South": BRDFParameters(r_d=0.026, r_s=0.3, n_phong=200.0),
            "Antenna": BRDFParameters(r_d=0.01, r_s=0.4, n_phong=200.0),
            "Antenna_Dish_West": BRDFParameters(r_d=0.01, r_s=0.4, n_phong=200.0),
            "Antenna_Dish_East": BRDFParameters(r_d=0.01, r_s=0.4, n_phong=200.0)
        }
        
        # Default articulation rules (matching current hardcoded behavior)
        default_articulation_rules = {
            "Solar_Panel_North": ArticulationParameters(
                rule_type="track_sun",
                rotation_center=[0.0, 0.0, 0.0],
                rotation_axis=[0.0, 0.0, 1.0],
                reference_normal=[1.0, 0.0, 0.0]
            ),
            "Solar_Panel_South": ArticulationParameters(
                rule_type="track_sun",
                rotation_center=[0.0, 0.0, 0.0],
                rotation_axis=[0.0, 0.0, 1.0],
                reference_normal=[1.0, 0.0, 0.0]
            )
        }
        
        return RSO_Config(
            model_info=ModelInfo(
                name="Intelsat 901",
                model_file="intelsat_901_model.yaml",
                brdf_file="",
                articulation_file=""
            ),
            spice_config=SpiceConfig(
                satellite_id=-126824,
                metakernel_path=str(self.project_root / "data" / "spice_kernels" / "missions" / "dst-is901" / "INTELSAT_901-metakernel.tm"),
                body_frame="IS901_BUS_FRAME"
            ),
            simulation_defaults=SimulationDefaults(
                subdivision_level=3,
                start_time="2020-02-05T10:00:00",
                end_time="2020-02-05T16:00:00",
                output_dir="lightcurve_results_pytorch"
            ),
            brdf_mappings=default_brdf_mappings,
            articulation_rules=default_articulation_rules,
            default_brdf=BRDFParameters(r_d=0.1, r_s=0.1, n_phong=10.0)
        )
    
    def get_model_path(self, config: RSO_Config) -> Path:
        """Get full path to model file."""
        if not config.model_info.model_file:
            # Default to Intelsat 901 model
            return self.models_dir / "intelsat_901_model.yaml"
        
        model_path = Path(config.model_info.model_file)
        if not model_path.is_absolute():
            model_path = self.models_dir / model_path
            
        return model_path
    
    def get_metakernel_path(self, config: RSO_Config) -> Path:
        """Get full path to metakernel file."""
        metakernel_path = Path(config.spice_config.metakernel_path)
        if not metakernel_path.is_absolute():
            metakernel_path = self.project_root / metakernel_path
            
        return metakernel_path


# Global instance for easy access
_config_manager = None

def get_config_manager() -> ModelConfigManager:
    """Get global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ModelConfigManager()
    return _config_manager
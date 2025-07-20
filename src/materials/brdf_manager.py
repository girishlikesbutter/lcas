"""
BRDF material parameter management system.
Provides flexible mapping of BRDF parameters to satellite components.
"""

from typing import Dict, Any, Optional
from src.config.config_schemas import RSO_Config, BRDFParameters
from src.models.model_definitions import Satellite, Component, BRDFMaterialProperties


class BRDFManager:
    """Manages BRDF material parameter assignment for satellite components."""
    
    def __init__(self, config: RSO_Config):
        """
        Initialize BRDF manager with configuration.
        
        Args:
            config: RSO configuration containing BRDF mappings
        """
        self.config = config
        self.brdf_mappings = config.brdf_mappings
        self.default_brdf = config.default_brdf
        
        # Legacy hardcoded BRDF parameters for backward compatibility
        self.legacy_brdf_params = {
            "Bus": {"r_d": 0.02, "r_s": 0.5, "n_phong": 300.0},
            "Solar_Panel": {"r_d": 0.026, "r_s": 0.3, "n_phong": 200.0},
            "Antenna": {"r_d": 0.01, "r_s": 0.4, "n_phong": 200.0}
        }
    
    def get_brdf_parameters(self, component: Component) -> BRDFMaterialProperties:
        """
        Get BRDF parameters for a component with flexible mapping.
        
        Args:
            component: Component to get BRDF parameters for
            
        Returns:
            BRDFMaterialProperties: BRDF parameters for the component
        """
        # First, try exact component name match
        if component.name in self.brdf_mappings:
            params = self.brdf_mappings[component.name]
            return BRDFMaterialProperties(
                r_d=params.r_d,
                r_s=params.r_s,
                n_phong=params.n_phong
            )
        
        # Second, try substring matching for backward compatibility
        for key, params in self.brdf_mappings.items():
            if key in component.name:
                return BRDFMaterialProperties(
                    r_d=params.r_d,
                    r_s=params.r_s,
                    n_phong=params.n_phong
                )
        
        # Third, try legacy hardcoded parameters for backward compatibility
        for key, params in self.legacy_brdf_params.items():
            if key in component.name:
                return BRDFMaterialProperties(
                    r_d=params["r_d"],
                    r_s=params["r_s"],
                    n_phong=params["n_phong"]
                )
        
        # Finally, use default parameters
        return BRDFMaterialProperties(
            r_d=self.default_brdf.r_d,
            r_s=self.default_brdf.r_s,
            n_phong=self.default_brdf.n_phong
        )
    
    def update_satellite_brdf_parameters(self, satellite: Satellite) -> None:
        """
        Update BRDF parameters for all components in a satellite.
        
        Args:
            satellite: Satellite model to update
        """
        for component in satellite.components:
            # Get BRDF parameters using flexible mapping
            brdf_params = self.get_brdf_parameters(component)
            
            # Update component default material
            if component.default_material is None:
                component.default_material = brdf_params
            else:
                component.default_material.r_d = brdf_params.r_d
                component.default_material.r_s = brdf_params.r_s
                component.default_material.n_phong = brdf_params.n_phong
            
            # Update all facets in this component
            for facet in component.facets:
                facet.material_properties.r_d = brdf_params.r_d
                facet.material_properties.r_s = brdf_params.r_s
                facet.material_properties.n_phong = brdf_params.n_phong
    
    def get_legacy_brdf_dict(self) -> Dict[str, Dict[str, float]]:
        """
        Get legacy-format BRDF dictionary for backward compatibility.
        
        Returns:
            Dictionary in legacy format for existing code
        """
        legacy_dict = {}
        
        # Add configured mappings
        for component_name, params in self.brdf_mappings.items():
            legacy_dict[component_name] = {
                "r_d": params.r_d,
                "r_s": params.r_s,
                "n_phong": params.n_phong
            }
        
        # Add legacy hardcoded values for backward compatibility
        legacy_dict.update(self.legacy_brdf_params)
        
        return legacy_dict
    
    def validate_brdf_parameters(self) -> bool:
        """
        Validate BRDF parameters for physical correctness.
        
        Returns:
            True if all parameters are valid, False otherwise
        """
        def validate_single_params(params: BRDFParameters) -> bool:
            """Validate a single set of BRDF parameters."""
            if not (0.0 <= params.r_d <= 1.0):
                return False
            if not (0.0 <= params.r_s <= 1.0):
                return False
            if params.n_phong < 0.0:
                return False
            if params.r_d + params.r_s > 1.0:
                return False
            return True
        
        # Validate all component mappings
        for component_name, params in self.brdf_mappings.items():
            if not validate_single_params(params):
                print(f"Invalid BRDF parameters for component '{component_name}'")
                return False
        
        # Validate default parameters
        if not validate_single_params(self.default_brdf):
            print("Invalid default BRDF parameters")
            return False
        
        return True
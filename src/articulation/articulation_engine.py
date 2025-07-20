"""
Articulation engine for flexible component movement.
Manages articulation behaviors and applies transformations.
"""

import numpy as np
from typing import Dict, Any, Optional
from src.config.config_schemas import RSO_Config, ArticulationParameters
from src.models.model_definitions import Satellite, Component
from .behaviors.base_behavior import ArticulationBehavior
from .behaviors.sun_tracking import SunTrackingBehavior
from .behaviors.earth_tracking import EarthTrackingBehavior
from .behaviors.oscillating import OscillatingBehavior


class ArticulationEngine:
    """Manages articulation behaviors and applies transformations to components."""
    
    def __init__(self, config: RSO_Config):
        """
        Initialize articulation engine with configuration.
        
        Args:
            config: RSO configuration containing articulation rules
        """
        self.config = config
        self.articulation_rules = config.articulation_rules
        self.behavior_registry = {
            'track_sun': SunTrackingBehavior,
            'track_earth': EarthTrackingBehavior,
            'oscillate': OscillatingBehavior,
            'none': None
        }
    
    def register_behavior(self, behavior_name: str, behavior_class: type):
        """
        Register a new articulation behavior.
        
        Args:
            behavior_name: Name of the behavior
            behavior_class: Class implementing the behavior
        """
        self.behavior_registry[behavior_name] = behavior_class
    
    def get_component_behavior(self, component: Component) -> Optional[ArticulationBehavior]:
        """
        Get articulation behavior for a component.
        
        Args:
            component: Component to get behavior for
            
        Returns:
            ArticulationBehavior instance or None if no articulation
        """
        # Check if component has articulation rule in config
        if component.name in self.articulation_rules:
            rule = self.articulation_rules[component.name]
            behavior_class = self.behavior_registry.get(rule.rule_type)
            
            if behavior_class:
                # Convert ArticulationParameters to dictionary
                params = {
                    'rotation_center': rule.rotation_center,
                    'rotation_axis': rule.rotation_axis,
                    'reference_normal': rule.reference_normal,
                    'limits': rule.limits,
                    **rule.parameters  # Add any custom parameters
                }
                return behavior_class(params)
        
        # Check legacy articulation_rule field for backward compatibility
        if hasattr(component, 'articulation_rule') and component.articulation_rule:
            if component.articulation_rule == "TRACK_SUN_PRIMARY_AXIS_Z_SECONDARY_X":
                # Legacy hardcoded behavior
                params = {
                    'rotation_center': [0.0, 0.0, 0.0],
                    'rotation_axis': [0.0, 0.0, 1.0],
                    'reference_normal': [1.0, 0.0, 0.0],
                    'limits': None
                }
                return SunTrackingBehavior(params)
        
        return None
    
    def calculate_articulation_rotation(self, 
                                      component: Component,
                                      sun_vector_body: np.ndarray,
                                      earth_vector_body: np.ndarray,
                                      epoch: float,
                                      offset_deg: float = 0.0) -> Optional[np.ndarray]:
        """
        Calculate articulation rotation matrix for a component.
        
        Args:
            component: Component to calculate rotation for
            sun_vector_body: Sun vector in body frame
            earth_vector_body: Earth vector in body frame
            epoch: Current epoch time
            offset_deg: Additional offset in degrees
            
        Returns:
            4x4 rotation matrix or None if no articulation
        """
        behavior = self.get_component_behavior(component)
        if behavior is None:
            return None
        
        # Calculate rotation angle using behavior
        angle_deg = behavior.calculate_rotation_angle(
            sun_vector_body, earth_vector_body, epoch, offset_deg
        )
        
        # Get rotation matrix
        rotation_matrix = behavior.get_rotation_matrix(angle_deg)
        
        return rotation_matrix
    
    def apply_articulation_to_satellite(self,
                                      satellite: Satellite,
                                      sun_vector_body: np.ndarray,
                                      earth_vector_body: np.ndarray,
                                      epoch: float,
                                      offset_deg: float = 0.0) -> Dict[str, float]:
        """
        Apply articulation to all components in a satellite.
        
        Args:
            satellite: Satellite to apply articulation to
            sun_vector_body: Sun vector in body frame
            earth_vector_body: Earth vector in body frame
            epoch: Current epoch time
            offset_deg: Additional offset in degrees
            
        Returns:
            Dictionary of component names to rotation angles
        """
        articulation_angles = {}
        
        for component in satellite.components:
            behavior = self.get_component_behavior(component)
            if behavior is not None:
                # Calculate rotation angle
                angle_deg = behavior.calculate_rotation_angle(
                    sun_vector_body, earth_vector_body, epoch, offset_deg
                )
                
                # Get rotation matrix
                rotation_matrix = behavior.get_rotation_matrix(angle_deg)
                
                # Apply transformation to component
                satellite.transform_components_by_name([component.name], rotation_matrix)
                
                # Store angle for reporting
                articulation_angles[component.name] = angle_deg
        
        return articulation_angles
    
    def is_component_articulated(self, component: Component) -> bool:
        """
        Check if a component has articulation.
        
        Args:
            component: Component to check
            
        Returns:
            True if component has articulation, False otherwise
        """
        return self.get_component_behavior(component) is not None
    
    def get_legacy_articulation_check(self, component: Component) -> bool:
        """
        Legacy check for articulation compatibility.
        
        Args:
            component: Component to check
            
        Returns:
            True if component has legacy articulation
        """
        # Check new system
        if self.is_component_articulated(component):
            return True
        
        # Check legacy system
        if hasattr(component, 'articulation_rule') and component.articulation_rule:
            return component.articulation_rule == "TRACK_SUN_PRIMARY_AXIS_Z_SECONDARY_X"
        
        return False
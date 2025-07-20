"""
Earth tracking articulation behavior.
Implements earth pointing functionality for antennas or other components.
"""

import numpy as np
from typing import Dict, Any
from .base_behavior import ArticulationBehavior
from src.utils.geometry_utils import calculate_sun_pointing_rotation


class EarthTrackingBehavior(ArticulationBehavior):
    """Earth tracking articulation behavior for antennas."""
    
    def __init__(self, parameters: Dict[str, Any]):
        """
        Initialize earth tracking behavior.
        
        Args:
            parameters: Dictionary containing articulation parameters
        """
        super().__init__(parameters)
    
    def calculate_rotation_angle(self, 
                               sun_vector_body: np.ndarray,
                               earth_vector_body: np.ndarray,
                               epoch: float,
                               offset_deg: float = 0.0) -> float:
        """
        Calculate rotation angle to track earth.
        
        Args:
            sun_vector_body: Sun vector in body frame (unused for earth tracking)
            earth_vector_body: Earth vector in body frame
            epoch: Current epoch time (unused for earth tracking)
            offset_deg: Additional offset in degrees
            
        Returns:
            Rotation angle in degrees
        """
        # Use existing calculation function but with earth vector
        panel_angle_deg = calculate_sun_pointing_rotation(
            earth_vector_body, 
            self.rotation_axis, 
            self.reference_normal
        )
        
        # Apply offset
        final_angle = panel_angle_deg + offset_deg
        
        # Apply limits
        final_angle = self.apply_limits(final_angle)
        
        return final_angle
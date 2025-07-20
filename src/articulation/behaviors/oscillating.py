"""
Oscillating articulation behavior.
Implements sinusoidal oscillation for components.
"""

import numpy as np
from typing import Dict, Any
from .base_behavior import ArticulationBehavior


class OscillatingBehavior(ArticulationBehavior):
    """Oscillating articulation behavior for periodic movement."""
    
    def __init__(self, parameters: Dict[str, Any]):
        """
        Initialize oscillating behavior.
        
        Args:
            parameters: Dictionary containing articulation parameters
        """
        super().__init__(parameters)
        self.amplitude = parameters.get('amplitude', 30.0)  # degrees
        self.frequency = parameters.get('frequency', 0.5)   # Hz
        self.phase = parameters.get('phase', 0.0)          # radians
    
    def calculate_rotation_angle(self, 
                               sun_vector_body: np.ndarray,
                               earth_vector_body: np.ndarray,
                               epoch: float,
                               offset_deg: float = 0.0) -> float:
        """
        Calculate oscillating rotation angle.
        
        Args:
            sun_vector_body: Sun vector in body frame (unused for oscillation)
            earth_vector_body: Earth vector in body frame (unused for oscillation)
            epoch: Current epoch time (used for oscillation)
            offset_deg: Additional offset in degrees
            
        Returns:
            Rotation angle in degrees
        """
        # Calculate oscillation angle
        time_radians = 2 * np.pi * self.frequency * epoch + self.phase
        oscillation_angle = self.amplitude * np.sin(time_radians)
        
        # Apply offset
        final_angle = oscillation_angle + offset_deg
        
        # Apply limits
        final_angle = self.apply_limits(final_angle)
        
        return final_angle
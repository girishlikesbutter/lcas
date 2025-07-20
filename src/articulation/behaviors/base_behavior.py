"""
Base class for articulation behaviors.
Defines the interface for all articulation behaviors.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class ArticulationBehavior(ABC):
    """Base class for all articulation behaviors."""
    
    def __init__(self, parameters: Dict[str, Any]):
        """
        Initialize the articulation behavior.
        
        Args:
            parameters: Dictionary of behavior-specific parameters
        """
        self.parameters = parameters
        self.rotation_center = np.array(parameters.get('rotation_center', [0.0, 0.0, 0.0]))
        self.rotation_axis = np.array(parameters.get('rotation_axis', [0.0, 0.0, 1.0]))
        self.reference_normal = np.array(parameters.get('reference_normal', [1.0, 0.0, 0.0]))
        self.limits = parameters.get('limits', None)
    
    @abstractmethod
    def calculate_rotation_angle(self, 
                               sun_vector_body: np.ndarray,
                               earth_vector_body: np.ndarray,
                               epoch: float,
                               offset_deg: float = 0.0) -> float:
        """
        Calculate rotation angle for this epoch.
        
        Args:
            sun_vector_body: Sun vector in body frame
            earth_vector_body: Earth vector in body frame
            epoch: Current epoch time
            offset_deg: Additional offset in degrees
            
        Returns:
            Rotation angle in degrees
        """
        pass
    
    def apply_limits(self, angle_deg: float) -> float:
        """
        Apply rotation limits to the calculated angle.
        
        Args:
            angle_deg: Calculated angle in degrees
            
        Returns:
            Limited angle in degrees
        """
        if self.limits is None:
            return angle_deg
        
        min_angle = self.limits.get('min_angle', -180.0)
        max_angle = self.limits.get('max_angle', 180.0)
        
        return np.clip(angle_deg, min_angle, max_angle)
    
    def get_rotation_matrix(self, angle_deg: float) -> np.ndarray:
        """
        Get 4x4 homogeneous rotation matrix for the given angle.
        
        Args:
            angle_deg: Rotation angle in degrees
            
        Returns:
            4x4 rotation matrix
        """
        angle_rad = np.radians(angle_deg)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        
        # Normalize rotation axis
        axis = self.rotation_axis / np.linalg.norm(self.rotation_axis)
        
        # Create rotation matrix using Rodrigues' formula
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        
        R = np.eye(3) + sin_a * K + (1 - cos_a) * np.dot(K, K)
        
        # Convert to 4x4 homogeneous matrix
        rotation_matrix = np.eye(4)
        rotation_matrix[:3, :3] = R
        
        return rotation_matrix
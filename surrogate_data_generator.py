#!/usr/bin/env python3
"""
Surrogate Data Generator
=======================

This script generates synthetic training data for surrogate models
using satellite models loaded through the LCAS framework.

Usage:
    python surrogate_data_generator.py --config path/to/config.yaml --samples 1000 --output output.csv
"""

import sys
import argparse
import logging
from pathlib import Path
import numpy as np
from typing import Optional, Dict
import trimesh
from trimesh.ray.ray_triangle import RayMeshIntersector
import quaternion
from tqdm import tqdm

# Add project root to path (PATTERN from generate_lightcurves_pytorch.py)
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

# LCAS imports (after path setup)
from src.io.model_io import load_satellite_from_yaml  # noqa: E402
from src.config.model_config import ModelConfigManager  # noqa: E402


def sample_uniform_sphere(num_samples: int) -> np.ndarray:
    """
    Generate uniformly distributed 3D unit vectors for sampling sun directions.
    
    This function uses the Gaussian normalization method to generate points
    uniformly distributed on the unit sphere. Unlike naive spherical coordinate
    sampling (uniform θ and φ), this method avoids clustering at the poles.
    
    The mathematical principle: When 3D Gaussian random variables are normalized
    to unit length, they produce a uniform distribution on the sphere. This is
    because the multivariate Gaussian distribution is rotationally symmetric,
    and normalization projects points onto the unit sphere while preserving
    this symmetry.
    
    The naive approach of sampling uniform θ ∈ [0, π] and φ ∈ [0, 2π] causes
    clustering at poles due to the Jacobian of the spherical coordinate 
    transformation: dA = sin(θ)dθdφ. The sin(θ) factor means equal increments
    in θ near the poles (θ ≈ 0 or π) correspond to smaller surface areas.
    
    Args:
        num_samples: Number of uniform samples to generate on the unit sphere.
                    Must be a positive integer.
    
    Returns:
        A NumPy array of shape (num_samples, 3) containing unit vectors
        uniformly distributed on the unit sphere.
    
    Raises:
        ValueError: If num_samples is not a positive integer.
    
    Example:
        >>> sun_directions = sample_uniform_sphere(1000)
        >>> print(sun_directions.shape)
        (1000, 3)
        >>> # Verify unit length
        >>> norms = np.linalg.norm(sun_directions, axis=1)
        >>> assert np.allclose(norms, 1.0)
    """
    if not isinstance(num_samples, int) or num_samples <= 0:
        raise ValueError(f"num_samples must be a positive integer, got {num_samples}")
    
    # Generate random 3D Gaussian vectors
    # randn samples from standard normal distribution N(0, 1)
    random_vectors = np.random.randn(num_samples, 3)
    
    # Normalize each vector to unit length
    # axis=1 computes norm along each row (each 3D vector)
    # keepdims=True preserves the shape for broadcasting
    norms = np.linalg.norm(random_vectors, axis=1, keepdims=True)
    
    # Handle edge case where a vector might have zero norm (extremely rare)
    # Replace zero norms with 1 to avoid division by zero
    norms = np.where(norms == 0, 1, norms)
    
    # Normalize to create unit vectors
    unit_vectors = random_vectors / norms
    
    return unit_vectors


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
        
        # Create the combined satellite mesh for efficient shadow calculation
        self.logger.info("Creating combined satellite mesh for shadow calculations...")
        self._combined_mesh = self._create_combined_satellite_mesh()
        if self._combined_mesh is None:
            self.logger.warning("Failed to create combined satellite mesh - shadow calculations may not work")
        else:
            self.logger.info(f"Combined mesh created: {len(self._combined_mesh.vertices)} vertices, "
                           f"{len(self._combined_mesh.faces)} faces")
    
    def _create_combined_satellite_mesh(self) -> Optional[trimesh.Trimesh]:
        """
        Create combined satellite mesh for shadow calculations.
        
        This method builds a single trimesh object representing the entire satellite
        geometry by combining all component facets with proper transformations.
        Based on create_shadow_mesh from pytorch_shadow_engine.py but simplified
        for static (non-articulated) configuration.
        
        Returns:
            Combined trimesh object or None if no valid components found
        """
        component_meshes = []
        
        for component in self.satellite.components:
            if not component.facets:  # GOTCHA: Some components may be empty
                continue
            
            # Extract vertices and faces from facets (PATTERN from pytorch_shadow_engine.py)
            vertices_list: list[np.ndarray] = []
            faces_list: list[list[int]] = []
            vertex_offset = 0
            
            for facet in component.facets:
                vertices_list.extend(facet.vertices)  # facet.vertices is List[np.ndarray]
                faces_list.append([vertex_offset, vertex_offset + 1, vertex_offset + 2])
                vertex_offset += 3
            
            if not faces_list:  # GOTCHA: Skip if no valid facets
                continue
            
            # Create local component mesh
            local_mesh = trimesh.Trimesh(
                vertices=np.array(vertices_list),
                faces=np.array(faces_list),
                process=False  # CRITICAL: Disable auto-processing for performance
            )
            
            # Apply component transformation to body frame
            transform = np.eye(4)
            transform[:3, :3] = quaternion.as_rotation_matrix(component.relative_orientation)
            transform[:3, 3] = component.relative_position
            
            # Transform mesh to body frame
            body_mesh = local_mesh.copy()
            body_mesh.apply_transform(transform)
            component_meshes.append(body_mesh)
        
        if not component_meshes:  # GOTCHA: Handle case with no valid components
            return None
        
        # Combine all meshes (CRITICAL: trimesh.util.concatenate expects non-empty list)
        combined_mesh = trimesh.util.concatenate(component_meshes)
        combined_mesh.process()  # CRITICAL: Process after concatenation for optimization
        
        # Optimize for ray tracing
        combined_mesh.merge_vertices()
        combined_mesh.remove_degenerate_faces()
        
        return combined_mesh
    
    def calculate_lit_fractions(self, sun_vector_body: np.ndarray) -> Dict[str, float]:
        """
        Calculate lit fractions for each conceptual face using ray tracing.
        
        This method performs ray tracing from facet centers towards the provided
        sun direction to determine which portions of each conceptual face are
        illuminated (not in shadow).
        
        Args:
            sun_vector_body: 3D unit vector representing sun direction in satellite body frame
            
        Returns:
            Dictionary mapping conceptual face names to lit fractions (0.0 to 1.0)
            where 1.0 means fully lit and 0.0 means fully shadowed
            
        Raises:
            RuntimeError: If combined satellite mesh is not available
            ValueError: If sun_vector_body is not a valid 3D numpy array
        """
        # Input validation (PATTERN: Always validate inputs first)
        if self._combined_mesh is None:
            raise RuntimeError("Combined satellite mesh not available")
            
        if not isinstance(sun_vector_body, np.ndarray) or sun_vector_body.shape != (3,):
            raise ValueError("sun_vector_body must be 3D numpy array")
            
        # Normalize to unit vector (GOTCHA: Ensure unit length)
        sun_vector_norm = np.linalg.norm(sun_vector_body)
        if sun_vector_norm == 0:
            raise ValueError("sun_vector_body cannot be zero vector")
        sun_vector_body = sun_vector_body / sun_vector_norm
        
        # Create ray intersector once for all queries
        ray_intersector = RayMeshIntersector(self._combined_mesh)
        
        lit_fractions = {}
        total_rays_cast = 0
        total_hits = 0
        
        # Process each component and its conceptual faces
        for component in self.satellite.components:
            if not component.facets or not component.conceptual_faces_map:
                continue
                
            # Get component transformation
            comp_rot_matrix = quaternion.as_rotation_matrix(component.relative_orientation)
            comp_pos = component.relative_position
            
            # Process each conceptual face
            for face_name, facet_indices in component.conceptual_faces_map.items():
                lit_count = 0
                forward_facing_count = 0
                
                for facet_idx in facet_indices:
                    if facet_idx >= len(component.facets):  # GOTCHA: Index bounds check
                        continue
                        
                    facet = component.facets[facet_idx]
                    
                    # Transform facet normal and centroid to body frame
                    facet_normal_body = comp_rot_matrix @ facet.normal
                    facet_centroid_local = np.mean(facet.vertices, axis=0)
                    facet_center_body = comp_rot_matrix @ facet_centroid_local + comp_pos
                    
                    # Back-face culling (CRITICAL: Only forward-facing facets can be lit)
                    dot_product = np.dot(facet_normal_body, sun_vector_body)
                    if dot_product <= 0:
                        continue  # Back-facing facet
                        
                    forward_facing_count += 1
                    
                    # Create ray with epsilon offset (PATTERN from validation code)
                    epsilon = 1e-2
                    ray_origin = facet_center_body + facet_normal_body * epsilon
                    ray_direction = sun_vector_body
                    
                    # Perform ray tracing
                    try:
                        locations, ray_indices, _ = ray_intersector.intersects_location(
                            ray_origin[np.newaxis, :], ray_direction[np.newaxis, :], 
                            multiple_hits=False
                        )
                        
                        total_rays_cast += 1
                        if len(ray_indices) == 0:
                            lit_count += 1  # No intersection, facet is lit
                        else:
                            total_hits += 1  # Intersection found, facet is shadowed
                            
                    except Exception as e:
                        self.logger.warning(f"Ray tracing error for {face_name}: {e}")
                        lit_count += 1  # Assume lit on error
                
                # Calculate lit fraction for this conceptual face
                if forward_facing_count > 0:
                    lit_fractions[face_name] = lit_count / forward_facing_count
                else:
                    lit_fractions[face_name] = 1.0  # Default to fully lit if no forward-facing facets
        
        # Log statistics
        self.logger.info(f"Ray tracing complete: {total_rays_cast} rays cast, "
                        f"{total_hits} hits, {len(lit_fractions)} faces processed")
        
        return lit_fractions
    
    def run_generation(self, num_samples: int, output_path: str) -> None:
        """
        Generate training data samples for surrogate model training.
        
        This method generates uniformly distributed sun vectors, calculates lit fractions
        for each conceptual face using ray tracing, and outputs structured CSV data.
        
        Args:
            num_samples: Number of data samples to generate. Must be positive.
            output_path: Path where the generated CSV data will be saved.
            
        Raises:
            ValueError: If num_samples is not positive or output path is invalid.
            RuntimeError: If data generation or file writing fails.
        """
        # Input validation (PATTERN: Always validate inputs first)
        if not isinstance(num_samples, int) or num_samples <= 0:
            raise ValueError(f"Number of samples must be positive integer, got {num_samples}")
        
        # Convert to Path object and validate
        output_path_obj = Path(output_path)
        
        # Create output directory if needed (PATTERN: Ensure directories exist)
        output_dir = output_path_obj.parent
        if output_dir and not output_dir.exists():
            self.logger.info(f"Creating output directory: {output_dir}")
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Log generation start (PATTERN: Informative logging)
        self.logger.info(f"Starting generation of {num_samples} samples...")
        self.logger.info(f"Using satellite model: {self.satellite.name}")
        self.logger.info(f"Number of components: {len(self.satellite.components)}")
        
        # Count total facets for reference
        total_facets = sum(len(comp.facets) for comp in self.satellite.components if comp.facets)
        self.logger.info(f"Total facets in model: {total_facets}")
        
        try:
            # STEP 1: Generate uniformly distributed sun vectors
            self.logger.info("Generating uniform sun vectors...")
            sun_vectors = sample_uniform_sphere(num_samples)
            self.logger.info(f"Generated {len(sun_vectors)} sun vectors")
            
            # STEP 2: Determine face names from first sample (CRITICAL: Consistent ordering)
            self.logger.info("Determining conceptual face structure...")
            sample_lit_fractions = self.calculate_lit_fractions(sun_vectors[0])
            face_names = list(sample_lit_fractions.keys())
            self.logger.info(f"Found {len(face_names)} conceptual faces: {face_names}")
            
            if not face_names:
                raise RuntimeError("No conceptual faces found in satellite model")
            
            # STEP 3: Data collection with progress tracking
            self.logger.info("Starting data collection loop...")
            data_rows = []
            failed_samples = 0
            
            # PATTERN: Use tqdm for user-facing progress tracking
            for i, sun_vector in tqdm(enumerate(sun_vectors), 
                                    total=num_samples,
                                    desc="Generating training data",
                                    unit="samples"):
                try:
                    # Calculate lit fractions for this sun vector
                    lit_fractions = self.calculate_lit_fractions(sun_vector)
                    
                    # STRUCTURE: [sun_x, sun_y, sun_z, face1_lit, face2_lit, ...]
                    # Maintain consistent face ordering across all samples
                    face_values = [lit_fractions.get(face_name, 0.0) for face_name in face_names]
                    row = [sun_vector[0], sun_vector[1], sun_vector[2]] + face_values
                    data_rows.append(row)
                    
                    # PATTERN: Logging milestones like pytorch_shadow_engine.py
                    if (i + 1) % 50 == 0:
                        self.logger.info(f"  Processed {i + 1}/{num_samples} samples...")
                        
                except Exception as e:
                    # DECISION: Log error but continue (skip failed samples)
                    self.logger.warning(f"Failed to process sample {i}: {e}")
                    failed_samples += 1
                    continue
            
            # Validate we have data to save
            if not data_rows:
                raise RuntimeError("No valid samples generated - all samples failed")
                
            successful_samples = len(data_rows)
            self.logger.info(f"Data collection complete: {successful_samples} successful, {failed_samples} failed")
            
            # STEP 4: Create structured data array (PATTERN from generate_lightcurves_pytorch.py)
            self.logger.info("Creating structured data array...")
            data_array = np.array(data_rows)
            self.logger.info(f"Data array shape: {data_array.shape}")
            
            # STEP 5: Create CSV header string
            header = "sun_vec_x,sun_vec_y,sun_vec_z," + ",".join(face_names)
            self.logger.info(f"CSV header: {header}")
            
            # STEP 6: Save to CSV using numpy.savetxt (PATTERN from generate_lightcurves_pytorch.py:538)
            self.logger.info(f"Saving data to {output_path_obj}...")
            np.savetxt(output_path_obj, data_array, delimiter=',', 
                      header=header, fmt='%.6f', comments='')
            
            # Final success logging
            self.logger.info(f"Successfully generated {successful_samples} samples saved to {output_path_obj}")
            if failed_samples > 0:
                self.logger.warning(f"Note: {failed_samples} samples failed during generation")
                
        except Exception as e:
            self.logger.error(f"Data generation failed: {e}")
            raise RuntimeError(f"Data generation failed: {e}") from e


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic training data from satellite models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python surrogate_data_generator.py --config intelsat_901_model.yaml --samples 1000 --output data/training/is901_data.csv
  python surrogate_data_generator.py --config path/to/custom_model.yaml --samples 500 --output output.csv
        """
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        required=True,
        help="Path to model configuration YAML file"
    )
    
    parser.add_argument(
        "--samples",
        type=int,
        required=True,
        help="Number of training samples to generate"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output file path for generated data (e.g., output.csv)"
    )
    
    return parser.parse_args()


def main() -> None:
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
        
        logging.info("Data generation completed successfully")
        
    except Exception as e:
        logging.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
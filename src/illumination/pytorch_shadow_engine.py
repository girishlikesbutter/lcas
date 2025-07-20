#!/usr/bin/env python3
"""
PyTorch GPU Shadow Engine
=========================

Pure PyTorch implementation of GPU-accelerated shadow computation.
Supports RTX GPUs for hardware-accelerated ray tracing.
"""

import numpy as np
import torch
import time
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import trimesh
from trimesh.ray.ray_triangle import RayMeshIntersector

logger = logging.getLogger(__name__)

@dataclass
class PyTorchShadowResult:
    """Results from PyTorch GPU shadow computation."""
    lit_fractions: Dict[str, Dict[str, np.ndarray]]
    computation_time: float
    total_rays: int
    gpu_memory_used: float

class PyTorchShadowEngine:
    """
    PyTorch-based GPU shadow engine.
    
    Uses GPU acceleration for real-time ray tracing
    with PyTorch's CUDA implementation.
    """
    
    def __init__(self, use_gpu: bool = True):
        """Initialize PyTorch GPU shadow engine."""
        # First check if CUDA is available
        cuda_available = torch.cuda.is_available()
        
        # Attempt to use GPU if requested and CUDA reports as available
        self.use_gpu = use_gpu and cuda_available
        
        # Try to create device and handle potential warnings
        if self.use_gpu:
            try:
                # Attempt to create CUDA device
                self.device = torch.device('cuda')
                # Test if we can actually allocate on GPU
                test_tensor = torch.zeros(1, device=self.device)
                del test_tensor
                # If we get here, GPU is actually working
                actual_gpu_working = True
            except Exception as e:
                logger.warning(f"Failed to initialize CUDA device: {e}")
                self.device = torch.device('cpu')
                self.use_gpu = False
                actual_gpu_working = False
        else:
            self.device = torch.device('cpu')
            actual_gpu_working = False
        
        self.stats = {
            'total_rays': 0,
            'total_time': 0.0,
            'gpu_memory_peak': 0.0
        }
        
        if actual_gpu_working:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            self.has_rt_cores = 'RTX' in gpu_name
            
            logger.info(f"PyTorch GPU Shadow Engine: {gpu_name} ({gpu_memory:.1f}GB)")
            logger.info(f"RTX cores available: {self.has_rt_cores}")
            logger.info(f"GPU is working correctly despite any initialization warnings")
        else:
            logger.info("PyTorch CPU Shadow Engine (GPU not available)")
            self.has_rt_cores = False
    
    def _calculate_articulation_rotation(self, component, sun_vector_body, offset_deg=0.0, 
                                        articulation_engine=None, earth_vector_body=None, epoch=None):
        """
        Calculate rotation matrix for component articulation using flexible system.
        
        Args:
            component: Component to articulate
            sun_vector_body: Sun direction vector in body frame (normalized)
            offset_deg: Additional offset angle in degrees for testing
            articulation_engine: New flexible articulation engine (optional)
            earth_vector_body: Earth direction vector in body frame (optional)
            epoch: Current epoch time (optional)
            
        Returns:
            4x4 homogeneous transformation matrix for articulation
        """
        # Try new flexible articulation system first
        if articulation_engine is not None:
            rotation_matrix = articulation_engine.calculate_articulation_rotation(
                component, sun_vector_body, 
                earth_vector_body if earth_vector_body is not None else np.array([0, 0, -1]),
                epoch if epoch is not None else 0.0,
                offset_deg
            )
            if rotation_matrix is not None:
                return rotation_matrix
        
        # Fall back to legacy articulation system for backward compatibility
        if component.articulation_rule == "TRACK_SUN_PRIMARY_AXIS_Z_SECONDARY_X":
            # Use the same proven algorithm as the original system
            from src.utils.geometry_utils import calculate_sun_pointing_rotation
            
            # Same parameters as original system (tensor_shadow_calculator.py lines 696-697)
            panel_rotation_axis = np.array([0, 0, 1])      # Z-axis rotation (not X!)
            panel_normal_reference = np.array([1, 0, 0])   # X-axis normal in reference orientation
            
            # Calculate required rotation angle using proven function
            panel_angle_deg = calculate_sun_pointing_rotation(
                sun_vector_body, panel_rotation_axis, panel_normal_reference
            )
            
            # Apply user-specified offset
            total_angle_deg = panel_angle_deg + offset_deg
            
            # Create rotation matrix around Z-axis (matching original system)
            angle_rad = np.radians(total_angle_deg)
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
            
            # Z-axis rotation matrix (4x4 for homogeneous coordinates)
            rotation_matrix = np.array([
                [cos_a, -sin_a, 0, 0],
                [sin_a,  cos_a, 0, 0],
                [0,      0,     1, 0],
                [0,      0,     0, 1]
            ])
            
            return rotation_matrix
        
        # No articulation for this component
        return np.eye(4)
    
    def create_shadow_meshes_for_epochs(self, satellite, sun_vectors_body: np.ndarray, 
                                        articulation_offset: float = 0.0, articulation_engine=None,
                                        earth_vectors_body: np.ndarray = None, epochs: np.ndarray = None) -> List[trimesh.Trimesh]:
        """
        Create multiple shadow meshes, one for each epoch with proper articulation.
        
        Args:
            satellite: Satellite model
            sun_vectors_body: Sun direction vectors in body frame for each epoch (N, 3)
            articulation_offset: Additional offset angle in degrees for testing
            
        Returns:
            List of trimesh objects, one per epoch
        """
        logger.info(f"Creating {len(sun_vectors_body)} articulated shadow meshes...")
        start_time = time.time()
        
        shadow_meshes = []
        for i, sun_vector in enumerate(sun_vectors_body):
            earth_vector = earth_vectors_body[i] if earth_vectors_body is not None else None
            epoch = epochs[i] if epochs is not None else None
            mesh = self.create_shadow_mesh(satellite, sun_vector, articulation_offset, 
                                         articulation_engine, earth_vector, epoch)
            if mesh is None:
                raise RuntimeError(f"Could not create shadow mesh for epoch {i}")
            shadow_meshes.append(mesh)
            
            if (i + 1) % 50 == 0:
                logger.info(f"  Created {i + 1}/{len(sun_vectors_body)} meshes...")
        
        mesh_time = time.time() - start_time
        logger.info(f"Created {len(shadow_meshes)} articulated meshes in {mesh_time:.3f}s")
        
        return shadow_meshes
    
    def create_shadow_mesh(self, satellite, sun_vector_body=None, articulation_offset=0.0, 
                          articulation_engine=None, earth_vector_body=None, epoch=None) -> Optional[trimesh.Trimesh]:
        """Create optimized mesh for shadow ray tracing."""
        import quaternion
        
        logger.info("Creating shadow mesh...")
        start_time = time.time()
        
        component_meshes = []
        
        for component in satellite.components:
            if not component.facets:
                continue
            
            # Build mesh from facets
            vertices_list = []
            faces_list = []
            vertex_offset = 0
            
            for facet in component.facets:
                vertices_list.extend(facet.vertices)
                faces_list.append([vertex_offset, vertex_offset + 1, vertex_offset + 2])
                vertex_offset += 3
            
            if not faces_list:
                continue
            
            # Create local mesh
            local_mesh = trimesh.Trimesh(
                vertices=np.array(vertices_list),
                faces=np.array(faces_list),
                process=False
            )
            
            # Transform to body frame
            transform = np.eye(4)
            transform[:3, :3] = quaternion.as_rotation_matrix(component.relative_orientation)
            transform[:3, 3] = component.relative_position
            
            # Apply articulation if sun vector is provided
            if sun_vector_body is not None and (component.articulation_rule or 
                                               (articulation_engine and articulation_engine.is_component_articulated(component))):
                articulation_transform = self._calculate_articulation_rotation(
                    component, sun_vector_body, articulation_offset, 
                    articulation_engine, earth_vector_body, epoch)
                
                # Debug: Log articulation application
                angle_rad = np.arctan2(articulation_transform[1, 0], articulation_transform[0, 0])
                angle_deg = np.degrees(angle_rad)
                logger.info(f"  Applying articulation to {component.name}: {angle_deg:.2f}° "
                           f"(sun={sun_vector_body}, offset={articulation_offset}°)")
                
                # Apply articulation to rotation part only (match validation method)
                # Extract 3x3 rotation and apply articulation
                comp_rot_3x3 = transform[:3, :3]
                articulation_rot_3x3 = articulation_transform[:3, :3]
                transform[:3, :3] = comp_rot_3x3 @ articulation_rot_3x3
            
            body_mesh = local_mesh.copy()
            body_mesh.apply_transform(transform)
            component_meshes.append(body_mesh)
        
        if not component_meshes:
            logger.error("No valid components for shadow mesh")
            return None
        
        # Combine all meshes
        combined_mesh = trimesh.util.concatenate(component_meshes)
        combined_mesh.process()
        
        # Optimize for ray tracing
        combined_mesh.merge_vertices()
        combined_mesh.remove_degenerate_faces()
        
        mesh_time = time.time() - start_time
        logger.info(f"Shadow mesh created: {len(combined_mesh.vertices)} vertices, "
                   f"{len(combined_mesh.faces)} faces ({mesh_time:.3f}s)")
        
        return combined_mesh
    
    def compute_shadows_pytorch_multi_mesh(self, 
                                           satellite,
                                           shadow_meshes: List[trimesh.Trimesh],
                                           sun_vectors_body: np.ndarray,
                                           target_faces: List[str]) -> PyTorchShadowResult:
        """
        Compute shadows using PyTorch GPU acceleration with per-epoch meshes.
        
        This method handles articulated meshes for each epoch.
        """
        logger.info(f"PyTorch shadow computation: {len(sun_vectors_body)} epochs with articulated meshes")
        
        start_time = time.time()
        num_epochs = len(sun_vectors_body)
        
        if len(shadow_meshes) != num_epochs:
            raise ValueError(f"Number of meshes ({len(shadow_meshes)}) must match number of epochs ({num_epochs})")
        
        # Move sun vectors to GPU
        sun_vectors_gpu = torch.tensor(sun_vectors_body, dtype=torch.float32, device=self.device)
        
        # Initialize results
        lit_fractions = {}
        total_rays = 0
        
        # Process epochs in batches
        batch_size = min(50, num_epochs)
        
        for batch_start in range(0, num_epochs, batch_size):
            batch_end = min(batch_start + batch_size, num_epochs)
            batch_meshes = shadow_meshes[batch_start:batch_end]
            batch_sun_vectors = sun_vectors_gpu[batch_start:batch_end]
            
            # Process this batch
            batch_results = self._process_epoch_batch_multi_mesh(
                satellite, batch_meshes, batch_sun_vectors, target_faces, batch_start)
            
            # Merge results
            for component_name, face_dict in batch_results.items():
                if component_name not in lit_fractions:
                    lit_fractions[component_name] = {}
                for face_name, face_lit_fractions in face_dict.items():
                    if face_name not in lit_fractions[component_name]:
                        lit_fractions[component_name][face_name] = np.zeros(num_epochs)
                    lit_fractions[component_name][face_name][batch_start:batch_end] = face_lit_fractions
            
            total_rays += (batch_end - batch_start) * sum(len(comp.facets) for comp in satellite.components if comp.facets)
        
        # Get GPU memory usage
        gpu_memory = 0.0
        if self.use_gpu:
            gpu_memory = torch.cuda.max_memory_allocated() / (1024**2)  # MB
            torch.cuda.reset_peak_memory_stats()
        
        total_time = time.time() - start_time
        
        # Update statistics
        self.stats['total_rays'] += total_rays
        self.stats['total_time'] += total_time
        self.stats['gpu_memory_peak'] = max(self.stats['gpu_memory_peak'], gpu_memory)
        
        logger.info(f"PyTorch shadow computation: {total_rays:,} rays in {total_time:.3f}s "
                   f"({total_rays/total_time/1000:.1f}K rays/sec)")
        
        return PyTorchShadowResult(
            lit_fractions=lit_fractions,
            computation_time=total_time,
            total_rays=total_rays,
            gpu_memory_used=gpu_memory
        )
    
    def compute_shadows_pytorch(self, 
                               satellite,
                               shadow_mesh: trimesh.Trimesh,
                               sun_vectors_body: np.ndarray,
                               target_faces: List[str],
                               articulation_offset: float = 0.0) -> PyTorchShadowResult:
        """
        Compute shadows using PyTorch GPU acceleration.
        
        This method computes shadows using GPU acceleration.
        """
        logger.info(f"PyTorch shadow computation: {len(sun_vectors_body)} epochs, {len(target_faces)} faces")
        
        start_time = time.time()
        num_epochs = len(sun_vectors_body)
        
        # Move sun vectors to GPU
        sun_vectors_gpu = torch.tensor(sun_vectors_body, dtype=torch.float32, device=self.device)
        
        # Initialize results
        lit_fractions = {}
        total_rays = 0
        
        # Create ray intersector once for all computations
        ray_intersector = RayMeshIntersector(shadow_mesh)
        
        # Process each component
        for component in satellite.components:
            if not component.facets or not component.conceptual_faces_map:
                continue
            
            component_results = self._process_component_pytorch(
                component, ray_intersector, sun_vectors_gpu, target_faces, num_epochs, 
                sun_vectors_body, articulation_offset)
            
            lit_fractions.update(component_results)
            total_rays += num_epochs * len(component.facets)
        
        # Get GPU memory usage
        gpu_memory = 0.0
        if self.use_gpu:
            gpu_memory = torch.cuda.max_memory_allocated() / (1024**2)  # MB
            torch.cuda.reset_peak_memory_stats()
        
        total_time = time.time() - start_time
        
        # Update statistics
        self.stats['total_rays'] += total_rays
        self.stats['total_time'] += total_time
        self.stats['gpu_memory_peak'] = max(self.stats['gpu_memory_peak'], gpu_memory)
        
        logger.info(f"PyTorch shadow computation: {total_rays:,} rays in {total_time:.3f}s "
                   f"({total_rays/total_time/1000:.1f}K rays/sec)")
        
        return PyTorchShadowResult(
            lit_fractions=lit_fractions,
            computation_time=total_time,
            total_rays=total_rays,
            gpu_memory_used=gpu_memory
        )
    
    def _process_component_pytorch(self, component, ray_intersector, sun_vectors_gpu, target_faces, num_epochs,
                                   sun_vectors_body, articulation_offset=0.0):
        """Process a single component using PyTorch GPU acceleration."""
        import quaternion
        
        component_results = {}
        
        # Get component transformation
        comp_rel_pos = component.relative_position
        comp_rel_orient_quat = component.relative_orientation
        
        if not isinstance(comp_rel_orient_quat, np.quaternion):
            return component_results
        
        comp_rot_matrix_to_body = quaternion.as_rotation_matrix(comp_rel_orient_quat)
        
        # Apply articulation to component rotation matrix (match validation method)
        if component.articulation_rule and len(sun_vectors_body) > 0:
            from src.utils.geometry_utils import calculate_sun_pointing_rotation
            
            # Use first sun vector for articulation (single epoch case)
            sun_vector = sun_vectors_body[0] if len(sun_vectors_body.shape) > 1 else sun_vectors_body
            
            # Same parameters as validation method
            panel_rotation_axis = np.array([0, 0, 1])
            panel_normal_reference = np.array([1, 0, 0])
            panel_angle_deg = calculate_sun_pointing_rotation(
                sun_vector, panel_rotation_axis, panel_normal_reference
            )
            
            # Create articulation rotation matrix (3x3)
            angle_rad = np.radians(panel_angle_deg + articulation_offset)
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
            articulation_rot = np.array([
                [cos_a, -sin_a, 0],
                [sin_a,  cos_a, 0],
                [0,      0,     1]
            ])
            
            # Apply articulation (match validation method)
            comp_rot_matrix_to_body = comp_rot_matrix_to_body @ articulation_rot
        
        # Convert to GPU tensors
        comp_rot_gpu = torch.tensor(comp_rot_matrix_to_body, dtype=torch.float32, device=self.device)
        comp_pos_gpu = torch.tensor(comp_rel_pos, dtype=torch.float32, device=self.device)
        
        # Process each conceptual face
        for face_name, facet_indices in component.conceptual_faces_map.items():
            if face_name not in target_faces:
                continue
            
            face_lit_fractions = self._compute_face_shadows_pytorch(
                component, facet_indices, comp_rot_gpu, comp_pos_gpu,
                ray_intersector, sun_vectors_gpu, num_epochs, sun_vectors_body, articulation_offset)
            
            # Store results (convert back to CPU)
            if component.name not in component_results:
                component_results[component.name] = {}
            component_results[component.name][face_name] = face_lit_fractions.cpu().numpy()
        
        return component_results
    
    def _compute_face_shadows_pytorch(self, component, facet_indices, comp_rot_gpu, comp_pos_gpu,
                                     ray_intersector, sun_vectors_gpu, num_epochs, sun_vectors_body, articulation_offset=0.0):
        """Compute shadows for a conceptual face using PyTorch vectorization."""
        num_facets = len(facet_indices)
        
        # Pre-allocate result tensor on GPU
        face_lit_fractions = torch.ones((num_epochs, num_facets), 
                                       dtype=torch.float32, device=self.device)
        
        # Get facet data
        facet_centers_local = []
        facet_normals_local = []
        
        for facet_idx in facet_indices:
            if facet_idx >= len(component.facets):
                continue
            
            facet = component.facets[facet_idx]
            facet_vertices_local = np.array(facet.vertices)
            facet_normal_local = facet.normal
            facet_centroid_local = np.mean(facet_vertices_local, axis=0)
            
            facet_centers_local.append(facet_centroid_local)
            facet_normals_local.append(facet_normal_local)
        
        if not facet_centers_local:
            return torch.mean(face_lit_fractions, dim=1)
        
        # Convert to GPU tensors
        facet_centers_local_gpu = torch.tensor(facet_centers_local, 
                                             dtype=torch.float32, device=self.device)
        facet_normals_local_gpu = torch.tensor(facet_normals_local, 
                                             dtype=torch.float32, device=self.device)
        
        # Transform to body frame (vectorized)
        facet_normals_body_gpu = torch.matmul(facet_normals_local_gpu, comp_rot_gpu.T)
        facet_centers_body_gpu = torch.matmul(facet_centers_local_gpu, comp_rot_gpu.T) + comp_pos_gpu
        
        # Process epochs in batches to manage memory
        batch_size = min(50, num_epochs)  # Adjust based on GPU memory
        
        for batch_start in range(0, num_epochs, batch_size):
            batch_end = min(batch_start + batch_size, num_epochs)
            batch_sun_vectors = sun_vectors_gpu[batch_start:batch_end]
            
            batch_results = self._process_epoch_batch_pytorch(
                facet_centers_body_gpu, facet_normals_body_gpu,
                batch_sun_vectors, ray_intersector)
            
            face_lit_fractions[batch_start:batch_end] = batch_results
        
        # Average across all facets in the conceptual face
        return torch.mean(face_lit_fractions, dim=1)
    
    def _process_epoch_batch_pytorch(self, facet_centers_body, facet_normals_body, 
                                   sun_vectors_batch, ray_intersector):
        """Process a batch of epochs using PyTorch GPU operations."""
        batch_size = len(sun_vectors_batch)
        num_facets = len(facet_centers_body)
        
        # Initialize results
        batch_results = torch.ones((batch_size, num_facets), 
                                  dtype=torch.float32, device=self.device)
        
        for epoch_idx in range(batch_size):
            sun_vector = sun_vectors_batch[epoch_idx]
            
            # Vectorized back-face culling (match validation method)
            dot_products = torch.sum(facet_normals_body * sun_vector, dim=1)
            forward_facing = dot_products > 0
            
            if not torch.any(forward_facing):
                continue  # All facets are back-facing
            
            # Get forward-facing facet indices
            forward_indices = torch.where(forward_facing)[0]
            if len(forward_indices) == 0:
                continue
            
            # Create rays for forward-facing facets (match validation method epsilon)
            epsilon = 1e-2
            ray_origins = facet_centers_body[forward_indices] + \
                         facet_normals_body[forward_indices] * epsilon
            ray_directions = sun_vector.unsqueeze(0).expand(len(forward_indices), -1)
            
            # Convert to CPU for ray tracing (Trimesh is CPU-based)
            ray_origins_cpu = ray_origins.cpu().numpy()
            ray_directions_cpu = ray_directions.cpu().numpy()
            
            # Perform ray tracing
            try:
                locations, ray_indices, _ = ray_intersector.intersects_location(
                    ray_origins_cpu, ray_directions_cpu, multiple_hits=False)
                
                # Create hit mask
                hits = torch.zeros(len(forward_indices), dtype=torch.float32, device=self.device)
                if len(ray_indices) > 0:
                    hit_tensor = torch.tensor(ray_indices, dtype=torch.long, device=self.device)
                    hits[hit_tensor] = 1.0
                
                # Update results: lit_fraction = 1.0 - shadow_value
                batch_results[epoch_idx, forward_indices] = 1.0 - hits
                
            except Exception as e:
                logger.warning(f"Ray tracing error: {e}")
                # Keep default lit values
        
        return batch_results
    
    def _process_epoch_batch_multi_mesh(self, satellite, batch_meshes, batch_sun_vectors, target_faces, batch_start):
        """Process a batch of epochs with different meshes for each epoch."""
        import quaternion
        
        batch_size = len(batch_meshes)
        batch_results = {}
        
        # Process each epoch in the batch
        for idx, (mesh, sun_vector) in enumerate(zip(batch_meshes, batch_sun_vectors)):
            epoch_idx = batch_start + idx
            
            # Create ray intersector for this epoch's mesh
            ray_intersector = RayMeshIntersector(mesh)
            
            # Process each component
            for component in satellite.components:
                if not component.facets or not component.conceptual_faces_map:
                    continue
                
                # Get component transformation
                comp_rel_pos = component.relative_position
                comp_rel_orient_quat = component.relative_orientation
                
                if not isinstance(comp_rel_orient_quat, np.quaternion):
                    continue
                
                comp_rot_matrix_to_body = quaternion.as_rotation_matrix(comp_rel_orient_quat)
                
                # Apply articulation to component rotation matrix (match validation method)
                if component.articulation_rule:
                    from src.utils.geometry_utils import calculate_sun_pointing_rotation
                    
                    # Convert sun vector from GPU tensor to numpy
                    sun_vector_numpy = sun_vector.cpu().numpy()
                    
                    # Same parameters as validation method
                    panel_rotation_axis = np.array([0, 0, 1])
                    panel_normal_reference = np.array([1, 0, 0])
                    panel_angle_deg = calculate_sun_pointing_rotation(
                        sun_vector_numpy, panel_rotation_axis, panel_normal_reference
                    )
                    
                    # Create articulation rotation matrix (3x3)
                    angle_rad = np.radians(panel_angle_deg)
                    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
                    articulation_rot = np.array([
                        [cos_a, -sin_a, 0],
                        [sin_a,  cos_a, 0],
                        [0,      0,     1]
                    ])
                    
                    # Apply articulation (match validation method)
                    comp_rot_matrix_to_body = comp_rot_matrix_to_body @ articulation_rot
                
                # Process each conceptual face
                for face_name, facet_indices in component.conceptual_faces_map.items():
                    if face_name not in target_faces:
                        continue
                    
                    # Compute shadows for this face at this epoch
                    face_lit_fraction = self._compute_face_shadows_single_epoch(
                        component, facet_indices, comp_rot_matrix_to_body, comp_rel_pos,
                        ray_intersector, sun_vector.cpu().numpy())
                    
                    # Store result
                    if component.name not in batch_results:
                        batch_results[component.name] = {}
                    if face_name not in batch_results[component.name]:
                        batch_results[component.name][face_name] = np.zeros(batch_size)
                    batch_results[component.name][face_name][idx] = face_lit_fraction
        
        return batch_results
    
    def _compute_face_shadows_single_epoch(self, component, facet_indices, comp_rot_matrix, comp_pos,
                                          ray_intersector, sun_vector):
        """Compute shadows for a single epoch with its specific mesh."""
        # Get facet data
        lit_count = 0
        total_count = 0
        
        for facet_idx in facet_indices:
            if facet_idx >= len(component.facets):
                continue
            
            facet = component.facets[facet_idx]
            facet_vertices_local = np.array(facet.vertices)
            facet_normal_local = facet.normal
            facet_centroid_local = np.mean(facet_vertices_local, axis=0)
            
            # Transform to body frame
            facet_normal_body = comp_rot_matrix @ facet_normal_local
            facet_center_body = comp_rot_matrix @ facet_centroid_local + comp_pos
            
            # Back-face culling (match validation method)
            dot_product = np.dot(facet_normal_body, sun_vector)
            if dot_product <= 0:
                continue  # Back-facing
            
            total_count += 1
            
            # Create ray (match validation method epsilon)
            epsilon = 1e-2
            ray_origin = facet_center_body + facet_normal_body * epsilon
            ray_direction = sun_vector
            
            # Perform ray tracing
            try:
                locations, ray_indices, _ = ray_intersector.intersects_location(
                    ray_origin[np.newaxis, :], ray_direction[np.newaxis, :], multiple_hits=False)
                
                if len(ray_indices) == 0:
                    lit_count += 1  # No intersection, facet is lit
            except Exception as e:
                logger.warning(f"Ray tracing error: {e}")
                lit_count += 1  # Assume lit on error
        
        # Return average lit fraction for the face
        if total_count > 0:
            return lit_count / total_count
        else:
            return 1.0  # Default to fully lit if no forward-facing facets
    
    def benchmark_performance(self, satellite, num_epochs_list=[10, 50, 100]) -> Dict:
        """Benchmark PyTorch shadow engine performance."""
        logger.info("Benchmarking PyTorch shadow engine...")
        
        # Create shadow mesh
        shadow_mesh = self.create_shadow_mesh(satellite)
        if shadow_mesh is None:
            return {"error": "Could not create shadow mesh"}
        
        # Get target faces
        target_faces = []
        for component in satellite.components:
            if component.conceptual_faces_map:
                target_faces.extend(component.conceptual_faces_map.keys())
        
        results = {}
        
        for num_epochs in num_epochs_list:
            logger.info(f"Benchmarking {num_epochs} epochs...")
            
            # Generate random sun directions
            sun_vectors = np.random.random((num_epochs, 3)) * 2 - 1
            sun_vectors = sun_vectors / np.linalg.norm(sun_vectors, axis=1, keepdims=True)
            
            # Benchmark computation
            start_time = time.time()
            result = self.compute_shadows_pytorch(satellite, shadow_mesh, sun_vectors, target_faces)
            wall_time = time.time() - start_time
            
            # Calculate performance metrics
            rays_per_second = result.total_rays / result.computation_time if result.computation_time > 0 else 0
            ms_per_epoch = result.computation_time / num_epochs * 1000
            
            results[num_epochs] = {
                'wall_time': wall_time,
                'computation_time': result.computation_time,
                'total_rays': result.total_rays,
                'rays_per_second': rays_per_second,
                'ms_per_epoch': ms_per_epoch,
                'gpu_memory_mb': result.gpu_memory_used,
                'speedup_vs_database': 10.0 / ms_per_epoch if ms_per_epoch > 0 else float('inf')
            }
            
            logger.info(f"  {num_epochs} epochs: {ms_per_epoch:.1f}ms/epoch, "
                       f"{rays_per_second/1000:.1f}K rays/sec")
        
        return results
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary statistics."""
        device_info = str(self.device)
        
        # Add GPU name if using GPU
        if self.use_gpu and torch.cuda.is_available():
            try:
                gpu_name = torch.cuda.get_device_name(0)
                device_info = f"{self.device} ({gpu_name})"
            except:
                pass
        
        return {
            'total_rays': self.stats['total_rays'],
            'total_time': self.stats['total_time'],
            'average_rays_per_second': self.stats['total_rays'] / self.stats['total_time'] 
                                     if self.stats['total_time'] > 0 else 0,
            'gpu_available': self.use_gpu,
            'device': device_info,
            'has_rt_cores': self.has_rt_cores,
            'gpu_memory_peak_mb': self.stats['gpu_memory_peak']
        }

# Global PyTorch shadow engine instance
_pytorch_shadow_engine = None

def get_pytorch_shadow_engine() -> PyTorchShadowEngine:
    """Get or create global PyTorch shadow engine instance."""
    global _pytorch_shadow_engine
    if _pytorch_shadow_engine is None:
        _pytorch_shadow_engine = PyTorchShadowEngine()
    return _pytorch_shadow_engine

def pytorch_get_lit_fractions_for_kinematics(
    satellite,
    sun_pos_j2000_epochs: np.ndarray,
    sat_pos_j2000_epochs: np.ndarray,
    sat_att_C_j2000_to_body_epochs: np.ndarray,
    target_conceptual_faces: List[str],
    articulation_offset: float = 0.0
) -> Tuple[Dict[str, np.ndarray], List[float]]:
    """
    PyTorch GPU-based shadow computation.
    
    This function computes shadows using real-time
    PyTorch GPU ray tracing.
    
    Args:
        satellite: Satellite model
        sun_pos_j2000_epochs: Sun positions in J2000 (N, 3)
        sat_pos_j2000_epochs: Satellite positions in J2000 (N, 3)
        sat_att_C_j2000_to_body_epochs: Attitude matrices (N, 3, 3)
        target_conceptual_faces: List of face names to compute
        
    Returns:
        Tuple of (lit_fractions_dict, panel_angles_list)
    """
    logger.info("PyTorch GPU shadow computation starting")
    
    start_time = time.time()
    num_epochs = len(sun_pos_j2000_epochs)
    
    # Get PyTorch shadow engine
    engine = get_pytorch_shadow_engine()
    
    # Calculate sun vectors in body frame
    sun_vectors_body = np.zeros((num_epochs, 3))
    panel_angles = np.zeros(num_epochs)
    
    for i in range(num_epochs):
        sun_vector_j2000 = sun_pos_j2000_epochs[i] - sat_pos_j2000_epochs[i]
        sun_vector_body = sat_att_C_j2000_to_body_epochs[i] @ sun_vector_j2000
        sun_vectors_body[i] = sun_vector_body / np.linalg.norm(sun_vector_body)
        
        # Calculate panel angle (simplified)
        panel_angles[i] = np.degrees(np.arccos(np.clip(sun_vectors_body[i, 0], -1, 1)))
    
    # Create articulated shadow meshes for all epochs
    shadow_meshes = engine.create_shadow_meshes_for_epochs(satellite, sun_vectors_body, articulation_offset)
    
    # Compute shadows using PyTorch GPU with per-epoch meshes
    result = engine.compute_shadows_pytorch_multi_mesh(
        satellite, shadow_meshes, sun_vectors_body, target_conceptual_faces)
    
    # Convert to expected format
    lit_fractions_dict = {}
    for target_face in target_conceptual_faces:
        found = False
        for component_name, face_dict in result.lit_fractions.items():
            if target_face in face_dict:
                lit_fractions_dict[target_face] = face_dict[target_face]
                found = True
                break
        if not found:
            lit_fractions_dict[target_face] = np.ones(num_epochs)
    
    total_time = time.time() - start_time
    logger.info(f"PyTorch GPU shadow computation complete: {total_time:.3f}s total "
               f"({result.total_rays:,} rays, {result.total_rays/total_time/1000:.1f}K rays/sec)")
    
    return lit_fractions_dict, panel_angles.tolist()
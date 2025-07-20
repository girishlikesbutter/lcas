#!/usr/bin/env python3
"""
Modular Shadow and Vectors Validation Animation
==============================================

Creates an animation showing any satellite model with:
- Satellite attitude changes in J2000 frame (satellite at origin)
- Per-facet shadow visualization
- Direction vectors to Earth, DST, and Sun in body frame

This modular version works with any satellite model configuration.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pathlib import Path
import logging
import quaternion
from trimesh.ray.ray_triangle import RayMeshIntersector
import trimesh
import argparse

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from src.io.model_io import load_satellite_from_yaml
from src.spice.spice_handler import SpiceHandler
from src.utils.geometry_utils import calculate_sun_pointing_rotation
from src.config.model_config import ModelConfigManager
from src.articulation.articulation_engine import ArticulationEngine

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Animation configuration
NUM_FRAMES = 120  # 2 frames per minute
DST_OBSERVER_ID = "1972"  # DST ground station
EARTH_RADIUS_KM = 6371.0

def create_shadow_mesh(satellite, articulation_engine, sun_vector_body, epoch):
    """Create a combined trimesh for shadow ray tracing with proper articulation."""
    all_meshes = []
    
    for component in satellite.components:
        if not component.facets:
            continue
        
        # Create mesh from facets
        vertices = []
        faces = []
        vertex_offset = 0
        
        for facet in component.facets:
            # Add vertices
            facet_vertices = np.array(facet.vertices)
            vertices.extend(facet_vertices)
            
            # Add face (assuming triangular facets)
            faces.append([vertex_offset, vertex_offset + 1, vertex_offset + 2])
            vertex_offset += 3
        
        if not vertices:
            continue
        
        # Create component mesh
        mesh = trimesh.Trimesh(vertices=np.array(vertices), faces=np.array(faces), process=False)
        
        # Apply component transformation
        transform = np.eye(4)
        transform[:3, :3] = quaternion.as_rotation_matrix(component.relative_orientation)
        transform[:3, 3] = component.relative_position
        
        # Apply articulation using the modular system
        if articulation_engine.is_component_articulated(component):
            try:
                articulation_rotation = articulation_engine.compute_articulation_rotation(
                    component, sun_vector_body, epoch
                )
                if articulation_rotation is not None:
                    articulation_transform = np.eye(4)
                    articulation_transform[:3, :3] = articulation_rotation
                    transform = transform @ articulation_transform
            except Exception as e:
                logger.debug(f"Articulation failed for {component.name}: {e}")
        
        mesh.apply_transform(transform)
        all_meshes.append(mesh)
    
    # Combine all meshes
    if all_meshes:
        combined_mesh = trimesh.util.concatenate(all_meshes)
        return combined_mesh
    else:
        # Return empty mesh if no components have facets
        return trimesh.Trimesh()

def compute_facet_shadows(satellite, sun_vector_body, shadow_mesh, articulation_engine, epoch):
    """Compute shadow status for each facet with proper articulation."""
    facet_shadows = {}
    
    # Create ray intersector
    ray_intersector = RayMeshIntersector(shadow_mesh)
    
    for component in satellite.components:
        if not component.facets:
            continue
        
        facet_shadows[component.name] = {}
        
        # Get component transformation
        comp_rot_matrix = quaternion.as_rotation_matrix(component.relative_orientation)
        comp_position = component.relative_position
        
        # Apply articulation using the modular system
        if articulation_engine.is_component_articulated(component):
            try:
                articulation_rotation = articulation_engine.compute_articulation_rotation(
                    component, sun_vector_body, epoch
                )
                if articulation_rotation is not None:
                    comp_rot_matrix = comp_rot_matrix @ articulation_rotation
            except Exception as e:
                logger.debug(f"Articulation failed for {component.name}: {e}")
        
        for facet_idx, facet in enumerate(component.facets):
            # Transform facet to body frame
            facet_normal_local = np.array(facet.normal)
            facet_centroid_local = np.mean(facet.vertices, axis=0)
            
            facet_normal_body = comp_rot_matrix @ facet_normal_local
            facet_center_body = comp_rot_matrix @ facet_centroid_local + comp_position
            
            # Check if facet faces the sun
            dot_product = np.dot(facet_normal_body, sun_vector_body)
            if dot_product <= 0:
                # Back-facing, shadowed
                facet_shadows[component.name][facet_idx] = 0.0
                continue
            
            # Ray trace to check for shadows
            epsilon = 1e-2  # Increase epsilon to avoid self-intersection
            ray_origin = facet_center_body + facet_normal_body * epsilon
            ray_direction = sun_vector_body
            
            # Check intersection
            try:
                locations, ray_indices, _ = ray_intersector.intersects_location(
                    ray_origin[np.newaxis, :], 
                    ray_direction[np.newaxis, :], 
                    multiple_hits=False
                )
                
                if len(ray_indices) == 0:
                    # No intersection, facet is lit
                    facet_shadows[component.name][facet_idx] = 1.0
                else:
                    # Intersection found, facet is shadowed
                    facet_shadows[component.name][facet_idx] = 0.0
            except Exception:
                # If ray tracing fails, assume lit
                facet_shadows[component.name][facet_idx] = 1.0
    
    return facet_shadows

def draw_facets_with_articulation(ax, satellite, facet_shadows, sun_vector_body, articulation_engine, epoch):
    """Draw satellite facets with proper articulation and shadowing."""
    facet_collection = []
    facet_colors = []
    articulation_info = {}
    
    for component in satellite.components:
        if not component.facets:
            continue
        
        # Get component transformation
        comp_rot_matrix = quaternion.as_rotation_matrix(component.relative_orientation)
        comp_position = component.relative_position
        
        # Apply articulation using the modular system
        if articulation_engine.is_component_articulated(component):
            try:
                articulation_rotation = articulation_engine.compute_articulation_rotation(
                    component, sun_vector_body, epoch
                )
                if articulation_rotation is not None:
                    comp_rot_matrix = comp_rot_matrix @ articulation_rotation
                    # Store articulation info for display
                    articulation_info[component.name] = "articulated"
            except Exception as e:
                logger.debug(f"Articulation failed for {component.name}: {e}")
        
        # Draw facets
        for facet_idx, facet in enumerate(component.facets):
            # Transform vertices to body frame
            vertices_local = np.array(facet.vertices)
            vertices_body = (comp_rot_matrix @ vertices_local.T).T + comp_position
            
            facet_collection.append(vertices_body)
            
            # Get shadow value
            shadow_value = facet_shadows.get(component.name, {}).get(facet_idx, 1.0)
            
            # Color based on shadow (yellow = lit, red = shadowed)
            if shadow_value > 0.5:
                color = (0.9, 0.9, 0.2, 0.8)  # Yellow for lit
            else:
                color = (0.8, 0.2, 0.2, 0.8)  # Red for shadowed
            facet_colors.append(color)
    
    # Add all facets to plot
    if facet_collection:
        poly3d = Poly3DCollection(facet_collection, facecolors=facet_colors,
                                edgecolors='black', linewidths=0.5, alpha=0.8)
        ax.add_collection3d(poly3d)
    
    return articulation_info

def create_shadow_vectors_animation(satellite, epochs, spice_handler, config, articulation_engine):
    """Create animation showing satellite with shadows and direction vectors."""
    logger.info(f"Creating shadow and vectors animation for {config.model_info.name}...")
    
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Pre-compute data for all frames
    logger.info("Pre-computing animation data...")
    frame_data = []
    
    for i, epoch in enumerate(epochs):
        if i % 10 == 0:
            logger.info(f"  Processing frame {i+1}/{NUM_FRAMES}")
        
        # Get positions in J2000
        sat_pos_j2000, _ = spice_handler.get_body_position(str(config.spice_config.satellite_id), epoch, "J2000", "EARTH")
        sun_pos_j2000, _ = spice_handler.get_body_position("SUN", epoch, "J2000", "EARTH")
        earth_pos_j2000 = np.array([0, 0, 0])  # Earth is at origin in Earth-centered J2000
        
        # DST ground station position
        dst_lat_rad = np.radians(-31.74)
        dst_lon_rad = np.radians(116.09)
        dst_x = EARTH_RADIUS_KM * np.cos(dst_lat_rad) * np.cos(dst_lon_rad)
        dst_y = EARTH_RADIUS_KM * np.cos(dst_lat_rad) * np.sin(dst_lon_rad)
        dst_z = EARTH_RADIUS_KM * np.sin(dst_lat_rad)
        dst_pos_j2000 = np.array([dst_x, dst_y, dst_z])
        
        # Get satellite attitude (J2000 to body frame transformation)
        j2000_to_body = spice_handler.get_target_orientation("J2000", config.spice_config.body_frame, epoch)
        
        # Calculate direction vectors in J2000 and transform to body frame
        vec_to_earth_j2000 = earth_pos_j2000 - sat_pos_j2000
        vec_to_dst_j2000 = dst_pos_j2000 - sat_pos_j2000
        vec_to_sun_j2000 = sun_pos_j2000 - sat_pos_j2000
        
        # Normalize and transform to body frame
        vec_to_earth_body = j2000_to_body @ (vec_to_earth_j2000 / np.linalg.norm(vec_to_earth_j2000))
        vec_to_dst_body = j2000_to_body @ (vec_to_dst_j2000 / np.linalg.norm(vec_to_dst_j2000))
        vec_to_sun_body = j2000_to_body @ (vec_to_sun_j2000 / np.linalg.norm(vec_to_sun_j2000))
        
        # Sun vector for shadow calculations (already normalized)
        sun_vector_body = vec_to_sun_body
        
        # Create shadow mesh and compute shadows
        shadow_mesh = create_shadow_mesh(satellite, articulation_engine, sun_vector_body, epoch)
        facet_shadows = compute_facet_shadows(satellite, sun_vector_body, shadow_mesh, articulation_engine, epoch)
        
        frame_data.append({
            'epoch': epoch,
            'facet_shadows': facet_shadows,
            'vec_to_earth_body': vec_to_earth_body,
            'vec_to_dst_body': vec_to_dst_body,
            'vec_to_sun_body': vec_to_sun_body,
            'sun_vector_body': sun_vector_body
        })
    
    logger.info("Pre-computation complete!")
    
    # Animation function
    def animate(frame_idx):
        ax.clear()
        
        # Set up axes
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_zlabel('Z (m)', fontsize=12)
        
        data = frame_data[frame_idx]
        utc_time = spice_handler.et_to_utc(data['epoch'], "C", 0)
        ax.set_title(f'{config.model_info.name} Attitude in J2000 Frame\n{utc_time}', 
                    fontsize=16, pad=20)
        
        # Draw satellite facets with shadowing and articulation
        articulation_info = draw_facets_with_articulation(ax, satellite, data['facet_shadows'], 
                                                         data['sun_vector_body'], articulation_engine, data['epoch'])
        
        # Draw direction vectors from origin
        vector_length = 15.0  # meters, scaled for satellite size
        origin = np.array([0, 0, 0])
        
        # Vector to Earth (blue)
        ax.quiver(origin[0], origin[1], origin[2],
                 data['vec_to_earth_body'][0], data['vec_to_earth_body'][1], data['vec_to_earth_body'][2],
                 length=vector_length, color='blue', arrow_length_ratio=0.2,
                 linewidth=3, label='To Earth')
        
        # Vector to DST (green)
        ax.quiver(origin[0], origin[1], origin[2],
                 data['vec_to_dst_body'][0], data['vec_to_dst_body'][1], data['vec_to_dst_body'][2],
                 length=vector_length, color='green', arrow_length_ratio=0.2,
                 linewidth=3, label='To DST')
        
        # Vector to Sun (orange)
        ax.quiver(origin[0], origin[1], origin[2],
                 data['vec_to_sun_body'][0], data['vec_to_sun_body'][1], data['vec_to_sun_body'][2],
                 length=vector_length, color='orange', arrow_length_ratio=0.2,
                 linewidth=3, label='To Sun')
        
        # Draw body frame axes for reference
        axis_length = 10.0  # scaled for satellite size
        ax.quiver(0, 0, 0, axis_length, 0, 0, color='red', alpha=0.5, 
                 arrow_length_ratio=0.1, linewidth=1)
        ax.quiver(0, 0, 0, 0, axis_length, 0, color='green', alpha=0.5,
                 arrow_length_ratio=0.1, linewidth=1)
        ax.quiver(0, 0, 0, 0, 0, axis_length, color='blue', alpha=0.5,
                 arrow_length_ratio=0.1, linewidth=1)
        ax.text(axis_length*1.1, 0, 0, 'X', color='red', alpha=0.5)
        ax.text(0, axis_length*1.1, 0, 'Y', color='green', alpha=0.5)
        ax.text(0, 0, axis_length*1.1, 'Z', color='blue', alpha=0.5)
        
        # Set axis limits to properly show satellite model
        axis_limit = 10.0  # meters - accommodate full satellite
        ax.set_xlim([-axis_limit, axis_limit])
        ax.set_ylim([-axis_limit, axis_limit])
        ax.set_zlim([-axis_limit, axis_limit])
        
        # Make axes equal
        ax.set_box_aspect([1,1,1])
        
        # Add info box
        info_str = f'Frame: {frame_idx+1}/{NUM_FRAMES}\n'
        
        # Show articulation info
        if articulation_info:
            info_str += f'Articulated: {", ".join(articulation_info.keys())}\n'
        else:
            info_str += f'Articulated: None\n'
        
        # Count lit/shadowed facets
        total_facets = sum(len(shadows) for shadows in data['facet_shadows'].values())
        lit_facets = sum(sum(1 for s in shadows.values() if s > 0.5) 
                        for shadows in data['facet_shadows'].values())
        shadow_percentage = (1 - lit_facets / total_facets) * 100 if total_facets > 0 else 0
        info_str += f'Shadow Coverage: {shadow_percentage:.1f}%'
        
        ax.text2D(0.02, 0.98, info_str, transform=ax.transAxes,
                 fontsize=12, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Legend
        ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1.0))
        
        # Grid
        ax.grid(True, alpha=0.3)
        
        # Set viewing angle
        ax.view_init(elev=20, azim=45)
        
        return []
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=NUM_FRAMES,
                                  interval=100, blit=False)
    
    return anim, fig

def main():
    """Main function to create shadow and vectors validation animation."""
    parser = argparse.ArgumentParser(description='Create shadow and vectors validation animation for any satellite model')
    parser.add_argument('--model-config', type=str, default=None,
                       help='Path to model configuration file (default: Intelsat 901)')
    parser.add_argument('--start-time', type=str, default="2020-02-05T10:00:00",
                       help='Start time in UTC (default: 2020-02-05T10:00:00)')
    parser.add_argument('--end-time', type=str, default="2020-02-05T16:00:00",
                       help='End time in UTC (default: 2020-02-05T16:00:00)')
    parser.add_argument('--frames', type=int, default=120,
                       help='Number of animation frames (default: 120)')
    parser.add_argument('--subdivision', type=int, default=3,
                       help='Facet subdivision level (default: 3)')
    parser.add_argument('--no-display', action='store_true',
                       help='Skip showing the plot window')
    
    args = parser.parse_args()
    
    # Update global variables
    global NUM_FRAMES
    NUM_FRAMES = args.frames
    
    # Load configuration
    config_manager = ModelConfigManager()
    config = config_manager.load_config(args.model_config)
    
    # Override time range if specified
    if args.start_time != "2020-02-05T10:00:00":
        config.simulation_defaults.start_time = args.start_time
    if args.end_time != "2020-02-05T16:00:00":
        config.simulation_defaults.end_time = args.end_time
    
    # Create articulation engine
    articulation_engine = ArticulationEngine(config)
    
    # Get paths from configuration
    model_path = config_manager.get_model_path(config)
    metakernel_path = config_manager.get_metakernel_path(config)
    
    logger.info(f"🛰️ Shadow & Vectors Animation for {config.model_info.name}")
    logger.info("=" * 60)
    
    # Create output directory
    output_dir = PROJECT_ROOT / "validation_animations"
    output_dir.mkdir(exist_ok=True)
    
    # Load satellite model
    logger.info("Loading satellite model...")
    satellite = load_satellite_from_yaml(str(model_path), 
                                       facet_subdivision_level=args.subdivision)
    logger.info(f"✅ Loaded: {satellite.name}")
    logger.info(f"   Components: {len(satellite.components)}")
    total_facets = sum(len(comp.facets) for comp in satellite.components if comp.facets)
    logger.info(f"   Total facets: {total_facets:,}")
    
    # Initialize SPICE
    logger.info("Initializing SPICE...")
    spice_handler = SpiceHandler()
    
    if not Path(metakernel_path).exists():
        logger.error(f"Metakernel not found at {metakernel_path}")
        return
    
    spice_handler.load_metakernel_programmatically(str(metakernel_path))
    logger.info("✅ SPICE initialized")
    
    # Generate time series
    start_time = config.simulation_defaults.start_time
    end_time = config.simulation_defaults.end_time
    logger.info(f"Time range: {start_time} to {end_time}")
    logger.info(f"Frames: {NUM_FRAMES}")
    
    start_et = spice_handler.utc_to_et(start_time)
    end_et = spice_handler.utc_to_et(end_time)
    epochs = np.linspace(start_et, end_et, NUM_FRAMES)
    
    # Create animation
    anim, fig = create_shadow_vectors_animation(satellite, epochs, spice_handler, config, articulation_engine)
    
    # Generate filenames based on model name
    safe_model_name = config.model_info.name.lower().replace(' ', '_')
    
    # Save as GIF
    gif_path = output_dir / f"shadow_vectors_{safe_model_name}.gif"
    logger.info(f"Saving GIF to: {gif_path}")
    anim.save(str(gif_path), writer='pillow', fps=5)  # Slower for detailed viewing
    logger.info("✅ GIF saved!")
    
    # Show plot unless disabled
    if not args.no_display:
        plt.show()
    
    # Cleanup
    spice_handler.unload_all_kernels()
    
    logger.info("✨ Shadow & vectors validation animation complete!")
    logger.info(f"📁 Results saved to: {output_dir}")

if __name__ == "__main__":
    main()
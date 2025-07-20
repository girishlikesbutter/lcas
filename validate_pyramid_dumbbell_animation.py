#!/usr/bin/env python3
"""
Pyramid Dumbbell Shadow Vectors Animation - Demonstrates modular system with custom model.
Adapted from validate_shadow_vectors_animation.py to work with the pyramid-dumbbell model.
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Dict, Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.model_config import ModelConfigManager
from src.materials.brdf_manager import BRDFManager
from src.articulation.articulation_engine import ArticulationEngine
from src.io.model_io import load_satellite_from_yaml
from src.spice.spice_handler import SpiceHandler
from src.illumination.pytorch_shadow_engine import pytorch_get_lit_fractions_for_kinematics, get_pytorch_shadow_engine
from src.simulation.lightcurve_engine import LightCurveEngine

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_pyramid_dumbbell_animation(
    num_frames: int = 120,
    start_time: str = "2020-02-05T10:00:00",
    end_time: str = "2020-02-05T16:00:00",
    output_dir: str = "pyramid_dumbbell_animations",
    use_shadows: bool = True,
    articulation_offset: float = 0.0,
    save_mp4: bool = True,
    save_gif: bool = True
):
    """
    Create shadow vectors animation for pyramid-dumbbell model.
    
    Args:
        num_frames: Number of animation frames
        start_time: Start time in UTC
        end_time: End time in UTC
        output_dir: Output directory for animations
        use_shadows: Whether to compute shadows
        articulation_offset: Articulation offset in degrees
        save_mp4: Save MP4 animation
        save_gif: Save GIF animation
    """
    
    logger.info("🔺 Pyramid Dumbbell Shadow Vectors Animation")
    logger.info("=" * 60)
    
    # Load configuration
    config_manager = ModelConfigManager()
    config = config_manager.load_config("pyramid_dumbbell_config.yaml")
    
    # Create managers
    brdf_manager = BRDFManager(config)
    articulation_engine = ArticulationEngine(config)
    
    # Get paths
    model_path = config_manager.get_model_path(config)
    metakernel_path = config_manager.get_metakernel_path(config)
    
    # Load satellite model
    logger.info("Loading pyramid-dumbbell model...")
    satellite = load_satellite_from_yaml(str(model_path), facet_subdivision_level=3)
    
    # Update BRDF parameters
    brdf_manager.update_satellite_brdf_parameters(satellite)
    
    logger.info(f"✅ Model loaded: {satellite.name}")
    logger.info(f"   Components: {len(satellite.components)}")
    
    total_facets = sum(len(comp.facets) for comp in satellite.components if comp.facets)
    logger.info(f"   Total facets: {total_facets:,}")
    
    # Initialize SPICE
    logger.info("Initializing SPICE...")
    spice_handler = SpiceHandler()
    spice_handler.load_metakernel_programmatically(str(metakernel_path))
    logger.info("✅ SPICE initialized")
    
    # Generate time series
    logger.info(f"Time range: {start_time} to {end_time}")
    logger.info(f"Frames: {num_frames}")
    
    start_et = spice_handler.utc_to_et(start_time)
    end_et = spice_handler.utc_to_et(end_time)
    epochs = np.linspace(start_et, end_et, num_frames)
    
    # Get satellite positions and orientations
    logger.info("Computing satellite positions...")
    satellite_positions = []
    sun_positions = []
    sat_att_matrices = []
    
    satellite_id = config.spice_config.satellite_id
    body_frame = config.spice_config.body_frame
    
    for i, epoch in enumerate(epochs):
        if (i + 1) % 20 == 0:
            logger.info(f"  Processing frame {i + 1}/{num_frames}")
        
        # Get satellite position
        sat_pos, _ = spice_handler.get_body_position(str(satellite_id), epoch, "J2000", "EARTH")
        satellite_positions.append(sat_pos)
        
        # Get sun position
        sun_pos, _ = spice_handler.get_body_position("SUN", epoch, "J2000", "EARTH")
        sun_positions.append(sun_pos)
        
        # Get satellite attitude
        att_matrix = spice_handler.get_target_orientation("J2000", body_frame, epoch)
        sat_att_matrices.append(att_matrix)
    
    logger.info(f"✅ Computed {len(satellite_positions)} positions")
    
    # Compute orbit statistics
    satellite_positions = np.array(satellite_positions)
    distances = np.linalg.norm(satellite_positions, axis=1)
    altitudes = distances - 6371.0  # Earth radius in km
    
    logger.info("Orbit statistics:")
    logger.info(f"  Min altitude: {np.min(altitudes):.1f} km")
    logger.info(f"  Max altitude: {np.max(altitudes):.1f} km")
    logger.info(f"  Mean altitude: {np.mean(altitudes):.1f} km")
    
    # Compute shadow data if requested
    lit_fractions_dict = {}
    if use_shadows:
        logger.info("Computing shadow data...")
        
        # Get all conceptual face names
        target_faces = []
        for component in satellite.components:
            if component.conceptual_faces_map:
                target_faces.extend(component.conceptual_faces_map.keys())
        
        logger.info(f"Target faces: {len(target_faces)}")
        
        # Compute shadows using PyTorch
        lit_fractions_dict, _ = pytorch_get_lit_fractions_for_kinematics(
            satellite=satellite,
            sun_pos_j2000_epochs=np.array(sun_positions),
            sat_pos_j2000_epochs=np.array(satellite_positions),
            sat_att_C_j2000_to_body_epochs=np.array(sat_att_matrices),
            target_conceptual_faces=target_faces,
            articulation_offset=articulation_offset
        )
        
        logger.info("✅ Shadow computation complete")
    else:
        logger.info("Skipping shadow computation")
    
    # Create animation
    logger.info("Creating animation...")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Setup figure
    fig = plt.figure(figsize=(16, 12))
    
    # Create subplots
    ax_3d = fig.add_subplot(2, 2, 1, projection='3d')
    ax_orbit = fig.add_subplot(2, 2, 2)
    ax_vectors = fig.add_subplot(2, 2, 3)
    ax_shadows = fig.add_subplot(2, 2, 4)
    
    # Initialize plots
    def init_plots():
        # 3D orbit plot
        ax_3d.clear()
        ax_3d.set_xlabel('X (km)')
        ax_3d.set_ylabel('Y (km)')
        ax_3d.set_zlabel('Z (km)')
        ax_3d.set_title('Pyramid Dumbbell Orbit')
        
        # Draw Earth
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        earth_x = 6371 * np.outer(np.cos(u), np.sin(v))
        earth_y = 6371 * np.outer(np.sin(u), np.sin(v))
        earth_z = 6371 * np.outer(np.ones(np.size(u)), np.cos(v))
        ax_3d.plot_surface(earth_x, earth_y, earth_z, alpha=0.3, color='blue')
        
        # Plot full orbit
        ax_3d.plot(satellite_positions[:, 0], satellite_positions[:, 1], satellite_positions[:, 2], 
                   'gray', alpha=0.5, linewidth=1)
        
        # 2D orbit projection
        ax_orbit.clear()
        ax_orbit.set_xlabel('X (km)')
        ax_orbit.set_ylabel('Y (km)')
        ax_orbit.set_title('Orbit Projection (XY)')
        ax_orbit.plot(satellite_positions[:, 0], satellite_positions[:, 1], 'gray', alpha=0.5)
        ax_orbit.set_aspect('equal')
        
        # Vectors plot
        ax_vectors.clear()
        ax_vectors.set_xlabel('Time (hours)')
        ax_vectors.set_ylabel('Component')
        ax_vectors.set_title('Sun Direction Vectors')
        
        # Shadow plot
        ax_shadows.clear()
        ax_shadows.set_xlabel('Time (hours)')
        ax_shadows.set_ylabel('Lit Fraction')
        ax_shadows.set_title('Shadow Analysis')
        ax_shadows.set_ylim(0, 1)
    
    # Animation function
    def animate(frame):
        # Current time
        current_time = (frame / num_frames) * (end_et - start_et) / 3600.0  # hours
        
        # Update 3D plot
        ax_3d.clear()
        ax_3d.set_xlabel('X (km)')
        ax_3d.set_ylabel('Y (km)')
        ax_3d.set_zlabel('Z (km)')
        ax_3d.set_title(f'Pyramid Dumbbell at t={current_time:.1f}h')
        
        # Draw Earth
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        earth_x = 6371 * np.outer(np.cos(u), np.sin(v))
        earth_y = 6371 * np.outer(np.sin(u), np.sin(v))
        earth_z = 6371 * np.outer(np.ones(np.size(u)), np.cos(v))
        ax_3d.plot_surface(earth_x, earth_y, earth_z, alpha=0.3, color='blue')
        
        # Plot orbit trail
        trail_start = max(0, frame - 20)
        ax_3d.plot(satellite_positions[trail_start:frame+1, 0], 
                   satellite_positions[trail_start:frame+1, 1], 
                   satellite_positions[trail_start:frame+1, 2], 
                   'yellow', alpha=0.7, linewidth=2)
        
        # Plot current position
        current_pos = satellite_positions[frame]
        ax_3d.scatter(current_pos[0], current_pos[1], current_pos[2], 
                     color='red', s=100, label='Pyramid Dumbbell')
        
        # Plot sun direction
        sun_dir = sun_positions[frame] - current_pos
        sun_dir_norm = sun_dir / np.linalg.norm(sun_dir) * 5000  # Scale for visibility
        ax_3d.quiver(current_pos[0], current_pos[1], current_pos[2],
                    sun_dir_norm[0], sun_dir_norm[1], sun_dir_norm[2],
                    color='yellow', arrow_length_ratio=0.1, linewidth=2, label='Sun')
        
        ax_3d.legend()
        ax_3d.set_box_aspect([1,1,1])
        
        # Update 2D orbit
        ax_orbit.clear()
        ax_orbit.set_xlabel('X (km)')
        ax_orbit.set_ylabel('Y (km)')
        ax_orbit.set_title('Orbit Projection (XY)')
        ax_orbit.plot(satellite_positions[:, 0], satellite_positions[:, 1], 'gray', alpha=0.3)
        ax_orbit.plot(satellite_positions[:frame+1, 0], satellite_positions[:frame+1, 1], 'yellow', linewidth=2)
        ax_orbit.scatter(current_pos[0], current_pos[1], color='red', s=50, zorder=5)
        ax_orbit.set_aspect('equal')
        
        # Update vectors plot
        ax_vectors.clear()
        ax_vectors.set_xlabel('Time (hours)')
        ax_vectors.set_ylabel('Sun Vector Component')
        ax_vectors.set_title('Sun Direction in Body Frame')
        
        # Calculate sun vectors in body frame
        times = np.linspace(0, (end_et - start_et) / 3600.0, frame + 1)
        sun_vectors_body = []
        for i in range(frame + 1):
            sun_vec_j2000 = sun_positions[i] - satellite_positions[i]
            sun_vec_body = sat_att_matrices[i] @ sun_vec_j2000
            sun_vec_body_norm = sun_vec_body / np.linalg.norm(sun_vec_body)
            sun_vectors_body.append(sun_vec_body_norm)
        
        sun_vectors_body = np.array(sun_vectors_body)
        ax_vectors.plot(times, sun_vectors_body[:, 0], 'r-', label='X', linewidth=2)
        ax_vectors.plot(times, sun_vectors_body[:, 1], 'g-', label='Y', linewidth=2)
        ax_vectors.plot(times, sun_vectors_body[:, 2], 'b-', label='Z', linewidth=2)
        ax_vectors.legend()
        ax_vectors.grid(True, alpha=0.3)
        ax_vectors.set_ylim(-1, 1)
        
        # Update shadow plot
        ax_shadows.clear()
        ax_shadows.set_xlabel('Time (hours)')
        ax_shadows.set_ylabel('Lit Fraction')
        ax_shadows.set_title('Pyramid Shadow Analysis')
        
        if use_shadows and lit_fractions_dict:
            for face_name, lit_fractions in lit_fractions_dict.items():
                if len(lit_fractions) > frame:
                    ax_shadows.plot(times, lit_fractions[:frame+1], 
                                  label=face_name, linewidth=2, alpha=0.7)
        else:
            ax_shadows.text(0.5, 0.5, 'No shadow computation', 
                          transform=ax_shadows.transAxes, ha='center', va='center')
        
        ax_shadows.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax_shadows.grid(True, alpha=0.3)
        ax_shadows.set_ylim(0, 1)
        
        plt.tight_layout()
    
    # Create animation
    init_plots()
    anim = FuncAnimation(fig, animate, frames=num_frames, interval=100, blit=False)
    
    # Save animations
    if save_mp4:
        mp4_path = output_path / "pyramid_dumbbell_animation.mp4"
        logger.info(f"Saving MP4 to: {mp4_path}")
        anim.save(str(mp4_path), writer='ffmpeg', fps=10, bitrate=2000, 
                 metadata={'artist': 'lcforge'})
        logger.info("✅ MP4 saved!")
    
    if save_gif:
        gif_path = output_path / "pyramid_dumbbell_animation.gif"
        logger.info(f"Saving GIF to: {gif_path}")
        anim.save(str(gif_path), writer='pillow', fps=10)
        logger.info("✅ GIF saved!")
    
    logger.info("Animation creation complete!")
    return str(output_path)

def main():
    """Main function with command line arguments."""
    parser = argparse.ArgumentParser(description='Create pyramid-dumbbell shadow vectors animation')
    parser.add_argument('--frames', type=int, default=120, help='Number of animation frames')
    parser.add_argument('--start-time', default="2020-02-05T10:00:00", help='Start time (UTC)')
    parser.add_argument('--end-time', default="2020-02-05T16:00:00", help='End time (UTC)')
    parser.add_argument('--output-dir', default="pyramid_dumbbell_animations", help='Output directory')
    parser.add_argument('--no-shadows', action='store_true', help='Disable shadow computation')
    parser.add_argument('--articulation-offset', type=float, default=0.0, help='Articulation offset (degrees)')
    parser.add_argument('--no-mp4', action='store_true', help='Skip MP4 output')
    parser.add_argument('--no-gif', action='store_true', help='Skip GIF output')
    
    args = parser.parse_args()
    
    # Create animation
    output_path = create_pyramid_dumbbell_animation(
        num_frames=args.frames,
        start_time=args.start_time,
        end_time=args.end_time,
        output_dir=args.output_dir,
        use_shadows=not args.no_shadows,
        articulation_offset=args.articulation_offset,
        save_mp4=not args.no_mp4,
        save_gif=not args.no_gif
    )
    
    logger.info(f"🎉 Animation saved to: {output_path}")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Orbit Validation Animation
==========================

Creates an animation showing Intelsat 901's orbit in the J2000 reference frame
with Earth visible, to validate the satellite's trajectory during the pass.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import logging

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from src.spice.spice_handler import SpiceHandler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
METAKERNEL_PATH = PROJECT_ROOT / "data" / "spice_kernels" / "missions" / "dst-is901" / "INTELSAT_901-metakernel.tm"
OUTPUT_DIR = PROJECT_ROOT / "validation_animations"

# Time range
START_TIME_UTC = "2020-02-05T10:00:00"
END_TIME_UTC = "2020-02-05T16:00:00"
SATELLITE_ID = -126824
NUM_FRAMES = 120  # 2 frames per minute

# Earth parameters
EARTH_RADIUS_KM = 6371.0

def create_earth_sphere(radius=EARTH_RADIUS_KM, resolution=20):
    """Create mesh data for Earth sphere."""
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v))
    return x, y, z

def plot_reference_frame(ax, scale=10000):
    """Plot J2000 reference frame axes."""
    # X-axis (red)
    ax.quiver(0, 0, 0, scale, 0, 0, color='red', arrow_length_ratio=0.1, linewidth=2, alpha=0.7)
    ax.text(scale*1.1, 0, 0, 'X', color='red', fontsize=12, weight='bold')
    
    # Y-axis (green)
    ax.quiver(0, 0, 0, 0, scale, 0, color='green', arrow_length_ratio=0.1, linewidth=2, alpha=0.7)
    ax.text(0, scale*1.1, 0, 'Y', color='green', fontsize=12, weight='bold')
    
    # Z-axis (blue)
    ax.quiver(0, 0, 0, 0, 0, scale, color='blue', arrow_length_ratio=0.1, linewidth=2, alpha=0.7)
    ax.text(0, 0, scale*1.1, 'Z', color='blue', fontsize=12, weight='bold')

def create_orbit_animation(positions, epochs, spice_handler):
    """Create the orbit visualization animation."""
    logger.info("Creating orbit animation...")
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Set up the plot
    ax.set_title('Intelsat 901 Orbit Validation (J2000 Frame)', fontsize=16, pad=20)
    ax.set_xlabel('X (km)', fontsize=12)
    ax.set_ylabel('Y (km)', fontsize=12)
    ax.set_zlabel('Z (km)', fontsize=12)
    
    # Create Earth
    earth_x, earth_y, earth_z = create_earth_sphere()
    
    # Set axis limits based on orbit
    max_coord = np.max(np.abs(positions)) * 1.2
    ax.set_xlim([-max_coord, max_coord])
    ax.set_ylim([-max_coord, max_coord])
    ax.set_zlim([-max_coord, max_coord])
    
    # Make axes equal
    ax.set_box_aspect([1,1,1])
    
    # Initialize plot elements
    earth_surface = None
    orbit_trail = None
    satellite_point = None
    time_text = None
    info_text = None
    
    def init():
        return []
    
    def animate(frame):
        nonlocal earth_surface, orbit_trail, satellite_point, time_text, info_text
        
        # Clear previous elements
        ax.clear()
        
        # Reset axis properties
        ax.set_xlabel('X (km)', fontsize=12)
        ax.set_ylabel('Y (km)', fontsize=12)
        ax.set_zlabel('Z (km)', fontsize=12)
        ax.set_xlim([-max_coord, max_coord])
        ax.set_ylim([-max_coord, max_coord])
        ax.set_zlim([-max_coord, max_coord])
        ax.set_box_aspect([1,1,1])
        
        # Plot Earth
        earth_surface = ax.plot_surface(earth_x, earth_y, earth_z, color='lightblue', 
                                       alpha=0.7, linewidth=0, antialiased=True)
        
        # Plot Earth wireframe for better visibility
        ax.plot_wireframe(earth_x, earth_y, earth_z, color='darkblue', 
                         alpha=0.2, linewidth=0.5)
        
        # Plot reference frame
        plot_reference_frame(ax)
        
        # Plot orbit trail up to current position
        if frame > 0:
            orbit_trail = ax.plot(positions[:frame+1, 0], 
                                 positions[:frame+1, 1], 
                                 positions[:frame+1, 2], 
                                 'cyan', linewidth=2, alpha=0.8, 
                                 label='Orbit trail')
        
        # Plot full orbit path (faded)
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                'gray', linewidth=1, alpha=0.3, linestyle='--',
                label='Full orbit')
        
        # Plot current satellite position
        satellite_point = ax.scatter(positions[frame, 0], 
                                   positions[frame, 1], 
                                   positions[frame, 2], 
                                   c='red', s=100, marker='o',
                                   edgecolors='darkred', linewidth=2,
                                   label='Intelsat 901')
        
        # Add velocity vector
        if frame > 0:
            vel_vector = (positions[min(frame+1, len(positions)-1)] - positions[frame]) * 10
            ax.quiver(positions[frame, 0], positions[frame, 1], positions[frame, 2],
                     vel_vector[0], vel_vector[1], vel_vector[2],
                     color='yellow', arrow_length_ratio=0.3, linewidth=2,
                     label='Velocity')
        
        # Time and info display
        utc_time = spice_handler.et_to_utc(epochs[frame], "C", 0)
        ax.set_title(f'Intelsat 901 Orbit Validation (J2000 Frame)\n{utc_time}', 
                    fontsize=16, pad=20)
        
        # Calculate orbital parameters
        r = np.linalg.norm(positions[frame])
        alt = r - EARTH_RADIUS_KM
        
        # Info box
        info_str = f'Radius: {r:.1f} km\nAltitude: {alt:.1f} km\nFrame: {frame+1}/{NUM_FRAMES}'
        ax.text2D(0.02, 0.98, info_str, transform=ax.transAxes, 
                 fontsize=12, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Set viewing angle
        ax.view_init(elev=20, azim=45 + frame * 0.5)  # Slowly rotate view
        
        # Legend
        ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1.0))
        
        # Grid
        ax.grid(True, alpha=0.3)
        
        return [earth_surface, orbit_trail, satellite_point]
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                  frames=NUM_FRAMES, interval=100, blit=False)
    
    return anim, fig

def main():
    """Main function to create orbit validation animation."""
    logger.info("🛰️ Orbit Validation Animation for Intelsat 901")
    logger.info("=" * 60)
    
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Initialize SPICE
    logger.info("Initializing SPICE...")
    spice_handler = SpiceHandler()
    
    if not METAKERNEL_PATH.exists():
        logger.error(f"Metakernel not found at {METAKERNEL_PATH}")
        return
    
    spice_handler.load_metakernel_programmatically(str(METAKERNEL_PATH))
    logger.info("✅ SPICE initialized")
    
    # Generate time series
    logger.info(f"Time range: {START_TIME_UTC} to {END_TIME_UTC}")
    logger.info(f"Frames: {NUM_FRAMES}")
    
    start_et = spice_handler.utc_to_et(START_TIME_UTC)
    end_et = spice_handler.utc_to_et(END_TIME_UTC)
    epochs = np.linspace(start_et, end_et, NUM_FRAMES)
    
    # Get satellite positions
    logger.info("Computing satellite positions...")
    positions = []
    
    for i, epoch in enumerate(epochs):
        if i % 20 == 0:
            logger.info(f"  Processing frame {i+1}/{NUM_FRAMES}")
        
        # Get position in J2000 frame relative to Earth
        pos, _ = spice_handler.get_body_position(str(SATELLITE_ID), epoch, "J2000", "EARTH")
        positions.append(pos)
    
    positions = np.array(positions)
    logger.info(f"✅ Computed {len(positions)} positions")
    
    # Analyze orbit
    distances = np.linalg.norm(positions, axis=1)
    altitudes = distances - EARTH_RADIUS_KM
    logger.info(f"Orbit statistics:")
    logger.info(f"  Min altitude: {np.min(altitudes):.1f} km")
    logger.info(f"  Max altitude: {np.max(altitudes):.1f} km")
    logger.info(f"  Mean altitude: {np.mean(altitudes):.1f} km")
    
    # Create animation
    logger.info("Creating animation...")
    anim, fig = create_orbit_animation(positions, epochs, spice_handler)
    
    # Save animation
    mp4_path = OUTPUT_DIR / "orbit_validation.mp4"
    logger.info(f"Saving MP4 to: {mp4_path}")
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=10, metadata=dict(artist='lcforge'), bitrate=2000)
    anim.save(str(mp4_path), writer=writer)
    logger.info("✅ MP4 saved!")
    
    # Save as GIF
    gif_path = OUTPUT_DIR / "orbit_validation.gif"
    logger.info(f"Saving GIF to: {gif_path}")
    anim.save(str(gif_path), writer='pillow', fps=10)
    logger.info("✅ GIF saved!")
    
    # Show plot
    plt.show()
    
    # Cleanup
    spice_handler.unload_all_kernels()
    
    logger.info("✨ Orbit validation animation complete!")
    logger.info(f"📁 Results saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
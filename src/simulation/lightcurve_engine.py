# lcforge/src/simulation/lightcurve_engine.py

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import os
import shutil

# Animation imports
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import imageio

from src.models.model_definitions import Satellite, Component, Facet, BRDFMaterialProperties
from src.spice.spice_handler import SpiceHandler
# Only PyTorch GPU shadow engine is supported in minimal project
from src.illumination.pytorch_shadow_engine import pytorch_get_lit_fractions_for_kinematics


@dataclass
class LightCurveResult:
    """
    Container for light curve generation results.
    
    Attributes:
        epochs: Array of epoch times (ET)
        magnitudes: Array of apparent magnitudes
        total_flux: Array of total flux values
        observer_distances: Array of observer distances (km)
    """
    epochs: np.ndarray
    magnitudes: np.ndarray
    total_flux: np.ndarray
    observer_distances: np.ndarray


class LightCurveEngine:
    """
    Light curve generation engine implementing the Ashikhmin-Shirley BRDF model.
    Integrates with shadow interpolation system to account for self-shadowing.
    """
    
    def __init__(self, spice_handler: SpiceHandler):
        """
        Initialize the light curve engine.
        
        Args:
            spice_handler: Configured SpiceHandler instance for ephemeris data
        """
        self.spice_handler = spice_handler
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Constants
        self.SUN_APPARENT_MAGNITUDE = -26.74  # m0 from reference
        self.DST_OBSERVATORY_ID = 399999  # SPICE ID for DST observatory
        
    def calculate_brdf(self, 
                      n_dot_k1: float, 
                      n_dot_k2: float, 
                      h_dot_k1: float, 
                      n_dot_h: float,
                      material: BRDFMaterialProperties) -> float:
        """
        Calculate the Ashikhmin-Shirley BRDF value.
        
        Args:
            n_dot_k1: Dot product of surface normal and light direction
            n_dot_k2: Dot product of surface normal and observer direction  
            h_dot_k1: Dot product of halfway vector and light direction
            n_dot_h: Dot product of surface normal and halfway vector
            material: BRDF material properties
            
        Returns:
            Total BRDF value (rho)
        """
        # Calculate intermediate terms for diffuse component
        alpha = 1.0 - n_dot_k1 / 2.0
        beta = 1.0 - n_dot_k2 / 2.0
        
        # Calculate Fresnel term for specular component
        fresnel = material.r_s + (1.0 - material.r_s) * (1.0 - h_dot_k1)**5
        
        # Diffuse component
        rho_diff = (28.0 * material.r_d / (23.0 * np.pi)) * (1.0 - material.r_s) * \
                   (1.0 - alpha**5) * (1.0 - beta**5)
        
        # Specular component
        denominator = h_dot_k1 * max(n_dot_k1, n_dot_k2)
        if denominator > 1e-10:  # Avoid division by zero
            rho_spec = ((material.n_phong + 1.0) / (8.0 * np.pi)) * \
                       (n_dot_h**material.n_phong / denominator) * fresnel
        else:
            rho_spec = 0.0
            
        return rho_diff + rho_spec
    
    def calculate_facet_flux(self,
                           facet: Facet,
                           sun_direction: np.ndarray,
                           observer_direction: np.ndarray,
                           observer_distance: float,
                           lit_fraction: float = 1.0,
                           debug_info: dict = None) -> float:
        """
        Calculate the reflected flux from a single facet.
        
        Args:
            facet: Facet object with geometry and material properties
            sun_direction: Unit vector from facet to sun (k1)
            observer_direction: Unit vector from facet to observer (k2)  
            observer_distance: Distance from satellite to observer (d)
            lit_fraction: Fraction of facet that is illuminated (0.0 to 1.0)
            debug_info: Optional dict to collect debug information
            
        Returns:
            Reflected flux from the facet
        """
        # Calculate dot products
        n_dot_k1 = np.dot(facet.normal, sun_direction)
        n_dot_k2 = np.dot(facet.normal, observer_direction)
        
        # Visibility check: facet must be illuminated and visible
        if n_dot_k1 <= 0 or n_dot_k2 <= 0:
            # Debug logging for back-face culling
            if debug_info is not None:
                debug_info['back_face_culled'] = True
                debug_info['n_dot_k1'] = n_dot_k1
                debug_info['n_dot_k2'] = n_dot_k2
                debug_info['reason'] = f"Back-face culled: n·k1={n_dot_k1:.4f}, n·k2={n_dot_k2:.4f}"
            return 0.0
            
        # Calculate halfway vector
        halfway = sun_direction + observer_direction
        halfway_norm = np.linalg.norm(halfway)
        if halfway_norm > 1e-10:
            halfway = halfway / halfway_norm
        else:
            return 0.0
            
        h_dot_k1 = np.dot(halfway, sun_direction)
        n_dot_h = np.dot(facet.normal, halfway)
        
        # Calculate BRDF
        rho = self.calculate_brdf(n_dot_k1, n_dot_k2, h_dot_k1, n_dot_h, 
                                facet.material_properties)
        
        # Calculate effective area accounting for shadowing
        effective_area = facet.area * lit_fraction
        
        # Calculate flux: ρb = ρ * A_eff * (n·k1) * (n·k2) / d²
        flux = rho * effective_area * n_dot_k1 * n_dot_k2 / (observer_distance**2)
        
        # Debug logging for successful BRDF calculation
        if debug_info is not None:
            debug_info['back_face_culled'] = False
            debug_info['n_dot_k1'] = n_dot_k1
            debug_info['n_dot_k2'] = n_dot_k2
            debug_info['lit_fraction'] = lit_fraction
            debug_info['effective_area'] = effective_area
            debug_info['area'] = facet.area
            debug_info['rho'] = rho
            debug_info['flux'] = flux
            debug_info['reason'] = f"BRDF contributes: flux={flux:.6e}"
        
        return flux
    
    def get_sun_direction_in_body_frame(self, epoch: float, satellite_id: int, 
                                      body_frame: str) -> np.ndarray:
        """
        Get the sun direction vector (k1) in the satellite body frame.
        k1 = unit vector from satellite to sun
        
        Args:
            epoch: Ephemeris time
            satellite_id: SPICE ID of the satellite
            body_frame: Name of the satellite body frame
            
        Returns:
            Unit vector from satellite to sun in body frame coordinates (k1)
        """
        # Step 1: Get positions in J2000 frame (match observer calculation reference frame)
        sun_pos_j2000, _ = self.spice_handler.get_body_position("SUN", epoch, "J2000", "EARTH")  # Sun relative to Earth
        sat_pos_j2000, _ = self.spice_handler.get_body_position(str(satellite_id), epoch, "J2000", "EARTH")  # Satellite relative to Earth
        
        # Step 2: Compute k1 vector (satellite to sun) in J2000
        k1_j2000 = sun_pos_j2000 - sat_pos_j2000
        k1_j2000_normalized = k1_j2000 / np.linalg.norm(k1_j2000)
        
        # Step 3: Transform k1 vector to satellite body frame
        # Get transformation matrix from J2000 TO body frame
        transform_matrix = self.spice_handler.get_target_orientation("J2000", body_frame, epoch)
        k1_body = transform_matrix @ k1_j2000_normalized
        
        # DEBUG: Try both directions to see which one is correct
        transform_matrix_inverse = self.spice_handler.get_target_orientation(body_frame, "J2000", epoch)  
        k1_body_alt = transform_matrix_inverse.T @ k1_j2000_normalized  # Transpose = inverse for rotation matrix
        
        # Remove debug for now - let's test if the alternative transform works
        # Try the alternative transformation direction
        k1_body = k1_body_alt  # Use the alternative transform
        
        return k1_body
    
    def get_observer_direction_and_distance(self, epoch: float, satellite_id: int,
                                          body_frame: str) -> Tuple[np.ndarray, float]:
        """
        Get the observer direction vector (k2) and distance in the satellite body frame.
        k2 = unit vector from satellite to observer
        
        Args:
            epoch: Ephemeris time  
            satellite_id: SPICE ID of the satellite
            body_frame: Name of the satellite body frame
            
        Returns:
            Tuple of (unit vector from satellite to observer in body frame (k2), distance in km)
        """
        # Step 1: Get positions in J2000 frame
        obs_pos_j2000, _ = self.spice_handler.get_body_position(str(self.DST_OBSERVATORY_ID), epoch, "J2000", "EARTH")
        sat_pos_j2000, _ = self.spice_handler.get_body_position(str(satellite_id), epoch, "J2000", "EARTH")
        
        # Step 2: Compute k2 vector (satellite to observer) in J2000
        k2_j2000 = obs_pos_j2000 - sat_pos_j2000
        distance = np.linalg.norm(k2_j2000)
        k2_j2000_normalized = k2_j2000 / distance
        
        # Step 3: Transform k2 vector to satellite body frame
        # Get transformation matrix from J2000 TO body frame
        transform_matrix = self.spice_handler.get_target_orientation("J2000", body_frame, epoch)
        k2_body = transform_matrix @ k2_j2000_normalized
        
        return k2_body, distance
    
    def calculate_phase_angle(self, sun_direction: np.ndarray, observer_direction: np.ndarray) -> float:
        """
        Calculate the phase angle between Sun-Satellite-Observer.
        
        Args:
            sun_direction: Unit vector from satellite to sun (k1)
            observer_direction: Unit vector from satellite to observer (k2)
            
        Returns:
            Phase angle in degrees (0° when sun and observer on same side, 180° when opposite)
        """
        # Phase angle is the angle between the two vectors pointing away from satellite
        cos_phase = np.dot(sun_direction, observer_direction)
        cos_phase = np.clip(cos_phase, -1.0, 1.0)
        phase_angle = np.degrees(np.arccos(cos_phase))
        return phase_angle
    
    def _generate_v3_style_animation_frame(self, satellite: Satellite, frame_index: int, 
                                         epoch: float, satellite_id: int,
                                         sun_direction: np.ndarray, observer_direction: np.ndarray,
                                         distance: float, phase_angle: float,
                                         output_dir: str, camera_params: Optional[Dict] = None,
                                         component_transforms: Optional[Dict[str, np.ndarray]] = None,
                                         shadow_data: Optional[Dict] = None):
        """
        Generate a single frame in V3 style showing satellite with illumination/shadowing colors.
        """
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        # V3 style colors
        lit_color_rgba = [0.0, 0.8, 0.0, 1.0]      # Green for lit triangles
        shadowed_color_rgba = [0.8, 0.0, 0.0, 1.0]  # Red for shadowed triangles
        error_color_rgba = [0.5, 0.5, 0.5, 1.0]    # Gray for missing data
        
        all_transformed_vertices = []
        
        for component in satellite.components:
            if not component.facets:
                continue
            
            # Get component transformation matrix
            comp_rel_pos = component.relative_position
            comp_rel_orient = component.relative_orientation
            if hasattr(comp_rel_orient, 'as_rotation_matrix'):
                comp_rot_matrix = comp_rel_orient.as_rotation_matrix()
            else:
                # Handle numpy.quaternion
                import quaternion
                comp_rot_matrix = quaternion.as_rotation_matrix(comp_rel_orient)
            
            # Build final transformation matrix
            final_transform_matrix = np.eye(4)
            final_transform_matrix[:3, :3] = comp_rot_matrix
            final_transform_matrix[:3, 3] = comp_rel_pos
            
            # Apply additional component transform if provided (for articulating panels)
            if component_transforms and component.name in component_transforms:
                additional_transform = component_transforms[component.name]
                final_transform_matrix = additional_transform @ final_transform_matrix
            
            component_triangles = []
            component_colors = []
            
            # Use ray-traced shadow data EXACTLY like V3 does
            if not component.facets or not component.conceptual_faces_map:
                continue

            comp_shadow_data = None
            if shadow_data and component.name in shadow_data:
                comp_shadow_data = shadow_data[component.name]

            for cf_name, facet_indices in component.conceptual_faces_map.items():
                triangle_statuses_for_cf = None
                if comp_shadow_data:
                    triangle_statuses_for_cf = comp_shadow_data.get(cf_name)

                if triangle_statuses_for_cf is None:
                    # No shadow data, color all facets gray
                    for facet_idx_loop in facet_indices:
                        if facet_idx_loop < len(component.facets):
                            facet = component.facets[facet_idx_loop]
                            local_vertices_np = np.array(facet.vertices)
                            vertices_homogeneous = np.hstack([local_vertices_np, np.ones((local_vertices_np.shape[0], 1))])
                            transformed_vertices_homogeneous = (final_transform_matrix @ vertices_homogeneous.T).T
                            transformed_vertices = transformed_vertices_homogeneous[:, :3]
                            component_triangles.append(transformed_vertices)
                            component_colors.append(error_color_rgba)
                            all_transformed_vertices.extend(transformed_vertices)
                    continue

                if len(triangle_statuses_for_cf) != len(facet_indices):
                    self.logger.error(f"Mismatch for CF '{cf_name}' in '{component.name}'. "
                                     f"Expected {len(facet_indices)} statuses, got {len(triangle_statuses_for_cf)}. Skipping this CF.")
                    continue

                for i, facet_idx in enumerate(facet_indices):
                    if facet_idx >= len(component.facets):
                        self.logger.error(f"Invalid facet index {facet_idx} for CF '{cf_name}'. Skipping.")
                        continue

                    facet = component.facets[facet_idx]
                    local_vertices_np = np.array(facet.vertices)
                    vertices_homogeneous = np.hstack([local_vertices_np, np.ones((local_vertices_np.shape[0], 1))])
                    transformed_vertices_homogeneous = (final_transform_matrix @ vertices_homogeneous.T).T
                    transformed_vertices = transformed_vertices_homogeneous[:, :3]
                    component_triangles.append(transformed_vertices)
                    all_transformed_vertices.extend(transformed_vertices)

                    # Use V3 shadow status: 0.0=lit, 1.0=shadowed
                    shadow_status = triangle_statuses_for_cf[i]
                    if shadow_status == 1.0:
                        component_colors.append(shadowed_color_rgba)
                    elif shadow_status == 0.0:
                        component_colors.append(lit_color_rgba)
                    else:
                        self.logger.warning(f"Unknown shadow status {shadow_status} for facet {facet_idx} in CF '{cf_name}'. Coloring grey.")
                        component_colors.append(error_color_rgba)
            
            # Add component triangles to plot
            if component_triangles:
                poly_collection = Poly3DCollection(
                    component_triangles,
                    facecolors=component_colors,
                    edgecolor='k', 
                    linewidths=0.2, 
                    alpha=1.0, 
                    shade=False, 
                    zsort='average'
                )
                ax.add_collection3d(poly_collection)
        
        # Calculate plot bounds
        plot_center = np.array([0.0, 0.0, 0.0])
        plot_range_val = 10.0
        if all_transformed_vertices:
            all_verts_array = np.array(all_transformed_vertices).reshape(-1, 3)
            min_coords = np.min(all_verts_array, axis=0)
            max_coords = np.max(all_verts_array, axis=0)
            plot_center = (min_coords + max_coords) / 2.0
            axis_ranges = max_coords - min_coords
            plot_range_val = np.max(axis_ranges) * 0.75
            if plot_range_val < 1.0:
                plot_range_val = 1.0
        
        # Set axis limits
        ax.set_xlim(plot_center[0] - plot_range_val, plot_center[0] + plot_range_val)
        ax.set_ylim(plot_center[1] - plot_range_val, plot_center[1] + plot_range_val)
        ax.set_zlim(plot_center[2] - plot_range_val, plot_center[2] + plot_range_val)
        
        # Labels
        ax.set_xlabel("X Body (m)")
        ax.set_ylabel("Y Body (m)")
        ax.set_zlabel("Z Body (m)")
        
        # Title with timestamp and info  
        utc_time = self.spice_handler.et_to_utc(epoch)
        # Extract panel angle from component transforms if available
        panel_angle_display = "N/A"
        if component_transforms and "Solar_Panel_North" in component_transforms:
            # Extract angle from rotation matrix
            transform = component_transforms["Solar_Panel_North"]
            panel_angle_display = f"{np.degrees(np.arctan2(transform[1,0], transform[0,0])):.1f}°"
        
        ax.set_title(f"Observer's View of Satellite\\n{utc_time[11:19]} | "
                    f"Phase: {phase_angle:.1f}° | Panel: {panel_angle_display}", fontsize=10)
        
        # Add sun vector (yellow arrow) - k1 vector from body frame origin toward sun
        # Since solar panels are pointing correctly toward sun, use same direction they're using
        arrow_length = plot_range_val * 0.8
        
        # Debug: Let's plot the sun direction that the solar panels are actually using
        # The panels rotate to point their normal toward the sun, so let's verify this
        panel_normal_current = np.array([1, 0, 0])  # Reference normal
        if component_transforms and "Solar_Panel_North" in component_transforms:
            # Get the actual transformed panel normal direction
            transform = component_transforms["Solar_Panel_North"]
            panel_normal_current = transform[:3, :3] @ np.array([1, 0, 0])
            # Plot the panel normal direction (which should point toward sun)
            ax.quiver(0.0, 0.0, 0.0,
                      panel_normal_current[0], panel_normal_current[1], panel_normal_current[2],
                      length=arrow_length, color='orange', normalize=True, 
                      arrow_length_ratio=0.1, pivot='tail', linewidth=2, label='Panel Normal (should point to sun)')
        
        # Plot the sun direction I'm calculating
        ax.quiver(0.0, 0.0, 0.0,  # Plot from body frame origin, not geometric center
                  sun_direction[0], sun_direction[1], sun_direction[2],
                  length=arrow_length, color='yellow', normalize=True, 
                  arrow_length_ratio=0.1, pivot='tail', linewidth=3, label='Sun Direction (k1)')
        
        # Don't show observer vector since we're looking from observer's perspective
        # Instead show a small indicator of viewing direction
        ax.text2D(0.02, 0.98, "Observer View", transform=ax.transAxes, 
                 fontsize=10, verticalalignment='top', 
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        # Body frame axes from origin (smaller since we're viewing from distance)
        axis_len = plot_range_val * 0.2
        ax.quiver(0.0, 0.0, 0.0, 1, 0, 0, 
                  length=axis_len, color='red', label='X_Body', arrow_length_ratio=0.15)
        ax.quiver(0.0, 0.0, 0.0, 0, 1, 0, 
                  length=axis_len, color='lime', label='Y_Body', arrow_length_ratio=0.15)
        ax.quiver(0.0, 0.0, 0.0, 0, 0, 1, 
                  length=axis_len, color='blue', label='Z_Body', arrow_length_ratio=0.15)
        
        # Set camera view from observer's perspective
        # observer_direction (k2) points FROM satellite TO observer
        # We want camera positioned at observer location looking toward satellite
        obs_x, obs_y, obs_z = observer_direction
        
        # For matplotlib view_init, we need to position camera in the k2 direction
        # Convert k2 vector to spherical coordinates for matplotlib view_init
        # Elevation: angle above XY plane (-90 to +90 degrees)
        elevation = np.degrees(np.arcsin(obs_z))
        # Azimuth: angle in XY plane from +X axis (0 to 360 degrees)  
        azimuth = np.degrees(np.arctan2(obs_y, obs_x))
        
        # Apply camera positioning
        if camera_params and 'use_observer_view' in camera_params and not camera_params['use_observer_view']:
            # User explicitly disabled observer view, use provided params
            ax.view_init(elev=camera_params.get('elev', 25), azim=camera_params.get('azim', -45))
        else:
            # Position camera along k2 vector (observer direction) looking at satellite
            ax.view_init(elev=elevation, azim=azimuth)
            
        # Debug info for camera positioning
        self.logger.info(f"FRAME DEBUG - Observer direction (k2): [{obs_x:.3f}, {obs_y:.3f}, {obs_z:.3f}]")
        self.logger.info(f"FRAME DEBUG - Sun direction (k1): [{sun_direction[0]:.3f}, {sun_direction[1]:.3f}, {sun_direction[2]:.3f}]")
        self.logger.info(f"FRAME DEBUG - Camera elev: {elevation:.1f}°, azim: {azimuth:.1f}°")
        self.logger.info(f"FRAME DEBUG - Phase angle: {phase_angle:.1f}°")
        
        # Legend
        ax.legend(fontsize='small', loc='upper right')
        fig.tight_layout()
        
        # Save frame
        frame_path = os.path.join(output_dir, f'frame_{frame_index:04d}.png')
        try:
            plt.savefig(frame_path, dpi=100)
            plt.close()
            return frame_path
        except Exception as e:
            self.logger.error(f"Failed to save frame {frame_path}: {e}")
            plt.close()
            return None
    
    def generate_geometry_animation(self, satellite: Satellite, epochs: np.ndarray, 
                                  satellite_id: int, num_frames: int = 30,
                                  output_filename: str = "geometry_animation.gif",
                                  frames_dir: str = "temp_geometry_frames",
                                  cleanup_frames: bool = True,
                                  camera_params: Optional[Dict] = None,
                                  shadow_database = None) -> str:
        """
        Generate animation showing satellite from observer's perspective over time.
        Shows illumination changes and geometry as seen from ground-based telescope.
        
        Args:
            satellite: Satellite model
            epochs: Array of epoch times  
            satellite_id: SPICE ID of satellite
            num_frames: Number of animation frames
            output_filename: Output GIF filename
            frames_dir: Temporary directory for frames
            cleanup_frames: Whether to delete frame images after GIF creation
            camera_params: Dictionary with camera parameters. Set 'use_observer_view': False 
                          to use manual 'elev' and 'azim' instead of observer perspective
            
        Returns:
            Path to generated GIF file
        """
        self.logger.info(f"Generating geometry animation with {num_frames} frames")
        
        # Create frames directory
        if os.path.exists(frames_dir):
            shutil.rmtree(frames_dir)
        os.makedirs(frames_dir)
        
        # Select epochs for animation frames
        frame_indices = np.linspace(0, len(epochs)-1, num_frames, dtype=int)
        frame_epochs = epochs[frame_indices]
        
        frame_paths = []
        
        for i, (frame_idx, epoch) in enumerate(zip(frame_indices, frame_epochs)):
            # Calculate geometry for this epoch
            sun_direction = self.get_sun_direction_in_body_frame(epoch, satellite_id, 
                                                               satellite.body_frame_name)
            observer_direction, distance = self.get_observer_direction_and_distance(
                epoch, satellite_id, satellite.body_frame_name)
            
            # Calculate phase angle using the dedicated method
            phase_angle = self.calculate_phase_angle(sun_direction, observer_direction)
            
            # Calculate solar panel sun-pointing angle
            from src.utils.geometry_utils import calculate_sun_pointing_rotation
            
            # Solar panel configuration from V3 script
            panel_rotation_axis = np.array([0, 0, 1])  # Z-axis rotation
            panel_normal_reference = np.array([1, 0, 0])  # Panel normal in reference orientation
            
            # Calculate required rotation angle for sun tracking
            panel_angle_deg = calculate_sun_pointing_rotation(
                sun_direction, panel_rotation_axis, panel_normal_reference
            )
            
            # Create rotation matrix around Z-axis
            panel_angle_rad = np.radians(panel_angle_deg)
            cos_a, sin_a = np.cos(panel_angle_rad), np.sin(panel_angle_rad)
            panel_transform = np.array([
                [cos_a, -sin_a, 0, 0],
                [sin_a,  cos_a, 0, 0],
                [0,      0,     1, 0],
                [0,      0,     0, 1]
            ])
            
            component_transforms = {
                "Solar_Panel_North": panel_transform,
                "Solar_Panel_South": panel_transform
            }
            
            # Get shadow data for current sun direction
            shadow_data = None
            if shadow_database:
                # Find closest shadow database entry for current sun direction
                best_match_shadow_map = {}
                max_dot_product = -np.inf
                for db_sun_vector, db_shadow_status_map in shadow_database:
                    dot_product = np.dot(sun_direction, db_sun_vector)
                    if dot_product > max_dot_product:
                        max_dot_product = dot_product
                        best_match_shadow_map = db_shadow_status_map
                shadow_data = best_match_shadow_map
            
            # Generate frame using V3 style with shadow data
            frame_path = self._generate_v3_style_animation_frame(
                satellite, i, epoch, satellite_id, sun_direction, observer_direction,
                distance, phase_angle, frames_dir, camera_params, component_transforms, shadow_data)
            
            if frame_path:
                frame_paths.append(frame_path)
            
            if (i + 1) % 5 == 0:
                self.logger.info(f"Generated frame {i+1}/{num_frames}")
        
        # Create GIF
        self.logger.info("Compiling animation...")
        with imageio.get_writer(output_filename, mode='I', duration=0.3) as writer:
            for frame_path in frame_paths:
                image = imageio.imread(frame_path)
                writer.append_data(image)
        
        # Cleanup
        if cleanup_frames:
            shutil.rmtree(frames_dir)
            self.logger.info(f"Cleaned up temporary frames directory: {frames_dir}")
        
        self.logger.info(f"Animation saved to: {output_filename}")
        return output_filename
    
    def update_satellite_brdf_parameters(self, satellite: Satellite, brdf_params: Dict[str, Dict[str, float]]):
        """
        Update BRDF parameters for specific components.
        
        Args:
            satellite: Satellite model to update
            brdf_params: Dictionary with component names as keys and BRDF parameters as values
                        Format: {"Bus": {"r_d": 0.4, "r_s": 0.2, "n_phong": 10.0}, 
                                "Solar_Panel": {"r_d": 0.2, "r_s": 0.3, "n_phong": 25.0}}
        """
        for component in satellite.components:
            # Check if this component type has BRDF parameters specified
            component_key = None
            for key in brdf_params.keys():
                if key in component.name:
                    component_key = key
                    break
            
            if component_key:
                params = brdf_params[component_key]
                # Update component default material
                component.default_material.r_d = params.get('r_d', component.default_material.r_d)
                component.default_material.r_s = params.get('r_s', component.default_material.r_s)
                component.default_material.n_phong = params.get('n_phong', component.default_material.n_phong)
                
                # Update all facets in this component to use the new material properties
                for facet in component.facets:
                    facet.material_properties.r_d = component.default_material.r_d
                    facet.material_properties.r_s = component.default_material.r_s
                    facet.material_properties.n_phong = component.default_material.n_phong
    
    def update_satellite_brdf_with_manager(self, satellite: Satellite, brdf_manager):
        """
        Update BRDF parameters using the new flexible BRDF manager.
        
        Args:
            satellite: Satellite model to update
            brdf_manager: BRDFManager instance with flexible parameter mapping
        """
        brdf_manager.update_satellite_brdf_parameters(satellite)

    def generate_light_curve(self,
                           satellite: Satellite,
                           epochs: np.ndarray,
                           satellite_id: int,
                           shadow_databases: Optional[List] = None,
                           database_angles: Optional[List[float]] = None,
                           panel_component_names: Optional[List[str]] = None,
                           brdf_params: Optional[Dict[str, Dict[str, float]]] = None,
                           panel_angle_profile: str = "sun_pointing",
                           constant_panel_angle: float = 0.0) -> LightCurveResult:
        """
        Generate a complete light curve for a satellite.
        
        Args:
            satellite: Satellite model with components and facets
            epochs: Array of ephemeris times for light curve generation
            satellite_id: SPICE ID of the satellite
            shadow_databases: List of shadow databases for articulating components
            panel_component_names: Names of articulating panel components
            brdf_params: Optional BRDF parameters to update before generating light curve
            panel_angle_profile: "sun_pointing" or "constant" 
            constant_panel_angle: Fixed angle in degrees (used when panel_angle_profile="constant")
            
        Returns:
            LightCurveResult containing magnitudes and related data
        """
        # Update BRDF parameters if provided
        if brdf_params is not None:
            self.update_satellite_brdf_parameters(satellite, brdf_params)
            
        self.logger.info(f"Generating light curve for {len(epochs)} epochs")
        
        import time
        engine_start_time = time.time()
        
        magnitudes = np.zeros(len(epochs))
        total_flux_array = np.zeros(len(epochs))
        observer_distances = np.zeros(len(epochs))
        
        # Use V3 shadow interpolation system
        lit_fractions = None
        if shadow_databases and len(shadow_databases) > 0:
            # Use tensor-based shadow calculator for maximum performance
            tensor_file = "shadow_databases/ultra_tensor_fixed.npz"
            
            # Get kinematics data in J2000 frame
            print("ENGINE: Getting kinematics data...")
            kinematics_start = time.time()
            sun_positions = []
            sat_positions = []
            sat_att_matrices = []
            
            for epoch in epochs:
                sun_pos, _ = self.spice_handler.get_body_position("SUN", epoch, "J2000", "EARTH")
                sat_pos, _ = self.spice_handler.get_body_position(str(satellite_id), epoch, "J2000", "EARTH")
                att_matrix = self.spice_handler.get_target_orientation("J2000", satellite.body_frame_name, epoch)
                
                sun_positions.append(sun_pos)
                sat_positions.append(sat_pos)
                sat_att_matrices.append(att_matrix)
            
            kinematics_time = time.time() - kinematics_start
            print(f"ENGINE: Kinematics calculation: {kinematics_time:.2f}s")
            
            # Get all conceptual face names
            target_faces = []
            for component in satellite.components:
                if component.conceptual_faces_map:
                    target_faces.extend(component.conceptual_faces_map.keys())
            
            if target_faces:
                # Call V3 shadow interpolation
                print("ENGINE: Starting shadow interpolation...")
                shadow_start = time.time()
                
                # Use provided database_angles or fallback to default spacing
                if database_angles is not None:
                    angles_to_use = database_angles
                else:
                    # Fallback: assume 5° spacing for backwards compatibility  
                    angles_to_use = list(range(0, len(shadow_databases) * 5, 5))
                
                lit_fractions_dict, panel_angles = pytorch_get_lit_fractions_for_kinematics(
                    satellite,
                    sun_pos_j2000_epochs=np.array(sun_positions),
                    sat_pos_j2000_epochs=np.array(sat_positions),
                    sat_att_C_j2000_to_body_epochs=np.array(sat_att_matrices),
                    target_conceptual_faces=target_faces,
                    articulation_offset=0.0
                )
                
                shadow_time = time.time() - shadow_start
                print(f"ENGINE: Shadow interpolation: {shadow_time:.2f}s")
                lit_fractions = lit_fractions_dict
        
        # Process each epoch
        print("ENGINE: Starting BRDF calculations...")
        brdf_start = time.time()
        for i, epoch in enumerate(epochs):
            # Get sun direction and observer direction/distance
            sun_direction = self.get_sun_direction_in_body_frame(epoch, satellite_id, 
                                                               satellite.body_frame_name)
            observer_direction, distance = self.get_observer_direction_and_distance(
                epoch, satellite_id, satellite.body_frame_name)
            
            observer_distances[i] = distance
            
            total_flux = 0.0
            
            # Sum flux from all components and facets
            for component in satellite.components:
                for facet_idx, facet in enumerate(component.facets):
                    # Get lit fraction for this facet
                    facet_lit_fraction = 1.0  # Default to fully lit
                    
                    if lit_fractions is not None:
                        # Find which conceptual face this facet belongs to
                        for face_name, facet_indices in component.conceptual_faces_map.items():
                            if facet_idx in facet_indices:
                                facet_lit_fraction = lit_fractions[face_name][i]
                                break
                    
                    # Calculate flux contribution from this facet
                    facet_flux = self.calculate_facet_flux(
                        facet, sun_direction, observer_direction, distance, facet_lit_fraction)
                    
                    total_flux += facet_flux
            
            total_flux_array[i] = total_flux
            
            # Convert flux to magnitude: m = m� - 2.5 * log��(�)
            if total_flux > 1e-20:  # Avoid log of zero
                magnitudes[i] = self.SUN_APPARENT_MAGNITUDE - 2.5 * np.log10(total_flux)
            else:
                magnitudes[i] = np.inf  # Satellite not visible
        
        brdf_time = time.time() - brdf_start
        total_engine_time = time.time() - engine_start_time
        print(f"ENGINE: BRDF calculations: {brdf_time:.2f}s")
        print(f"ENGINE: Total engine time: {total_engine_time:.2f}s")
        
        self.logger.info(f"Light curve generation complete. Magnitude range: "
                        f"{np.min(magnitudes[np.isfinite(magnitudes)]):.2f} to "
                        f"{np.max(magnitudes[np.isfinite(magnitudes)]):.2f}")
        
        return LightCurveResult(
            epochs=epochs,
            magnitudes=magnitudes,
            total_flux=total_flux_array,
            observer_distances=observer_distances
        )


if __name__ == '__main__':
    # Example usage and testing
    print("LightCurveEngine module loaded successfully")
    print("Use this module to generate satellite light curves with BRDF modeling")
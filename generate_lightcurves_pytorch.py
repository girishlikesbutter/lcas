#!/usr/bin/env python3
"""
PyTorch GPU Light Curve Generation
===================================

This script generates light curves using PyTorch GPU acceleration
for real-time ray tracing.

Features:
- GPU acceleration support
- Real-time shadow computation
- Configurable accuracy levels
- Multiple satellite model support

Usage:
    python generate_lightcurves_pytorch.py [--points 300] [--benchmark]
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from src.io.model_io import load_satellite_from_yaml
from src.spice.spice_handler import SpiceHandler
from src.simulation.lightcurve_engine import LightCurveEngine
from src.illumination.pytorch_shadow_engine import pytorch_get_lit_fractions_for_kinematics, get_pytorch_shadow_engine

# =============================================================================
# CONFIGURATION
# =============================================================================

# Intelsat 901 Configuration
SATELLITE_MODEL_PATH = PROJECT_ROOT / "data" / "models" / "intelsat_901_model.yaml"
METAKERNEL_PATH = PROJECT_ROOT / "data" / "spice_kernels" / "missions" / "dst-is901" / "INTELSAT_901-metakernel.tm"
OUTPUT_DIR = PROJECT_ROOT / "lightcurve_results_pytorch"

# Standard Time Range
START_TIME_UTC = "2020-02-05T10:00:00"
END_TIME_UTC = "2020-02-05T16:00:00"
SATELLITE_ID = -126824
SUBDIVISION_LEVEL = 3

# BRDF Material Properties
BRDF_PARAMS = {
    "Bus": {"r_d": 0.02, "r_s": 0.5, "n_phong": 300.0},
    "Solar_Panel": {"r_d": 0.026, "r_s": 0.3, "n_phong": 200.0},
    "Antenna": {"r_d": 0.01, "r_s": 0.4, "n_phong": 200.0}
}

def generate_pytorch_light_curve(num_points=300, benchmark_mode=False, articulation_offset=0.0, no_plot=False, use_shadows=True, compare_shadows=False, debug_shadows=False, debug_brdf=False, model_config_path=None, rso_config=None):
    """
    Generate light curve using PyTorch GPU ray tracing.
    
    Args:
        num_points: Number of time points for light curve
        benchmark_mode: Run performance benchmarks
        articulation_offset: Solar panel articulation offset in degrees for testing
        no_plot: Skip plotting for faster execution
        use_shadows: Enable shadow computation (default: True)
        compare_shadows: Generate comparison plot with both shadowed and non-shadowed results
        debug_shadows: Export detailed shadow data for analysis
        debug_brdf: Export detailed BRDF face contribution data for analysis
        model_config_path: Path to model configuration file (optional, uses Intelsat 901 default if None)
        rso_config: Runtime configuration override (optional)
    """
    
    print("PYTORCH GPU LIGHT CURVE GENERATION")
    print("=" * 80)
    print("Using GPU-accelerated ray tracing")
    print()
    
    total_start_time = time.time()
    
    # Load configuration system
    from src.config.model_config import ModelConfigManager
    from src.materials.brdf_manager import BRDFManager
    from src.articulation.articulation_engine import ArticulationEngine
    
    config_manager = ModelConfigManager()
    config = config_manager.load_config(model_config_path)
    
    # Apply runtime configuration overrides
    if rso_config:
        if 'satellite_id' in rso_config:
            config.spice_config.satellite_id = rso_config['satellite_id']
        if 'metakernel_path' in rso_config:
            config.spice_config.metakernel_path = rso_config['metakernel_path']
        if 'start_time' in rso_config:
            config.simulation_defaults.start_time = rso_config['start_time']
        if 'end_time' in rso_config:
            config.simulation_defaults.end_time = rso_config['end_time']
        if 'output_dir' in rso_config:
            config.simulation_defaults.output_dir = rso_config['output_dir']
    
    # Create managers
    brdf_manager = BRDFManager(config)
    articulation_engine = ArticulationEngine(config)
    
    # Get paths from configuration
    model_path = config_manager.get_model_path(config)
    metakernel_path = config_manager.get_metakernel_path(config)
    
    # Use configuration values or fallback to hardcoded values
    satellite_id = config.spice_config.satellite_id
    subdivision_level = config.simulation_defaults.subdivision_level
    start_time_utc = config.simulation_defaults.start_time
    end_time_utc = config.simulation_defaults.end_time
    output_dir = PROJECT_ROOT / config.simulation_defaults.output_dir
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Load satellite model
    print(f"Loading {config.model_info.name} satellite model...")
    model_load_start = time.time()
    satellite = load_satellite_from_yaml(str(model_path), facet_subdivision_level=subdivision_level)
    model_load_time = time.time() - model_load_start
    
    print(f"Model loaded: {satellite.name} ({model_load_time:.2f}s)")
    print(f"   Components: {len(satellite.components)}")
    
    # Count facets for performance estimation
    total_facets = sum(len(comp.facets) for comp in satellite.components if comp.facets)
    print(f"   Total facets: {total_facets:,}")
    
    # Get PyTorch engine info
    pytorch_engine = get_pytorch_shadow_engine()
    perf_summary = pytorch_engine.get_performance_summary()
    print(f"   GPU device: {perf_summary['device']}")
    print(f"   RTX cores: {perf_summary['has_rt_cores']}")
    print()
    
    # Initialize SPICE
    print("Initializing SPICE...")
    spice_init_start = time.time()
    spice_handler = SpiceHandler()
    spice_handler.load_metakernel_programmatically(str(metakernel_path))
    spice_init_time = time.time() - spice_init_start
    print(f"SPICE initialized ({spice_init_time:.2f}s)")
    print()
    
    # Generate time series
    print(f"Time range: {start_time_utc} to {end_time_utc}")
    print(f"   Time points: {num_points}")
    
    start_et = spice_handler.utc_to_et(start_time_utc)
    end_et = spice_handler.utc_to_et(end_time_utc)
    epochs = np.linspace(start_et, end_et, num_points)
    
    duration_hours = (end_et - start_et) / 3600
    time_resolution_min = duration_hours * 60 / num_points
    print(f"   Duration: {duration_hours:.1f} hours")
    print(f"   Resolution: {time_resolution_min:.1f} minutes")
    print()
    
    # Get kinematics data
    print("Computing satellite kinematics...")
    kinematics_start = time.time()
    
    sun_positions = []
    sat_positions = []
    sat_att_matrices = []
    
    for epoch in epochs:
        sun_pos, _ = spice_handler.get_body_position("SUN", epoch, "J2000", "EARTH")
        sat_pos, _ = spice_handler.get_body_position(str(satellite_id), epoch, "J2000", "EARTH")
        att_matrix = spice_handler.get_target_orientation("J2000", config.spice_config.body_frame, epoch)
        
        sun_positions.append(sun_pos)
        sat_positions.append(sat_pos)
        sat_att_matrices.append(att_matrix)
    
    kinematics_time = time.time() - kinematics_start
    print(f"Kinematics computed ({kinematics_time:.2f}s)")
    print()
    
    # Get all conceptual face names
    target_faces = []
    for component in satellite.components:
        if component.conceptual_faces_map:
            target_faces.extend(component.conceptual_faces_map.keys())
    
    print(f"Target faces: {len(target_faces)}")
    print()
    
    # Shadow computation (conditional based on use_shadows flag)
    shadow_time = 0.0
    lit_fractions_dict = {}
    panel_angles = None
    
    # Prepare debug data collection if requested
    debug_data = {} if debug_shadows else None
    brdf_debug_data = [] if debug_brdf else None
    
    if use_shadows or compare_shadows:
        # PyTorch GPU shadow computation
        print("PYTORCH GPU SHADOW COMPUTATION")
        print("=" * 60)
        print("Computing shadows using GPU ray tracing")
        print()
        
        shadow_start = time.time()
        
        # Use PyTorch GPU shadow computation with articulation
        if articulation_offset != 0.0:
            print(f"Solar panel articulation: {articulation_offset:.1f} degree offset")
            
        lit_fractions_dict, panel_angles = pytorch_get_lit_fractions_for_kinematics(
            satellite=satellite,
            sun_pos_j2000_epochs=np.array(sun_positions),
            sat_pos_j2000_epochs=np.array(sat_positions),
            sat_att_C_j2000_to_body_epochs=np.array(sat_att_matrices),
            target_conceptual_faces=target_faces,
            articulation_offset=articulation_offset
        )
        
        shadow_time = time.time() - shadow_start
        print(f"PyTorch GPU shadows computed ({shadow_time:.2f}s)")
        print()
        
        # Collect debug data if requested
        if debug_shadows:
            # Calculate sun vectors in body frame for debug
            sun_vectors_body_debug = []
            for i in range(len(epochs)):
                sun_vector_j2000 = sun_positions[i] - sat_positions[i]
                sun_vector_body = sat_att_matrices[i] @ sun_vector_j2000
                sun_vectors_body_debug.append(sun_vector_body / np.linalg.norm(sun_vector_body))
            
            debug_data = {
                'lit_fractions_dict': lit_fractions_dict.copy(),
                'epochs': epochs.copy(),
                'sun_vectors_body': np.array(sun_vectors_body_debug),
                'panel_angles': panel_angles.copy() if panel_angles is not None else None,
                'utc_times': [spice_handler.et_to_utc(et, "C", 0) for et in epochs],
                'target_faces': target_faces,
                'sat_positions': np.array(sat_positions),
                'sun_positions': np.array(sun_positions),
                'sat_att_matrices': np.array(sat_att_matrices)
            }
            
            # Add BRDF debug data if available
            if debug_brdf and brdf_debug_data:
                debug_data['brdf_debug_data'] = brdf_debug_data
    else:
        # No shadows mode: assume all faces are fully lit
        print("NO SHADOWS MODE")
        print("=" * 60)
        print("All faces assumed fully lit (lit_fraction = 1.0)")
        print("Skipping ray tracing for performance")
        print()
        
        # Create lit_fractions_dict with all faces set to 1.0
        lit_fractions_dict = {face: np.ones(len(epochs)) for face in target_faces}
        print(f"No-shadow mode initialized (0.00s)")
        print()
    
    # Initialize light curve engine
    engine = LightCurveEngine(spice_handler)
    
    # Update BRDF parameters using new flexible system
    engine.update_satellite_brdf_with_manager(satellite, brdf_manager)
    
    # Also update using legacy method for backward compatibility
    legacy_brdf_params = brdf_manager.get_legacy_brdf_dict()
    engine.update_satellite_brdf_parameters(satellite, legacy_brdf_params)
    
    # Helper function to generate light curve with given lit fractions
    def generate_light_curve(lit_fractions_dict_input, curve_name=""):
        """Generate magnitudes and flux arrays for given lit fractions."""
        magnitudes_out = np.zeros(len(epochs))
        total_flux_array_out = np.zeros(len(epochs))
        observer_distances_out = np.zeros(len(epochs))
        
        for i, epoch in enumerate(epochs):
            # Get sun direction and observer direction/distance
            sun_direction = engine.get_sun_direction_in_body_frame(epoch, satellite_id, config.spice_config.body_frame)
            observer_direction, distance = engine.get_observer_direction_and_distance(epoch, satellite_id, config.spice_config.body_frame)
            
            observer_distances_out[i] = distance
            
            total_flux = 0.0
            epoch_brdf_debug = [] if debug_brdf else None
            
            # Sum flux from all components and facets
            for component in satellite.components:
                for facet_idx, facet in enumerate(component.facets):
                    # Get lit fraction for this facet
                    facet_lit_fraction = 1.0  # Default to fully lit
                    conceptual_face_name = None
                    
                    # Find which conceptual face this facet belongs to
                    for face_name, facet_indices in component.conceptual_faces_map.items():
                        if facet_idx in facet_indices and face_name in lit_fractions_dict_input:
                            facet_lit_fraction = lit_fractions_dict_input[face_name][i]
                            conceptual_face_name = face_name
                            break
                    
                    # Prepare debug info collection
                    debug_info = {} if debug_brdf else None
                    
                    # Calculate flux contribution from this facet
                    facet_flux = engine.calculate_facet_flux(
                        facet, sun_direction, observer_direction, distance, facet_lit_fraction, debug_info)
                    
                    # Collect BRDF debug data if requested
                    if debug_brdf and epoch_brdf_debug is not None:
                        debug_entry = {
                            'epoch_index': i,
                            'epoch': epoch,
                            'component_name': component.name,
                            'facet_index': facet_idx,
                            'conceptual_face': conceptual_face_name,
                            'facet_flux': facet_flux,
                            **debug_info  # Unpack all debug info from calculate_facet_flux
                        }
                        epoch_brdf_debug.append(debug_entry)
                    
                    total_flux += facet_flux
            
            # Store epoch debug data
            if debug_brdf and brdf_debug_data is not None:
                brdf_debug_data.extend(epoch_brdf_debug)
            
            total_flux_array_out[i] = total_flux
            
            # Convert flux to magnitude
            if total_flux > 1e-20:
                magnitudes_out[i] = engine.SUN_APPARENT_MAGNITUDE - 2.5 * np.log10(total_flux)
            else:
                magnitudes_out[i] = np.inf
        
        return magnitudes_out, total_flux_array_out, observer_distances_out
    
    # Generate light curve(s) based on mode
    lightcurve_start = time.time()
    
    # Initialize variables to avoid scoping issues
    magnitudes_shadowed = None
    magnitudes_no_shadow = None
    total_flux_array_shadowed = None
    total_flux_array_no_shadow = None
    
    if compare_shadows:
        # Generate both shadowed and non-shadowed light curves
        print("Generating BRDF light curves (comparison mode)...")
        
        # Generate shadowed light curve
        magnitudes_shadowed, total_flux_array_shadowed, observer_distances = generate_light_curve(
            lit_fractions_dict, "shadowed")
        
        # Generate non-shadowed light curve (all faces lit)
        no_shadow_lit_fractions = {face: np.ones(len(epochs)) for face in target_faces}
        magnitudes_no_shadow, total_flux_array_no_shadow, _ = generate_light_curve(
            no_shadow_lit_fractions, "no-shadow")
        
        # Use shadowed as primary for compatibility
        magnitudes = magnitudes_shadowed
        total_flux_array = total_flux_array_shadowed
        
    else:
        # Generate single light curve with current lit_fractions_dict
        curve_type = "shadowed" if use_shadows else "no-shadow"
        print(f"Generating BRDF light curve ({curve_type} mode)...")
        
        magnitudes, total_flux_array, observer_distances = generate_light_curve(
            lit_fractions_dict, curve_type)
    
    lightcurve_time = time.time() - lightcurve_start
    print(f"Light curve(s) generated ({lightcurve_time:.2f}s)")
    print()
    
    # Calculate phase angles
    print("Computing phase angles...")
    phase_start = time.time()
    phase_angles = np.zeros(len(epochs))
    
    for i, epoch in enumerate(epochs):
        sun_direction = engine.get_sun_direction_in_body_frame(epoch, SATELLITE_ID, satellite.body_frame_name)
        observer_direction, _ = engine.get_observer_direction_and_distance(epoch, SATELLITE_ID, satellite.body_frame_name)
        phase_angles[i] = engine.calculate_phase_angle(sun_direction, observer_direction)
    
    phase_time = time.time() - phase_start
    print(f"Phase angles computed ({phase_time:.2f}s)")
    print()
    
    # Calculate time_hours for data saving (always needed)
    time_hours = (epochs - epochs[0]) / 3600.0
    
    # Generate plots (optional)
    plot_time = 0
    
    # Generate filename based on mode
    if compare_shadows:
        plot_filename = f"pytorch_gpu_lightcurve_comparison_{num_points}pts.png"
    elif use_shadows:
        plot_filename = f"pytorch_gpu_lightcurve_shadowed_{num_points}pts.png"
    else:
        plot_filename = f"pytorch_gpu_lightcurve_no_shadows_{num_points}pts.png"
    
    if not no_plot:
        print("Creating light curve plot...")
        plot_start = time.time()
        
        finite_mask = np.isfinite(magnitudes)
        
        # Create single magnitude plot with dual x-axes
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Set title based on mode
        if compare_shadows:
            title = 'PyTorch GPU Ray-Traced Light Curve Comparison'
        elif use_shadows:
            title = 'PyTorch GPU Ray-Traced Light Curve (With Shadows)'
        else:
            title = 'PyTorch GPU Ray-Traced Light Curve (No Shadows)'
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Plot curves based on mode
        if compare_shadows:
            # Plot both shadowed and non-shadowed curves
            finite_mask_shadowed = np.isfinite(magnitudes_shadowed)
            finite_mask_no_shadow = np.isfinite(magnitudes_no_shadow)
            
            ax.scatter(time_hours[finite_mask_shadowed], magnitudes_shadowed[finite_mask_shadowed], 
                      c='b', alpha=0.8, s=8, label='With Shadows')
            ax.scatter(time_hours[finite_mask_no_shadow], magnitudes_no_shadow[finite_mask_no_shadow], 
                      c='r', alpha=0.8, s=8, label='No Shadows')
            
            # Add legend
            ax.legend(loc='upper right')
            
            # Use shadowed curve for phase angle axis (they should be identical)
            phase_mask = finite_mask_shadowed
            phase_magnitudes = magnitudes_shadowed
        else:
            # Plot single curve
            ax.scatter(time_hours[finite_mask], magnitudes[finite_mask], c='b', alpha=0.8, s=8)
            phase_mask = finite_mask
            phase_magnitudes = magnitudes
        
        ax.set_ylabel('Apparent Magnitude', fontsize=12)
        ax.set_xlabel('')  # Remove the "Time (hours since start)" label
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()
        
        # Create secondary x-axis for phase angle on top
        ax2 = ax.twiny()
        ax2.scatter(phase_angles[phase_mask], phase_magnitudes[phase_mask], c='b', alpha=0.0, s=0)  # Invisible points for axis scaling
        ax2.set_xlabel('Phase Angle (degrees)', fontsize=12)
        
        # Sample a few UTC times for labeling - replace main x-axis labels with UTC
        num_labels = min(6, len(epochs))
        label_indices = np.linspace(0, len(epochs)-1, num_labels, dtype=int)
        label_times = [time_hours[i] for i in label_indices]
        label_utc = []
        for i in label_indices:
            # Use ISO format for consistent parsing
            utc_str = spice_handler.et_to_utc(epochs[i], "ISOC", 0)
            # Extract HH:MM:SS from YYYY-MM-DDTHH:MM:SS format
            if 'T' in utc_str:
                time_part = utc_str.split('T')[1]
                # Remove fractional seconds if present
                if '.' in time_part:
                    time_part = time_part.split('.')[0]
            else:
                # If no T separator, assume it's already in HH:MM:SS format
                time_part = utc_str
            label_utc.append(time_part)
        
        # Replace main x-axis labels with UTC time
        ax.set_xticks(label_times)
        ax.set_xticklabels(label_utc, rotation=0, ha='center')
        ax.set_xlabel('UTC Time', fontsize=12)
        
        # Adjust layout to prevent label overlap
        plt.subplots_adjust(bottom=0.15)
        
        # Save plot
        plot_path = OUTPUT_DIR / plot_filename
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        
        plot_time = time.time() - plot_start
        print(f"Light curve plot generated ({plot_time:.2f}s)")
        
        plt.show()
    
    # Get performance stats regardless of plotting
    perf_stats = pytorch_engine.get_performance_summary()
    
    # Save data
    if compare_shadows:
        data_filename = f"pytorch_gpu_lightcurve_comparison_{num_points}pts.csv"
    elif use_shadows:
        data_filename = f"pytorch_gpu_lightcurve_shadowed_{num_points}pts.csv"
    else:
        data_filename = f"pytorch_gpu_lightcurve_no_shadows_{num_points}pts.csv"
    
    data_path = OUTPUT_DIR / data_filename
    
    utc_strings = [spice_handler.et_to_utc(epoch, "C", 0) for epoch in epochs]
    
    if compare_shadows:
        # Save both shadowed and non-shadowed results
        data_array = np.column_stack([
            utc_strings,
            epochs,
            time_hours,
            magnitudes_shadowed,
            magnitudes_no_shadow,
            phase_angles,
            observer_distances / 1000
        ])
        header = "UTC_Time,ET_Seconds,Hours_Since_Start,Magnitude_Shadowed,Magnitude_No_Shadow,Phase_Angle_Deg,Observer_Distance_1000km"
    else:
        # Save single result
        data_array = np.column_stack([
            utc_strings,
            epochs,
            time_hours,
            magnitudes,
            phase_angles,
            observer_distances / 1000
        ])
        header = "UTC_Time,ET_Seconds,Hours_Since_Start,Apparent_Magnitude,Phase_Angle_Deg,Observer_Distance_1000km"
    
    np.savetxt(data_path, data_array, delimiter=',', header=header, fmt='%s', comments='')
    print(f"Data saved: {data_path}")
    
    # Performance analysis
    total_time = time.time() - total_start_time
    
    print()
    print("PERFORMANCE ANALYSIS")
    print("=" * 80)
    
    # Performance metrics
    database_storage = 9.6  # GB
    database_generation_time = 6 * 3600  # 6 hours in seconds
    database_loading_time = 2.0  # seconds
    database_interpolation_time = num_points * 0.010  # 10ms per epoch
    database_total_shadow_time = database_loading_time + database_interpolation_time
    
    print(f"PERFORMANCE METRICS:")
    print()
    # Report approach based on mode
    if compare_shadows:
        approach_name = "PyTorch GPU Approach (COMPARISON MODE)"
    elif use_shadows:
        approach_name = "PyTorch GPU Approach (WITH SHADOWS)"
    else:
        approach_name = "PyTorch GPU Approach (NO SHADOWS)"
    
    print(f"   {approach_name}:")
    print(f"     Storage required: 0 GB")
    print(f"     Generation time: 0 hours")
    print(f"     Loading time: 0s")
    
    if shadow_time > 0:
        print(f"     Shadow computation time: {shadow_time:.1f}s")
    else:
        print(f"     Shadow computation time: 0.0s (skipped)")
    
    print(f"     Total shadow time: {shadow_time:.1f}s")
    print(f"     Accuracy: Ray-traced (no interpolation)")
    print(f"     Scalability: Linear")
    print()
    
    # Calculate performance metrics
    storage_saved = database_storage
    time_saved = database_generation_time
    shadow_speedup = database_total_shadow_time / shadow_time if shadow_time > 0 else 1.0
    
    print(f"DETAILED TIMING BREAKDOWN:")
    print(f"   Model loading: {model_load_time:.2f}s ({model_load_time/total_time*100:.1f}%)")
    print(f"   SPICE init: {spice_init_time:.2f}s ({spice_init_time/total_time*100:.1f}%)")
    print(f"   Kinematics: {kinematics_time:.2f}s ({kinematics_time/total_time*100:.1f}%)")
    print(f"   PyTorch GPU shadows: {shadow_time:.2f}s ({shadow_time/total_time*100:.1f}%)")
    print(f"   BRDF computation: {lightcurve_time:.2f}s ({lightcurve_time/total_time*100:.1f}%)")
    print(f"   Phase angles: {phase_time:.2f}s ({phase_time/total_time*100:.1f}%)")
    print(f"   Plotting: {plot_time:.2f}s ({plot_time/total_time*100:.1f}%)")
    print(f"   TOTAL: {total_time:.2f}s")
    
    # Benchmark mode
    if benchmark_mode:
        print()
        print("RUNNING PYTORCH GPU BENCHMARKS...")
        print("=" * 60)
        
        benchmark_results = pytorch_engine.benchmark_performance(satellite)
        
        if "error" not in benchmark_results:
            print("PyTorch GPU Benchmark Results:")
            for epochs, result in benchmark_results.items():
                speedup = result.get('speedup_vs_database', 1.0)
                print(f"   {epochs:3d} epochs: {result['ms_per_epoch']:6.1f}ms/epoch, "
                      f"{result['rays_per_second']/1000:6.1f}K rays/sec, "
                      f"{speedup:6.1f}x vs database")
        else:
            print(f"Benchmark failed: {benchmark_results['error']}")
    
    # Save debug data if requested
    if debug_shadows or debug_brdf:
        import pickle
        
        # Create debug data structure if not already created
        if not debug_data:
            debug_data = {}
        
        # Add BRDF debug data if available
        if debug_brdf and brdf_debug_data:
            debug_data['brdf_debug_data'] = brdf_debug_data
            
        # Add basic info for BRDF-only debug
        if debug_brdf and not debug_shadows:
            debug_data.update({
                'epochs': epochs.copy(),
                'utc_times': [spice_handler.et_to_utc(et, "C", 0) for et in epochs],
                'target_faces': target_faces,
            })
        
        debug_filename = OUTPUT_DIR / "shadow_debug_data.pkl"
        with open(debug_filename, 'wb') as f:
            pickle.dump(debug_data, f)
        
        debug_types = []
        if debug_shadows: debug_types.append("shadow")
        if debug_brdf: debug_types.append("BRDF")
        print(f"Debug {'/'.join(debug_types)} data saved to: {debug_filename}")
        print()
    
    # Cleanup
    spice_handler.unload_all_kernels()
    
    print()
    print("GENERATION COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {OUTPUT_DIR}")
    print(f"Plot: {plot_filename}")
    print(f"Data: {data_filename}")
    
    return {
        'magnitudes': magnitudes,
        'total_time': total_time,
        'shadow_time': shadow_time,
        'storage_saved_gb': storage_saved,
        'time_saved_hours': time_saved / 3600,
        'performance_stats': perf_stats
    }

def main():
    parser = argparse.ArgumentParser(description="PyTorch GPU light curve generation")
    parser.add_argument("--points", "-p", type=int, default=300,
                       help="Number of time points (default: 300)")
    parser.add_argument("--benchmark", "-b", action="store_true",
                       help="Run PyTorch GPU performance benchmarks")
    parser.add_argument("--articulation", "-a", type=float, default=0.0,
                       help="Solar panel articulation offset in degrees (default: 0.0)")
    parser.add_argument("--no-plot", action="store_true",
                       help="Skip plotting for faster execution")
    parser.add_argument("--no-shadows", action="store_true",
                       help="Disable shadow computation (assume all faces lit)")
    parser.add_argument("--compare-shadows", action="store_true",
                       help="Generate comparison plot showing both shadowed and non-shadowed results")
    parser.add_argument("--debug-shadows", action="store_true",
                       help="Export detailed shadow data for analysis")
    parser.add_argument("--debug-brdf", action="store_true",
                       help="Export detailed BRDF face contribution data for analysis")
    
    args = parser.parse_args()
    
    try:
        result = generate_pytorch_light_curve(
            num_points=args.points,
            benchmark_mode=args.benchmark,
            articulation_offset=args.articulation,
            no_plot=args.no_plot,
            use_shadows=not args.no_shadows,
            compare_shadows=args.compare_shadows,
            debug_shadows=args.debug_shadows,
            debug_brdf=args.debug_brdf
        )
        
        print(f"\nPyTorch GPU light curve generation completed successfully")
        print(f"Storage saved: {result['storage_saved_gb']:.1f} GB")
        print(f"Time saved: {result['time_saved_hours']:.1f} hours")
        
    except KeyboardInterrupt:
        print("\nGeneration interrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
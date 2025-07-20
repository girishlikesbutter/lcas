# src/utils/geometry_utils.py

import numpy as np
import quaternion  # numpy-quaternion
import logging
from typing import Tuple, List, Dict, Any

# Assuming Facet and BRDFMaterialProperties are accessible
# This might require adjusting imports if this file is moved or run standalone.
try:
    from src.models.model_definitions import Facet, BRDFMaterialProperties
except ImportError:
    # Define dummy classes if running standalone for syntax checking,
    # but this indicates a path issue if it happens during project execution.
    logging.warning(
        "Could not import Facet/BRDFMaterialProperties from src.models.model_definitions. Using dummies for geometry_utils.")
    from dataclasses import dataclass, field


    @dataclass
    class BRDFMaterialProperties:
        r_d: float = 0.0; r_s: float = 0.0; n_phong: float = 1.0


    @dataclass
    class Facet:
        id: str; vertices: List[np.ndarray]; normal: np.ndarray; area: float; material_properties: Any = None

logger = logging.getLogger(__name__)


def calculate_orientation_from_body_vectors(
        body_x_axis: np.ndarray,
        body_y_axis: np.ndarray
) -> np.quaternion:
    """
    Calculates the orientation quaternion of a component's local frame relative
    to the body frame, given the component's local x and y axes expressed in
    the body frame.
    """
    if not isinstance(body_x_axis, np.ndarray) or not isinstance(body_y_axis, np.ndarray):
        raise TypeError("Input axes must be NumPy arrays.")
    if body_x_axis.shape != (3,) or body_y_axis.shape != (3,):
        raise ValueError("Input axes must be 3-element vectors.")

    norm_x = np.linalg.norm(body_x_axis)
    norm_y = np.linalg.norm(body_y_axis)

    if norm_x < 1e-9 or norm_y < 1e-9:
        raise ValueError("Input axis vectors must not be zero vectors.")

    x_ax_norm = body_x_axis / norm_x
    y_ax_norm = body_y_axis / norm_y

    if not np.isclose(np.dot(x_ax_norm, y_ax_norm), 0.0, atol=1e-6):
        raise ValueError(
            f"Provided body_x_axis and body_y_axis are not orthogonal. Dot product: {np.dot(x_ax_norm, y_ax_norm)}")

    z_ax_norm = np.cross(x_ax_norm, y_ax_norm)
    norm_z = np.linalg.norm(z_ax_norm)
    if norm_z < 1e-9:
        raise ValueError("Resulting z-axis is a zero vector, check input x and y axes.")
    z_ax_norm /= norm_z

    rotation_matrix = np.column_stack((x_ax_norm, y_ax_norm, z_ax_norm))
    q = quaternion.from_rotation_matrix(rotation_matrix)
    return q


def apply_initial_rotation_around_pivot(
        current_rel_pos_comp_origin_in_body: np.ndarray,
        current_rel_orient_comp_to_body: np.quaternion,
        comp_local_axis_of_rotation: np.ndarray,
        body_center_of_rotation: np.ndarray,
        starting_angle_rad: float
) -> Tuple[np.ndarray, np.quaternion]:
    """
    Applies an initial rotation to a component around a specified pivot point.
    """
    if not isinstance(current_rel_pos_comp_origin_in_body, np.ndarray) or \
            not isinstance(comp_local_axis_of_rotation, np.ndarray) or \
            not isinstance(body_center_of_rotation, np.ndarray):
        raise TypeError("Position and axis vectors must be NumPy arrays.")
    if not isinstance(current_rel_orient_comp_to_body, np.quaternion):  # type: ignore
        raise TypeError("Orientation must be a numpy.quaternion.")

    if np.linalg.norm(comp_local_axis_of_rotation) < 1e-9:
        raise ValueError("Component local axis of rotation must be a non-zero vector.")

    local_axis_norm = comp_local_axis_of_rotation / np.linalg.norm(comp_local_axis_of_rotation)
    local_axis_quat = np.quaternion(0, *local_axis_norm)

    body_axis_rotated_quat = current_rel_orient_comp_to_body * local_axis_quat * current_rel_orient_comp_to_body.conjugate()
    body_axis_of_rotation_vec = np.array([body_axis_rotated_quat.x, body_axis_rotated_quat.y, body_axis_rotated_quat.z])

    body_axis_norm = np.linalg.norm(body_axis_of_rotation_vec)
    if body_axis_norm < 1e-9:
        raise ValueError("Transformed axis of rotation is zero vector in body frame.")
    body_axis_of_rotation_vec_normalized = body_axis_of_rotation_vec / body_axis_norm

    angle_half = starting_angle_rad / 2.0
    sin_angle_half = np.sin(angle_half)
    q_rot = np.quaternion(
        np.cos(angle_half),
        sin_angle_half * body_axis_of_rotation_vec_normalized[0],
        sin_angle_half * body_axis_of_rotation_vec_normalized[1],
        sin_angle_half * body_axis_of_rotation_vec_normalized[2]
    ).normalized()

    P_pivot_body = body_center_of_rotation
    O_comp_origin_body = current_rel_pos_comp_origin_in_body
    V_pivot_to_origin = O_comp_origin_body - P_pivot_body
    V_pivot_to_origin_quat = np.quaternion(0, *V_pivot_to_origin)
    V_rotated_quat = q_rot * V_pivot_to_origin_quat * q_rot.conjugate()
    V_pivot_to_origin_rotated_vec = np.array([V_rotated_quat.x, V_rotated_quat.y, V_rotated_quat.z])

    O_new_comp_origin_body = P_pivot_body + V_pivot_to_origin_rotated_vec
    q_new_comp_to_body = (q_rot * current_rel_orient_comp_to_body).normalized()

    return O_new_comp_origin_body, q_new_comp_to_body


def calculate_sun_pointing_rotation(sun_vector: np.ndarray, panel_axis: np.ndarray, panel_normal: np.ndarray) -> float:
    """
    Calculates the required rotation angle around the panel axis to align the panel normal with the sun vector.
    
    Args:
        sun_vector: Direction to the sun in the satellite body frame (3D vector)
        panel_axis: Axis of rotation for the panel in the satellite body frame (3D vector)
        panel_normal: Current normal direction of the panel in the satellite body frame (3D vector)
    
    Returns:
        Rotation angle in degrees needed to align panel_normal with sun_vector around panel_axis
    """
    if not isinstance(sun_vector, np.ndarray) or not isinstance(panel_axis, np.ndarray) or not isinstance(panel_normal, np.ndarray):
        raise TypeError("All input vectors must be NumPy arrays.")
    
    if sun_vector.shape != (3,) or panel_axis.shape != (3,) or panel_normal.shape != (3,):
        raise ValueError("All input vectors must be 3-element vectors.")
    
    # Normalize input vectors
    sun_norm = np.linalg.norm(sun_vector)
    axis_norm = np.linalg.norm(panel_axis)
    normal_norm = np.linalg.norm(panel_normal)
    
    if sun_norm < 1e-9 or axis_norm < 1e-9 or normal_norm < 1e-9:
        raise ValueError("Input vectors must not be zero vectors.")
    
    sun_unit = sun_vector / sun_norm
    axis_unit = panel_axis / axis_norm
    normal_unit = panel_normal / normal_norm
    
    # Project sun vector and panel normal onto the plane perpendicular to the rotation axis
    sun_projected = sun_unit - np.dot(sun_unit, axis_unit) * axis_unit
    normal_projected = normal_unit - np.dot(normal_unit, axis_unit) * axis_unit
    
    # Normalize projected vectors
    sun_proj_norm = np.linalg.norm(sun_projected)
    normal_proj_norm = np.linalg.norm(normal_projected)
    
    if sun_proj_norm < 1e-9 or normal_proj_norm < 1e-9:
        # Sun vector or panel normal is parallel to rotation axis, no rotation needed
        return 0.0
    
    sun_proj_unit = sun_projected / sun_proj_norm
    normal_proj_unit = normal_projected / normal_proj_norm
    
    # Calculate angle between projected vectors using atan2 for robustness
    cos_angle = np.dot(normal_proj_unit, sun_proj_unit)
    sin_angle = np.dot(np.cross(normal_proj_unit, sun_proj_unit), axis_unit)
    
    angle_rad = np.arctan2(sin_angle, cos_angle)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg


def _subdivide_triangle_recursive(
        triangle_vertices_list_of_arrays: List[np.ndarray],
        current_level: int
) -> List[List[np.ndarray]]:
    """
    Recursively subdivides a triangle into 4 smaller triangles.
    Input: List of 3 np.ndarray vertices.
    Output: List of lists, where each inner list contains 3 np.ndarray vertices of a sub-triangle.
    """
    if current_level == 0:
        return [triangle_vertices_list_of_arrays]

    p0, p1, p2 = triangle_vertices_list_of_arrays[0], triangle_vertices_list_of_arrays[1], \
    triangle_vertices_list_of_arrays[2]

    m01 = (p0 + p1) / 2.0
    m12 = (p1 + p2) / 2.0
    m20 = (p2 + p0) / 2.0

    sub_triangles_vertex_sets: List[List[np.ndarray]] = []
    # Standard Loop-like subdivision for consistent winding if p0,p1,p2 is CCW
    sub_triangles_vertex_sets.extend(_subdivide_triangle_recursive([p0, m01, m20], current_level - 1))
    sub_triangles_vertex_sets.extend(_subdivide_triangle_recursive([m01, p1, m12], current_level - 1))
    # Ensure consistent winding for the third outer triangle: p2, m20, m12
    sub_triangles_vertex_sets.extend(_subdivide_triangle_recursive([p2, m20, m12], current_level - 1))
    sub_triangles_vertex_sets.extend(
        _subdivide_triangle_recursive([m01, m12, m20], current_level - 1))  # Central triangle

    return sub_triangles_vertex_sets


def generate_facets_from_conceptual_definitions(
        conceptual_defs: Dict[str, List[List[float]]],
        default_material: BRDFMaterialProperties,
        comp_name_for_id_prefix: str,
        subdivision_level: int
) -> Tuple[List[Facet], Dict[str, List[int]]]:
    """
    Generates a list of triangular Facet objects from conceptual face definitions,
    applying fan triangulation and recursive 1-to-4 subdivision. Also builds
    the conceptual_faces_map.

    Args:
        conceptual_defs: Dictionary mapping conceptual face names to lists of raw
                         vertex coordinates (List[List[float]]).
        default_material: The BRDFMaterialProperties to assign to generated facets.
        comp_name_for_id_prefix: A string prefix for generating unique facet IDs.
        subdivision_level: The number of 1-to-4 subdivisions to apply to each
                           base triangle from the initial fan triangulation.

    Returns:
        A tuple containing:
            - all_generated_facets: A list of final Facet objects.
            - conceptual_faces_map: A dictionary mapping original conceptual face
              names to lists of indices in all_generated_facets.
    """
    all_generated_facets: List[Facet] = []
    conceptual_faces_map: Dict[str, List[int]] = {}
    global_facet_index_counter = 0  # Tracks index in all_generated_facets

    if default_material is None:  # Ensure a material is available
        logger.warning(
            f"No default_material provided for {comp_name_for_id_prefix}, using a dummy BRDFMaterialProperties.")
        default_material = BRDFMaterialProperties()

    for conceptual_face_name, raw_vertex_lists_for_face in conceptual_defs.items():
        if len(raw_vertex_lists_for_face) < 3:
            logger.warning(f"Conceptual face '{conceptual_face_name}' in component '{comp_name_for_id_prefix}' "
                           f"has < 3 vertices ({len(raw_vertex_lists_for_face)}). Skipping this face.")
            conceptual_faces_map[conceptual_face_name] = []  # Mark as having no facets
            continue

        # Convert raw vertex lists (List[List[float]]) to list of np.ndarray
        polygon_vertices_np_list: List[np.ndarray] = [np.array(v, dtype=float) for v in raw_vertex_lists_for_face]

        # Initial Fan Triangulation
        base_triangles_vertices_list: List[List[np.ndarray]] = []  # Each item is [v0, v1, v2] as np.arrays
        v0_fan = polygon_vertices_np_list[0]
        for i in range(1, len(polygon_vertices_np_list) - 1):
            v1_fan = polygon_vertices_np_list[i]
            v2_fan = polygon_vertices_np_list[i + 1]
            base_triangles_vertices_list.append([v0_fan, v1_fan, v2_fan])

        facets_indices_for_this_conceptual_face: List[int] = []

        # Subdivide each base triangle and create Facet objects
        for base_tri_idx, base_triangle_vertex_arrays in enumerate(base_triangles_vertices_list):
            subdivided_vertex_sets = _subdivide_triangle_recursive(base_triangle_vertex_arrays, subdivision_level)

            for sub_tri_idx, single_sub_triangle_vertex_arrays in enumerate(subdivided_vertex_sets):
                # single_sub_triangle_vertex_arrays is [p0_arr, p1_arr, p2_arr]
                v0, v1, v2 = single_sub_triangle_vertex_arrays[0], single_sub_triangle_vertex_arrays[1], \
                single_sub_triangle_vertex_arrays[2]

                edge1 = v1 - v0
                edge2 = v2 - v0
                normal_vec = np.cross(edge1, edge2)
                norm_magnitude = np.linalg.norm(normal_vec)

                area_val = 0.0
                if norm_magnitude > 1e-9:
                    normal_vec /= norm_magnitude
                    area_val = 0.5 * norm_magnitude
                else:
                    normal_vec = np.array([0.0, 0.0, 1.0])  # Default for degenerate
                    logger.debug(f"Degenerate sub-triangle in '{comp_name_for_id_prefix}', "
                                 f"face '{conceptual_face_name}', base_tri {base_tri_idx}, sub_tri {sub_tri_idx}.")

                facet_id = f"{comp_name_for_id_prefix}_{conceptual_face_name}_f{global_facet_index_counter}"

                # Facet.vertices expects List[Vector3D], which is List[np.ndarray]
                current_facet = Facet(
                    id=facet_id,
                    vertices=single_sub_triangle_vertex_arrays,  # This is List[np.ndarray]
                    normal=normal_vec,
                    area=area_val,
                    material_properties=default_material
                )
                all_generated_facets.append(current_facet)
                facets_indices_for_this_conceptual_face.append(global_facet_index_counter)
                global_facet_index_counter += 1

        conceptual_faces_map[conceptual_face_name] = facets_indices_for_this_conceptual_face
        logger.debug(
            f"Generated {len(facets_indices_for_this_conceptual_face)} facets for conceptual face '{conceptual_face_name}' of '{comp_name_for_id_prefix}'.")

    logger.info(f"Total facets generated for component '{comp_name_for_id_prefix}': {len(all_generated_facets)}")
    return all_generated_facets, conceptual_faces_map


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)  # Enable debug for testing this module
    logger.info("--- Testing geometry_utils.py ---")

    # Test calculate_orientation_from_body_vectors (already has tests)
    # ...

    # Test apply_initial_rotation_around_pivot (already has tests)
    # ...

    logger.info("\n--- Testing generate_facets_from_conceptual_definitions ---")
    test_material = BRDFMaterialProperties(r_d=0.5, r_s=0.2, n_phong=10)

    # Test Case 1: Simple square face, no subdivision
    square_face_raw = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]  # Standard CCW for +Z normal
    test_conceptual_defs_1 = {"Front_Square": square_face_raw}
    facets1, map1 = generate_facets_from_conceptual_definitions(
        test_conceptual_defs_1, test_material, "TestComp1", subdivision_level=0
    )
    logger.info(f"Test 1 (Square, subdiv=0): Got {len(facets1)} facets. Map: {map1}")
    # Expected: 2 facets from fan triangulation.
    assert len(facets1) == 2
    assert len(map1["Front_Square"]) == 2
    if facets1:
        logger.info(
            f"  Facet 0 normal: {facets1[0].normal}, area: {facets1[0].area:.3f}")  # Should be [0,0,1], area 0.5
        assert np.allclose(facets1[0].normal, [0, 0, 1])
        assert np.isclose(facets1[0].area, 0.5)
        assert np.allclose(facets1[1].normal, [0, 0, 1])
        assert np.isclose(facets1[1].area, 0.5)

    # Test Case 2: Same square face, 1 level of subdivision
    test_conceptual_defs_2 = {"Front_Square_Sub1": square_face_raw}  # Use a new name for map key
    facets2, map2 = generate_facets_from_conceptual_definitions(
        test_conceptual_defs_2, test_material, "TestComp2", subdivision_level=1
    )
    logger.info(f"Test 2 (Square, subdiv=1): Got {len(facets2)} facets. Map: {map2}")
    # Expected: 2 base triangles * 4^1 sub-triangles/base = 8 facets
    assert len(facets2) == 8
    assert len(map2["Front_Square_Sub1"]) == 8
    if facets2:
        total_area_subdiv = sum(f.area for f in facets2)
        logger.info(f"  Total area from {len(facets2)} subdivided facets: {total_area_subdiv:.3f} (Expected: 1.0)")
        assert np.isclose(total_area_subdiv, 1.0)
        # All normals should still be [0,0,1] for this planar case
        for f_idx, f in enumerate(facets2):
            assert np.allclose(f.normal, [0, 0, 1]), f"Subdivided facet {f_idx} normal incorrect: {f.normal}"

    # Test Case 3: Single triangle face, 2 levels of subdivision
    triangle_face_raw = [[0, 0, 0], [1, 0, 0], [0.5, np.sqrt(3) / 2, 0]]  # Equilateral triangle
    test_conceptual_defs_3 = {"Front_Triangle_Sub2": triangle_face_raw}
    facets3, map3 = generate_facets_from_conceptual_definitions(
        test_conceptual_defs_3, test_material, "TestComp3", subdivision_level=2
    )
    logger.info(f"Test 3 (Triangle, subdiv=2): Got {len(facets3)} facets. Map: {map3}")
    # Expected: 1 base triangle * 4^2 sub-triangles/base = 16 facets
    assert len(facets3) == 16
    assert len(map3["Front_Triangle_Sub2"]) == 16
    if facets3:
        original_area_tri = 0.5 * 1 * (np.sqrt(3) / 2)  # base * height / 2
        total_area_subdiv_tri = sum(f.area for f in facets3)
        logger.info(
            f"  Total area from {len(facets3)} subdivided facets: {total_area_subdiv_tri:.4f} (Expected: {original_area_tri:.4f})")
        assert np.isclose(total_area_subdiv_tri, original_area_tri)
        for f_idx, f in enumerate(facets3):
            assert np.allclose(f.normal,
                               [0, 0, 1]), f"Subdivided facet {f_idx} normal incorrect for triangle: {f.normal}"

    logger.info("--- geometry_utils.py tests completed ---")


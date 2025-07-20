# lcforge/src/models/model_definitions.py

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Any, Dict

# Import base types from common_types
# Vector3D is np.ndarray, Quaternion is np.quaternion as defined in common_types.py
from src.core.common_types import Vector3D, Quaternion


@dataclass
class BRDFMaterialProperties:
    """
    Represents the Bidirectional Reflectance Distribution Function (BRDF)
    material properties for a facet, based on a Phong model. (FR1.4)

    Attributes:
        r_d: Diffuse reflectivity coefficient (0 to 1). (FR1.4)
        r_s: Specular reflectivity coefficient (0 to 1). (FR1.4)
        n_phong: Phong exponent (shininess). (FR1.4)
    """
    r_d: float = 0.0
    r_s: float = 0.0
    n_phong: float = 1.0


@dataclass
class Facet:
    """
    Represents a single reflective triangular facet of a satellite component.
    Used for light curve calculations. These are typically generated from
    conceptual_face_definitions in the Component class.

    Attributes:
        id: Unique identifier for the facet.
        vertices: List of 3D vectors (np.ndarray) defining the facet's corners.
        normal: The outward-pointing normal vector (np.ndarray) of the facet.
        area: Surface area of the facet.
        material_properties: BRDF properties of the facet.
    """
    id: str
    vertices: List[Vector3D] = field(default_factory=list)  # Expects list of np.ndarray
    normal: Vector3D = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    area: float = 0.0
    material_properties: BRDFMaterialProperties = field(default_factory=BRDFMaterialProperties)


@dataclass
class Component:
    """
    Represents a distinct component of a satellite (e.g., bus, solar panel, antenna).
    Geometry is defined via conceptual faces, which are then processed into triangular facets.
    """
    id: str
    name: str

    # User-defined raw vertices for conceptual faces in the component's local frame.
    # Key: conceptual face name (e.g., "Bus_Face_X_Positive", "Solar_Panel_Front").
    # Value: List of vertices, where each vertex is a List[float] of [x,y,z] coordinates.
    # This field is persisted (saved/loaded).
    conceptual_face_definitions: Dict[str, List[List[float]]] = field(default_factory=dict)

    # Dynamically generated triangular facets for light curve calculation and visualization.
    # Populated at load time or on demand from conceptual_face_definitions and a subdivision_level.
    # Not typically directly serialized if always derived. repr=False to keep __repr__ clean.
    facets: List[Facet] = field(default_factory=list, repr=False)

    # Maps conceptual face names to indices in the 'facets' list.
    # Also populated dynamically alongside 'facets'. repr=False for cleanliness.
    conceptual_faces_map: Dict[str, List[int]] = field(default_factory=dict, repr=False)

    relative_position: Vector3D = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    relative_orientation: Quaternion = field(default_factory=lambda: np.quaternion(1, 0, 0, 0))  # Identity quaternion

    # Articulation rule (e.g., string identifier for a SPICE frame or a specific tracking logic)
    # This defines dynamic articulation during simulation.
    articulation_rule: Optional[Any] = None

    # Default material properties for facets generated from this component's conceptual_face_definitions.
    # Can be overridden if individual Facet objects have their material_properties explicitly set
    # (though typically facets will inherit this default when generated).
    default_material: Optional[BRDFMaterialProperties] = None


@dataclass
class Satellite:
    """
    Represents the overall satellite model, composed of multiple components. (FR1.2)
    Defines the satellite's primary body-fixed reference frame.

    Attributes:
        id: Unique identifier for the satellite.
        name: Descriptive name of the satellite.
        components: List of Component objects that make up the satellite. (FR1.2)
        body_frame_name: The name of the SPICE reference frame rigidly attached
                         to the satellite's main body. Component positions and
                         orientations are defined relative to this frame. (FR2.2 related)
    """
    id: str
    name: str
    components: List[Component] = field(default_factory=list)
    body_frame_name: str = ""  # Should be a valid SPICE frame name

    def transform_components_by_name(self, component_names: List[str], transform_matrix: np.ndarray):
        """
        Applies a transformation matrix to the conceptual face definitions of specified components.
        This allows for dynamic articulation of components such as rotating solar panels.
        
        Args:
            component_names: List of component names to transform
            transform_matrix: 4x4 homogeneous transformation matrix to apply
        """
        for component in self.components:
            if component.name in component_names:
                # Transform the vertices in each conceptual face definition
                for face_name, vertices_list in component.conceptual_face_definitions.items():
                    # Convert vertices to homogeneous coordinates and apply transformation
                    transformed_vertices = []
                    for vertex in vertices_list:
                        # Convert to homogeneous coordinates (add w=1)
                        homogeneous_vertex = np.array([vertex[0], vertex[1], vertex[2], 1.0])
                        # Apply transformation
                        transformed_vertex = transform_matrix @ homogeneous_vertex
                        # Convert back to 3D coordinates
                        transformed_vertices.append([transformed_vertex[0], transformed_vertex[1], transformed_vertex[2]])
                    component.conceptual_face_definitions[face_name] = transformed_vertices


if __name__ == '__main__':
    # This section is for example usage and basic testing.
    # It demonstrates how the dataclasses might be instantiated.

    # 1. Define Material Properties
    bus_material = BRDFMaterialProperties(r_d=0.4, r_s=0.1, n_phong=5.0)

    # 2. Define a Component (Satellite Bus)
    # Raw vertex data for conceptual faces (example for one face)
    bus_face_x_pos_vertices_raw = [
        [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [0.5, 0.5, 0.5], [0.5, -0.5, 0.5]
    ]

    satellite_bus_component = Component(
        id="BUS001",
        name="SatelliteBus",
        conceptual_face_definitions={
            "Bus_X_Positive_Face": bus_face_x_pos_vertices_raw
            # Other faces would be defined here
        },
        relative_position=np.array([0.0, 0.0, 0.0]),
        relative_orientation=np.quaternion(1, 0, 0, 0),  # Identity
        default_material=bus_material
        # facets and conceptual_faces_map would be populated by a separate utility function
        # based on conceptual_face_definitions and a chosen subdivision level.
    )

    # 3. Define the Satellite
    my_satellite = Satellite(
        id="SAT001_TEST",
        name="TestSat",
        components=[satellite_bus_component],
        body_frame_name="TESTSAT_BODY_FIXED"
    )

    print("--- Example Satellite Configuration (Raw Definitions) ---")
    print(f"Satellite: {my_satellite.name} (ID: {my_satellite.id})")
    print(f"  Body Frame: {my_satellite.body_frame_name}")
    for comp in my_satellite.components:
        print(f"  Component: {comp.name} (ID: {comp.id})")
        print(f"    Rel Pos: {comp.relative_position}, Rel Orient: {comp.relative_orientation}")
        print(f"    Default Material: Rd={comp.default_material.r_d if comp.default_material else 'N/A'}")
        print(f"    Conceptual Face Definitions Count: {len(comp.conceptual_face_definitions)}")
        for face_name, raw_verts in comp.conceptual_face_definitions.items():
            print(f"      Face '{face_name}': {len(raw_verts)} raw vertices defined.")
        print(f"    Generated Facets Count: {len(comp.facets)} (Expected to be 0 or dynamically populated later)")
        print(f"    Conceptual Faces Map Size: {len(comp.conceptual_faces_map)}")

    print("\n--- End Example ---")

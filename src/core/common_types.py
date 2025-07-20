# lcforge/src/core/common_types.py

import numpy as np
import quaternion  # For numpy-quaternion library
from typing import List, Any, TYPE_CHECKING, Optional
from dataclasses import dataclass, field

# Forward declaration for type hinting with SpiceHandler to avoid circular imports
if TYPE_CHECKING:
    from src.spice.spice_handler import SpiceHandler  # Adjusted path

# --- Basic Geometric and Physical Types ---

# Vector3D: A 3-element NumPy array representing a vector in 3D space.
# Typically used for positions, velocities, pointing vectors, etc.
Vector3D = np.ndarray  # np.array([x, y, z])

# RotationMatrix: A 3x3 NumPy array representing a rotation in 3D space.
RotationMatrix = np.ndarray  # np.array([[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]])

# Quaternion: Using the numpy-quaternion library.
# This provides a robust quaternion implementation with many useful operations.
# np.quaternion(w, x, y, z) or quaternion.as_quat_array([...])
Quaternion = np.quaternion


# --- Time Handling ---

@dataclass(frozen=True)  # Making Epoch immutable by default
class Epoch:
    """
    Represents a specific point in time, primarily as Ephemeris Time (ET) seconds past J2000.
    Provides convenience methods for conversion to/from UTC if a SpiceHandler is available.
    """
    et: float  # Ephemeris Time (TDB), seconds past J2000 epoch

    def __str__(self) -> str:
        return f"ET: {self.et}"

    def to_utc_string(self, spice_handler: Optional['SpiceHandler'] = None, time_format: str = "ISOC",
                      precision: int = 3) -> str:
        """
        Converts the Ephemeris Time to a UTC string.

        Args:
            spice_handler: An optional SpiceHandler instance. If None, this method will raise an error.
                           It's passed explicitly to avoid making SpiceHandler a global or singleton.
            time_format: The desired format of the output UTC string (see spiceypy.et2utc).
            precision: The number of decimal places for the seconds component.

        Returns:
            The UTC time string.

        Raises:
            ValueError: If spice_handler is not provided.
            spiceypy.utils.exceptions.SpiceyError: If SPICE encounters an error during conversion.
        """
        if spice_handler is None:
            raise ValueError("SpiceHandler instance must be provided to convert ET to UTC string.")
        return spice_handler.et_to_utc(self.et, time_format=time_format, precision=precision)

    @classmethod
    def from_utc_string(cls, utc_string: str, spice_handler: Optional['SpiceHandler'] = None) -> 'Epoch':
        """
        Creates an Epoch instance from a UTC string.

        Args:
            utc_string: UTC time string in a format recognizable by SPICE.
            spice_handler: An optional SpiceHandler instance. If None, this method will raise an error.

        Returns:
            A new Epoch instance.

        Raises:
            ValueError: If spice_handler is not provided.
            spiceypy.utils.exceptions.SpiceyError: If SPICE encounters an error during conversion.
        """
        if spice_handler is None:
            raise ValueError("SpiceHandler instance must be provided to convert UTC string to ET.")
        et = spice_handler.utc_to_et(utc_string)
        return cls(et)


# --- Placeholder Satellite Model Components ---
# These will be significantly expanded in Phase 1: Satellite Modeling Core.

@dataclass
class Facet:
    """
    Placeholder for a single reflective facet of a satellite component.
    This will eventually hold vertices, normal vector, area, material properties, etc.
    """
    id: str
    # Example placeholder attributes:
    # vertices: Optional[List[Vector3D]] = None
    # normal: Optional[Vector3D] = None
    # area: float = 0.0
    # material_properties: Optional[Any] = None # Link to BRDF parameters later
    pass  # Add more attributes as defined in FR1.1, FR1.4 etc. during Phase 1


@dataclass
class Component:
    """
    Placeholder for a distinct component of a satellite (e.g., bus, solar panel, antenna).
    This will hold its geometry (mesh reference, facets), position/orientation relative to
    the satellite body, material properties, and articulation rules.
    """
    id: str
    name: str
    # Example placeholder attributes:
    # mesh_path: Optional[str] = None # Path to an STL/OBJ file for detailed shadowing (FR1.1)
    # facets: List[Facet] = field(default_factory=list) # For light curve calculations (FR1.1)
    # relative_position: Vector3D = field(default_factory=lambda: np.array([0.0, 0.0, 0.0])) # (FR1.3)
    # relative_orientation: Quaternion = field(default_factory=lambda: np.quaternion(1, 0, 0, 0)) # (FR1.3)
    # articulation_rule: Optional[Any] = None # (FR1.3, FR3.4)
    pass  # Add more attributes as defined in FR1.2, FR1.3, FR1.4 etc. during Phase 1


@dataclass
class Satellite:
    """
    Placeholder for the overall satellite model.
    This will be a collection of components and define the satellite's body-fixed frame.
    """
    id: str
    name: str
    # components: List[Component] = field(default_factory=list) # (FR1.2)
    # body_frame_name: Optional[str] = None # Name of the primary body-fixed frame
    pass  # Add more attributes as defined in FR1 during Phase 1


if __name__ == '__main__':
    # Example usage (Epoch requires a SpiceHandler for UTC conversions, so this part is conceptual here)

    # Basic types
    pos_vec: Vector3D = np.array([1000.0, 2000.0, 3000.0])
    rot_mat: RotationMatrix = np.identity(3)
    quat_val: Quaternion = np.quaternion(0.7071, 0, 0.7071, 0)  # Example quaternion

    print(f"Position Vector: {pos_vec}")
    print(f"Rotation Matrix:\n{rot_mat}")
    print(f"Quaternion: {quat_val}")

    # Epoch (ET only, UTC conversion would need a live SpiceHandler)
    et_example = 634169191.0  # From your previous successful test
    epoch_example = Epoch(et=et_example)
    print(f"Epoch Example: {epoch_example}")

    # Conceptual UTC conversion (would fail without a real SpiceHandler instance)
    # from src.spice.spice_handler import SpiceHandler # Would need to adjust if running directly
    # try:
    #     # Assuming a SpiceHandler instance 'sh' is available and kernels are loaded
    #     # sh = SpiceHandler()
    #     # sh.load_metakernel("path/to/your/metakernel.tm") # Metakernel must load an LSK
    #     # utc_str = epoch_example.to_utc_string(sh)
    #     # print(f"Epoch as UTC: {utc_str}")
    #     # new_epoch = Epoch.from_utc_string("2020-02-05T10:05:21.815116Z", sh)
    #     # print(f"New Epoch from UTC: {new_epoch}")
    #     pass # Placeholder for when SpiceHandler is integrated
    # except Exception as e:
    #     print(f"Conceptual UTC conversion example error: {e}")

    # Placeholder model components
    facet1 = Facet(id="F001")
    component1 = Component(id="C001", name="SolarPanel_A")
    satellite1 = Satellite(id="S001", name="MySat")

    print(f"Facet: {facet1}")
    print(f"Component: {component1}")
    print(f"Satellite: {satellite1}")

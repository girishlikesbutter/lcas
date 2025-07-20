# lcforge/src/io/model_io.py

import yaml
import numpy as np
import quaternion as npq
from dataclasses import is_dataclass, fields
from typing import Optional
import logging
import os  # Added missing import

# Ensure models, core_types, and utils are importable
try:
    from src.models.model_definitions import Satellite, Component, Facet, BRDFMaterialProperties
    from src.core.common_types import Vector3D, Quaternion
    from src.utils.geometry_utils import generate_facets_from_conceptual_definitions
except ImportError as e:
    logging.basicConfig(level=logging.WARNING)
    logger_io = logging.getLogger(__name__)
    logger_io.warning(
        f"ImportError in model_io.py: {e}. Using relative imports or dummies. This might fail if not run in correct context.")
    # Attempt relative imports for cases where src is not directly on path but this file is part of a package
    from src.models.model_definitions import Satellite, Component, Facet, BRDFMaterialProperties
    from ..core.common_types import Vector3D, Quaternion
    from ..utils.geometry_utils import generate_facets_from_conceptual_definitions

logger = logging.getLogger(__name__)


# --- Custom YAML Representers with Explicit Tagging ---

def numpy_array_representer(dumper: yaml.Dumper, data: np.ndarray) -> yaml.Node:
    """Custom representer for numpy.ndarray that explicitly adds the tag."""
    # Using '!numpy.ndarray' as the tag.
    return dumper.represent_sequence('!numpy.ndarray', data.tolist())


def numpy_quaternion_representer(dumper: yaml.Dumper, data: npq.quaternion) -> yaml.Node:
    """Custom representer for numpy.quaternion that explicitly adds the tag."""
    # Using '!numpy.quaternion' as the tag.
    return dumper.represent_sequence('!numpy.quaternion', [data.w, data.x, data.y, data.z])


yaml.add_representer(np.ndarray, numpy_array_representer)
yaml.add_representer(npq.quaternion, numpy_quaternion_representer)


# --- Custom YAML Constructors ---

def numpy_array_constructor(loader: yaml.Loader, node: yaml.SequenceNode) -> np.ndarray:
    return np.array(loader.construct_sequence(node, deep=True))


def numpy_quaternion_constructor(loader: yaml.Loader, node: yaml.SequenceNode) -> npq.quaternion:
    components = loader.construct_sequence(node, deep=True)
    if len(components) != 4:
        raise yaml.YAMLError(f"Quaternion constructor requires 4 components (w,x,y,z), got {len(components)}")
    return npq.quaternion(components[0], components[1], components[2], components[3])


class ModelLoader(yaml.FullLoader):
    pass


# Constructors are registered for the explicit tags we now write.
ModelLoader.add_constructor('!numpy.ndarray', numpy_array_constructor)
ModelLoader.add_constructor('!numpy.quaternion', numpy_quaternion_constructor)

# Fallbacks for untagged or differently tagged common types (less likely to be hit if saving is consistent)
ModelLoader.add_constructor('tag:yaml.org,2002:python/object/apply:numpy.core.multiarray.scalar',
                            numpy_array_constructor)
ModelLoader.add_constructor('tag:yaml.org,2002:python/object/apply:numpy.array', numpy_array_constructor)
ModelLoader.add_constructor('tag:yaml.org,2002:python/object:numpy.quaternion', numpy_quaternion_constructor)


# --- Dataclass Representer ---
def satellite_model_representer(dumper: yaml.Dumper, data) -> yaml.Node:
    if is_dataclass(data) and not isinstance(data, type):
        tag = f"!{data.__class__.__module__}.{data.__class__.__name__}"
        data_for_yaml = {}
        for f_info in fields(data):
            field_name = f_info.name
            if isinstance(data, Component):
                if field_name == 'facets' or field_name == 'conceptual_faces_map':
                    continue
            data_for_yaml[field_name] = getattr(data, field_name)
        return dumper.represent_mapping(tag, data_for_yaml)
    return dumper.represent_undefined(data)


yaml.add_representer(Satellite, satellite_model_representer)
yaml.add_representer(Component, satellite_model_representer)
yaml.add_representer(Facet, satellite_model_representer)
yaml.add_representer(BRDFMaterialProperties, satellite_model_representer)

# --- Dataclass Constructor ---
_KNOWN_MODEL_CLASSES = {
    f"{cls.__module__}.{cls.__name__}": cls
    for cls in [Satellite, Component, Facet, BRDFMaterialProperties]
}


def model_dataclass_constructor(loader: yaml.Loader, tag_suffix: str, node: yaml.MappingNode) -> object:
    cls_name_full = tag_suffix.lstrip('!')
    cls = _KNOWN_MODEL_CLASSES.get(cls_name_full)

    if cls is None:
        raise yaml.YAMLError(f"Unknown model class for tag: !{cls_name_full}")

    data_dict = loader.construct_mapping(node, deep=True)

    if cls is Component:
        data_dict.pop('facets', None)
        data_dict.pop('conceptual_faces_map', None)

    try:
        instance = cls(**data_dict)
        if cls is Component and not hasattr(instance, 'conceptual_face_definitions'):
            setattr(instance, 'conceptual_face_definitions', {})
        return instance
    except TypeError as e:
        expected_fields = {f.name for f in fields(cls) if f.init}
        actual_keys = set(data_dict.keys())
        missing_fields = expected_fields - actual_keys
        extra_fields = actual_keys - expected_fields

        logger.error(f"TypeError instantiating {cls.__name__}. Error: {e}")
        logger.error(f"  Data provided: {data_dict}")
        logger.error(f"  Expected fields (init=True): {expected_fields}")
        logger.error(f"  Actual keys in YAML data: {actual_keys}")
        if missing_fields: logger.error(f"  Missing fields in YAML data: {missing_fields}")
        if extra_fields: logger.error(f"  Extra fields in YAML data: {extra_fields}")
        raise yaml.YAMLError(f"Failed to instantiate {cls.__name__}. Error: {e}")


for full_cls_name, cls_obj in _KNOWN_MODEL_CLASSES.items():
    ModelLoader.add_constructor(f"!{full_cls_name}",
                                lambda l, n, c=cls_obj: model_dataclass_constructor(l, f"!{c.__module__}.{c.__name__}",
                                                                                    n))


# --- Main Save and Load Functions ---

def save_satellite_to_yaml(satellite: Satellite, file_path: str) -> None:
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            yaml.dump(satellite, f, sort_keys=False, Dumper=yaml.Dumper)
        logger.info(f"Satellite model saved to {file_path}")
    except Exception as e:
        logger.error(f"Error saving satellite model to {file_path}: {e}", exc_info=True)
        raise


def load_satellite_from_yaml(file_path: str, facet_subdivision_level: int = 0) -> Optional[Satellite]:
    logger.info(f"Loading satellite model from: {file_path} with subdivision level: {facet_subdivision_level}")
    try:
        with open(file_path, 'r') as f:
            data = yaml.load(f, Loader=ModelLoader)

        if not isinstance(data, Satellite):
            logger.error(f"Loaded data from {file_path} is not a Satellite instance (type: {type(data)}).")
            return None

        logger.info(f"Post-processing loaded satellite '{data.name}': Generating facets for components.")
        for comp in data.components:
            if not isinstance(comp, Component):
                logger.warning(
                    f"Item in satellite.components is not a Component instance (type: {type(comp)}). Skipping facet generation for this item.")
                continue

            logger.debug(f"  Generating facets for component: {comp.name}")
            if not hasattr(comp, 'conceptual_face_definitions') or comp.conceptual_face_definitions is None:
                logger.warning(
                    f"    Component '{comp.name}' is missing 'conceptual_face_definitions' or it's None. Initializing to empty.")
                comp.conceptual_face_definitions = {}

            if comp.conceptual_face_definitions and comp.default_material is not None:
                generated_facets, generated_map = generate_facets_from_conceptual_definitions(
                    conceptual_defs=comp.conceptual_face_definitions,
                    default_material=comp.default_material,
                    comp_name_for_id_prefix=comp.id,
                    subdivision_level=facet_subdivision_level
                )
                comp.facets = generated_facets
                comp.conceptual_faces_map = generated_map
                logger.debug(
                    f"    Generated {len(comp.facets)} facets for '{comp.name}'. Map size: {len(comp.conceptual_faces_map)}")
            elif not comp.conceptual_face_definitions:
                logger.warning(f"    Component '{comp.name}' has no conceptual_face_definitions. Facets will be empty.")
                comp.facets = []
                comp.conceptual_faces_map = {}
            elif comp.default_material is None:
                logger.warning(
                    f"    Component '{comp.name}' has no default_material. Facets cannot be generated. Facets will be empty.")
                comp.facets = []
                comp.conceptual_faces_map = {}

        logger.info(f"Satellite model '{data.name}' loaded and processed successfully from {file_path}")
        return data

    except FileNotFoundError:
        logger.error(f"Error: YAML file not found at {file_path}")
        return None
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file {file_path}: {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading {file_path}: {e}", exc_info=True)
        return None


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main_logger = logging.getLogger(__name__)

    bus_material_ex = BRDFMaterialProperties(r_d=0.4, r_s=0.1, n_phong=5.0)
    panel_material_ex = BRDFMaterialProperties(r_d=0.2, r_s=0.6, n_phong=20.0)

    bus_comp_ex = Component(
        id="BUS_MAIN_IO", name="MainBus_IO",
        conceptual_face_definitions={
            "X_Plus_Face_IO": [[0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [0.5, 0.5, 0.5], [0.5, -0.5, 0.5]]},
        relative_position=np.array([0.0, 0.0, 0.0]),
        relative_orientation=npq.quaternion(1, 0, 0, 0),
        default_material=bus_material_ex
    )
    solar_panel_A_ex = Component(
        id="SP_A_IO_TEST_V3", name="SolarPanel_Port_IO_V3",
        conceptual_face_definitions={"Front_V3": [[0.05, -1, -2], [0.05, 1, -2], [0.05, 1, 2], [0.05, -1, 2]]},
        relative_position=np.array([0.0, 1.5, 0.0]),
        relative_orientation=npq.quaternion(0.9659258, 0.0, 0.258819, 0.0),  # Example non-identity quaternion
        articulation_rule="TRACK_SUN_Y_AXIS_V3",
        default_material=panel_material_ex
    )
    my_satellite_ex = Satellite(
        id="SAT001_IO_TEST_V3", name="MyResearchSatForIO_V3",
        components=[bus_comp_ex, solar_panel_A_ex],
        body_frame_name="MYSAT_IO_BODY_FIXED_V3"
    )

    test_yaml_file = "test_satellite_model_io_v3_tagged.yaml"  # New filename to ensure fresh save
    main_logger.info(f"\n--- Saving satellite to {test_yaml_file} (V3 Tagged Test) ---")
    save_satellite_to_yaml(my_satellite_ex, test_yaml_file)

    main_logger.info(f"\n--- Loading satellite from {test_yaml_file} with subdivision_level=1 (V3 Tagged Test) ---")
    loaded_satellite = load_satellite_from_yaml(test_yaml_file, facet_subdivision_level=1)

    if loaded_satellite:
        main_logger.info("\n--- Verification of Loaded Satellite (V3 Tagged) ---")
        assert loaded_satellite.id == my_satellite_ex.id
        assert len(loaded_satellite.components) == len(my_satellite_ex.components)

        for comp_idx, comp_loaded in enumerate(loaded_satellite.components):
            assert isinstance(comp_loaded,
                              Component), f"Loaded component {comp_idx} is not a Component instance, but {type(comp_loaded)}"
            comp_original = my_satellite_ex.components[comp_idx]
            assert isinstance(comp_loaded.relative_position,
                              np.ndarray), f"Comp {comp_loaded.name} rel_pos is not ndarray"
            assert isinstance(comp_loaded.relative_orientation,
                              npq.quaternion), f"Comp {comp_loaded.name} rel_orient is not quaternion"
            assert np.array_equal(comp_loaded.relative_position, comp_original.relative_position)
            assert comp_loaded.relative_orientation == comp_original.relative_orientation

        loaded_bus = next((c for c in loaded_satellite.components if c.name == "MainBus_IO"), None)
        assert loaded_bus is not None
        expected_bus_facets = 2 * (4 ** 1)
        assert len(loaded_bus.facets) == expected_bus_facets
        assert isinstance(loaded_bus.relative_orientation, npq.quaternion)  # Specific check

        loaded_panel = next((c for c in loaded_satellite.components if c.name == "SolarPanel_Port_IO_V3"), None)
        assert loaded_panel is not None
        expected_panel_facets = 2 * (4 ** 1)
        assert len(loaded_panel.facets) == expected_panel_facets
        assert isinstance(loaded_panel.relative_orientation, npq.quaternion)  # Specific check
        main_logger.info(
            f"Panel rel_orient type: {type(loaded_panel.relative_orientation)}, value: {loaded_panel.relative_orientation}")

        main_logger.info("\nBasic verification passed for V3 Tagged (dynamic facet generation & type checks).")
    else:
        main_logger.error("Failed to load and verify V3 Tagged satellite.")

    if os.path.exists(test_yaml_file):
        os.remove(test_yaml_file)
        main_logger.info(f"\nCleaned up {test_yaml_file}")

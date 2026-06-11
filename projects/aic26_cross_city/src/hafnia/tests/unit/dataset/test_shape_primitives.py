from typing import Type

import numpy as np
import pytest
import yaml

from hafnia.dataset.dataset_details_uploader import DatasetImageMetadata
from hafnia.dataset.dataset_names import PrimitiveField, SampleField
from hafnia.dataset.hafnia_dataset_types import Sample
from hafnia.dataset.primitives import PRIMITIVE_TYPES, Bbox, Bitmask, Classification, Polygon
from hafnia.dataset.primitives.primitive import Primitive
from tests import helper_testing


def get_initialized_dummy_primitives_using_default_task_name(TypePrimitive: Type[Primitive]) -> Primitive:
    """
    Returns a list of initialized dummy primitives for testing.
    """
    if TypePrimitive == Classification:
        return Classification(class_name="dummy_classification", label="dummy_label")
    elif TypePrimitive == Bbox:
        return Bbox(top_left_x=0.1, top_left_y=0.2, width=0.3, height=0.4, class_name="dummy_bbox")
    elif TypePrimitive == Polygon:
        return Polygon.from_list_of_points(points=[(0.1, 0.2), (0.3, 0.4), (0.5, 0.6)], class_name="dummy_polygon")
    elif TypePrimitive == Bitmask:
        return Bitmask.from_mask(
            mask=np.array([[True, False], [False, True]], dtype=bool),
            top=1,
            left=1,
            class_name="dummy_bitmask",
        )
    else:
        raise ValueError(f"Unsupported primitive type: {TypePrimitive}")


def assert_bbox_is_close(actual: Bbox, expected: Bbox, atol: float = 0.001):
    atol = 0.001
    assert type(expected) is type(actual)
    assert np.isclose(actual.height, expected.height, atol=atol)
    assert np.isclose(actual.width, expected.width, atol=atol)
    assert np.isclose(actual.top_left_x, expected.top_left_x, atol=atol)
    assert np.isclose(actual.top_left_y, expected.top_left_y, atol=atol)


@pytest.mark.parametrize("TypePrimitive", PRIMITIVE_TYPES)
def test_sample_primitive_names(TypePrimitive: Type[Primitive]):
    sample = Sample(file_path="test_image.jpg", width=100, height=100, split="test_split")

    for expected_field in PrimitiveField.fields():
        assert expected_field in TypePrimitive.model_fields, (
            f"Expected field '{expected_field}' not found in {{TypePrimitive.__name__}} annotations."
        )

    msgs = (
        f"Naming mismatch for coordinate type '{TypePrimitive.__name__}'. "
        f"The column name defined in 'column_name() -> '{TypePrimitive.column_name()}' ' \n"
        f"does not match an attribute in the '{Sample.__name__}' class. Change either 'Sample' or 'column_name()' to match."
    )
    assert hasattr(sample, TypePrimitive.column_name()), msgs

    primitive = get_initialized_dummy_primitives_using_default_task_name(
        TypePrimitive
    )  # Ensure that the dummy primitive can be initialized without errors
    msg = (
        f"The `task_name` of the initialized primitive doesn't match the default task name "
        f"specified in '{TypePrimitive.__name__}.default_task_name()'. Likely, the `model_post_init` of '{Primitive.__name__}', "
        f"have been overridden in '{TypePrimitive.__name__}'. "
    )
    assert primitive.task_name == TypePrimitive.default_task_name(), msg


def test_dataset_image_metadata_schema():
    yaml_str = yaml.dump(DatasetImageMetadata.model_json_schema())

    path_annotations_schema = helper_testing.get_path_test_data() / "dataset_image_metadata_schema.yaml"

    if not path_annotations_schema.exists():
        path_annotations_schema.write_text(yaml_str)
        assert not path_annotations_schema.exists(), (
            f"Expected {path_annotations_schema} to not exist. It has been recreated. Rerun test"
        )

    expected_yaml_str = path_annotations_schema.read_text()
    assert yaml.safe_load(yaml_str) == yaml.safe_load(expected_yaml_str), (
        f"Schema has changed. Delete the file {path_annotations_schema}\nand rerun the test to regenerate. "
        "IMPORTANT: This schema is used in the frontend to parse dataset metadata. "
        "Notify the front-end team of changes to the schema."
    )


def test_dataset_image_metadata_serialization():
    sample = helper_testing.get_sample_micro_hafnia_dataset("micro-tiny-dataset")
    dataset_image_metadata = DatasetImageMetadata.from_sample(sample)

    metadata_dict = dataset_image_metadata.model_dump(exclude_none=True)

    assert "annotations" in metadata_dict
    annotations = metadata_dict["annotations"]
    assert "bboxes" in annotations
    assert "classifications" in annotations

    assert "meta" in metadata_dict
    meta = metadata_dict["meta"]

    assert SampleField.FILE_PATH in meta

import pytest

from hafnia.dataset.dataset_names import CameraInfoField, OrientationField, PositionField, SampleField, VideoInfoField
from hafnia.dataset.hafnia_dataset_types import CameraInfo, Orientation, Position, Sample, VideoInfo


@pytest.mark.parametrize(
    "field_class, model_class",
    [
        (SampleField, Sample),
        (VideoInfoField, VideoInfo),
        (CameraInfoField, CameraInfo),
        (PositionField, Position),
        (OrientationField, Orientation),
    ],
)
def test_field_class_matches_pydantic_model(field_class, model_class):
    field_variable_names = list(field_class.__annotations__)
    model_fields = model_class.model_fields.keys()
    for variable_name in field_variable_names:
        field_value = getattr(field_class, variable_name)
        assert field_value in model_fields, (
            f"Field name '{field_value}' defined in '{field_class.__name__}.{variable_name}' "
            f"not found in '{model_class.__name__}' fields. Available fields are: {list(model_fields)}"
        )

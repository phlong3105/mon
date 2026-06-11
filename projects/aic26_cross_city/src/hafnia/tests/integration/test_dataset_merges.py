import pytest

from hafnia.dataset.dataset_names import OPS_REMOVE_CLASS, SampleField
from hafnia.dataset.dataset_recipe.dataset_recipe import DatasetRecipe
from hafnia.dataset.hafnia_dataset import HafniaDataset
from hafnia.utils import is_hafnia_configured
from tests.helper_testing_datasets import DATASET_SPEC_COCO_2017, DATASET_SPEC_MIDWEST


def test_merge_midwest_and_coco_datasets():
    if not is_hafnia_configured():
        pytest.skip("Hafnia platform not configured. Skipping CLI integration test.")

    force_download = False
    mappings_coco = {
        "person": "Person",
        "bicycle": "Vehicle",
        "car": "Vehicle",
        "motorcycle": "Vehicle",
        "bus": "Vehicle",
        "train": "Vehicle",
        "truck": "Vehicle",
    }
    mapping_midwest = {
        "Person": "Person",
        "Vehicle.*": "Vehicle",
        "Vehicle.Trailer": OPS_REMOVE_CLASS,
    }
    coco_specs = DATASET_SPEC_COCO_2017
    midwest_specs = DATASET_SPEC_MIDWEST

    coco = HafniaDataset.from_name(coco_specs.name, version=coco_specs.version, force_redownload=force_download)
    coco_remapped = coco.class_mapper(
        class_mapping=mappings_coco, method="remove_undefined", task_name="object_detection"
    )

    midwest = HafniaDataset.from_name(
        midwest_specs.name, version=midwest_specs.version, force_redownload=force_download
    )
    midwest_remapped = midwest.class_mapper(class_mapping=mapping_midwest, task_name="object_detection")
    merged_dataset = HafniaDataset.merge(midwest_remapped, coco_remapped)
    merged_dataset.check_dataset()

    # Recreate as recipe
    dataset_recipe = DatasetRecipe.from_merger(
        recipes=[
            DatasetRecipe.from_name(name=midwest_specs.name, version=midwest_specs.version).class_mapper(
                class_mapping=mapping_midwest, task_name="object_detection"
            ),
            DatasetRecipe.from_name(name=coco_specs.name, version=coco_specs.version).class_mapper(
                class_mapping=mappings_coco, method="remove_undefined", task_name="object_detection"
            ),
        ]
    )

    dataset_from_recipe = dataset_recipe.build()
    dataset_from_recipe.check_dataset()

    # Ensure dataset names are
    expected_dataset_names = {coco_specs.name, midwest_specs.name}
    actual_dataset_names = set(merged_dataset.samples[SampleField.DATASET_NAME].unique())
    assert actual_dataset_names == expected_dataset_names, (
        f"The '{SampleField.DATASET_NAME}' should contain the original dataset names {expected_dataset_names}. "
        f"But found: {actual_dataset_names}"
    )

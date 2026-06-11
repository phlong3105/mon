import pytest

from hafnia.dataset.dataset_helpers import is_valid_version_string
from hafnia.dataset.dataset_recipe.dataset_recipe import DatasetRecipe
from hafnia.dataset.hafnia_dataset import HafniaDataset
from tests.helper_testing_datasets import DATASET_SPEC_MNIST


def test_from_name_versioning_failures():
    dataset_spec = DATASET_SPEC_MNIST

    with pytest.raises(ValueError, match="Version must be specified"):
        HafniaDataset.from_name(dataset_spec.name)

    with pytest.raises(ValueError, match="Invalid version string"):
        HafniaDataset.from_name(dataset_spec.name, version="invalid_version_string")

    non_existing_version = "999.0.0"
    assert non_existing_version != dataset_spec.version
    with pytest.raises(ValueError, match=f"Selected version '{non_existing_version}' not found in available versions"):
        HafniaDataset.from_name(dataset_spec.name, version=non_existing_version)

    HafniaDataset.from_name(dataset_spec.name, version="latest")  # Should pass


def test_from_name_and_version_str():
    # Test with name and version
    recipe = DatasetRecipe.from_name_and_version_string(f"mnist:{DATASET_SPEC_MNIST.version}")
    assert isinstance(recipe, DatasetRecipe)
    assert recipe.creation.name == "mnist"
    assert recipe.creation.version == DATASET_SPEC_MNIST.version

    # Test with name only and allow_missing_version=True
    recipe = DatasetRecipe.from_name_and_version_string("mnist", resolve_missing_version=True)
    assert isinstance(recipe, DatasetRecipe)
    assert recipe.creation.name == "mnist"
    assert is_valid_version_string(recipe.creation.version), (
        "No version was defined. The latest version is being used. Resulting e.g. '1.0.0'"
    )

    # Test with name only and allow_missing_version=False
    with pytest.raises(ValueError):
        DatasetRecipe.from_name_and_version_string("mnist", resolve_missing_version=False)

    # Test with invalid type
    with pytest.raises(TypeError):
        DatasetRecipe.from_name_and_version_string(123)

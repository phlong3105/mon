import inspect
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from hafnia.dataset import primitives
from hafnia.dataset.dataset_recipe.dataset_recipe import (
    DatasetRecipe,
    FromMerger,
    get_dataset_path_from_recipe,
)
from hafnia.dataset.dataset_recipe.recipe_types import RecipeCreation, RecipeTransform, Serializable
from hafnia.dataset.hafnia_dataset import HafniaDataset
from hafnia.utils import is_hafnia_configured, pascal_to_snake_case
from tests.helper_testing import (
    annotation_as_string,
    get_dummy_recipe,
    get_strict_class_mapping_mnist,
)
from tests.helper_testing_datasets import DATASET_SPEC_MNIST


def check_signature(cls):
    sig = inspect.signature(cls.get_function())
    params = [param for param in sig.parameters.values()]
    HafniaDatasetName = HafniaDataset.__name__
    function_name = cls.get_function().__name__
    function_str = f"{function_name}{sig}".replace(f"'{HafniaDatasetName}'", HafniaDatasetName)
    error_msg_base = f"Class definition of '{cls.__name__}' does not match the specified function '{function_str}'"

    if issubclass(cls, RecipeTransform):
        params.pop(0)

    for param in params:
        is_hafnia_dataset_parameter = annotation_as_string(param.annotation) == annotation_as_string(HafniaDataset)
        if is_hafnia_dataset_parameter:
            # Specific handling of dataset/recipe parameters
            # A function with a dataset parameter, should have a recipe parameter in the class definition.
            # function(dataset: HafniaDataset) -> Class(recipe: DatasetRecipe)
            # function(dataset_some_more: HafniaDataset) -> Class(dataset_some_more: DatasetRecipe)

            # This also covers other argument names: dataset0 -> recipe0
            param_name = param.name.replace("dataset", "recipe")
            if param_name not in cls.model_fields:
                error_msg = (
                    f"The argument '{param}' for the '{function_name}()' function is missing in the definition of '{cls.__name__}'.\n"
                    f"Action: Add '{param_name}: {DatasetRecipe.__name__}' to 'class {cls.__name__}'."
                )
                raise ValueError("\n".join([error_msg_base, error_msg]))

            param_type = annotation_as_string(cls.model_fields[param_name].annotation)
            has_recipe_type = param_type == DatasetRecipe.__name__
            if not has_recipe_type:
                error_msg = (
                    f"The argument '{param_name}' of the'{function_name}()' function is of type '{param_type}'"
                    f", but it should be '{DatasetRecipe.__name__}' in '{cls.__name__}'.\n"
                    f"Action: Change '{param_name}: {param_type}' to "
                    f"'{param_name}: {DatasetRecipe.__name__}' in 'class {cls.__name__}'."
                )
                raise TypeError("\n".join([error_msg_base, error_msg]))

            continue

        if param.name not in cls.model_fields:
            error_msg = (
                f"The argument '{param}' for the '{function_name}()' function is missing in the definition of '{cls.__name__}'.\n"
                f"Action: Add '{param}'  to 'class {cls.__name__}'."
            )
            raise ValueError("\n".join([error_msg_base, error_msg]))
        model_field = cls.model_fields[param.name]
        model_field_type = annotation_as_string(model_field.annotation)
        function_param_type = annotation_as_string(param.annotation)
        if model_field_type != function_param_type:
            if "." in model_field_type:
                # Hard to handle case so we will simply ignore it for now.
                continue

            error_msg = (
                f"Type mismatch for parameter '{param.name}': expected {model_field_type=}, got {function_param_type=}."
            )
            raise TypeError("\n".join([error_msg_base, error_msg]))


@pytest.mark.parametrize("recipe_transform", Serializable.get_nested_subclasses())
def test_cases_serializable_functions_check_signature(recipe_transform: Serializable):
    """
    RecipeTransform converts a function into a serializable class.
    It ensures that the function signature is the same as the expected model fields.
    """
    # if not issubclass(recipe_transform, Serializable):
    skip_list = (RecipeCreation, RecipeTransform, DatasetRecipe, FromMerger)
    if recipe_transform in skip_list:
        pytest.skip(f"Skipping {recipe_transform.__name__}. Not applicable for checking signature")
    check_signature(recipe_transform)  # Ensures that function signatures match the expected model fields


@pytest.mark.parametrize("recipe_transform", Serializable.get_nested_subclasses())
def test_cases_check_dataset_transformations_have_builders(recipe_transform: Serializable):
    """
    Ensure that all dataset transformations have a corresponding RecipeTransform.
    """
    skip_list = (RecipeCreation, RecipeTransform, DatasetRecipe)
    if recipe_transform in skip_list:
        pytest.skip(f"Skipping {recipe_transform.__name__} - not expected to have a function in HafniaDataset")

    function_name_pascal_case = pascal_to_snake_case(recipe_transform.__name__)

    hafnia_dataset_has_function = hasattr(HafniaDataset, function_name_pascal_case)
    assert hafnia_dataset_has_function, (
        f"We expect all '{RecipeCreation.__name__}' and '{RecipeTransform.__name__}' to have a corresponding function "
        f"in '{HafniaDataset.__name__}'. \nThis test found that '{HafniaDataset.__name__}' is missing a function "
        f"called '{function_name_pascal_case}' to match the '{recipe_transform.__name__}' recipe object. "
        f"Please add '{function_name_pascal_case}' to '{HafniaDataset.__name__}'."
    )

    dataset_recipe_has_function = hasattr(DatasetRecipe, function_name_pascal_case)
    assert dataset_recipe_has_function, (
        f"We expect all '{RecipeCreation.__name__}' and '{RecipeTransform.__name__}' to have a corresponding function "
        f"in '{DatasetRecipe.__name__}'. \nThis test found that '{DatasetRecipe.__name__}' is missing a function "
        f"called '{function_name_pascal_case}' to match the '{recipe_transform.__name__}' recipe object. "
        f"Please add '{function_name_pascal_case}' to '{DatasetRecipe.__name__}'."
    )


def test_dataset_recipe_serialization_deserialization_json():
    """
    Test that Serializable can be serialized and deserialized correctly.
    """
    dataset_recipe = get_dummy_recipe()

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_file:
        path_json = Path(temp_file.name)

        # Serialize the dataset recipe to JSON
        dataset_recipe.as_json_file(path_json=path_json)

        # Deserialize from JSON path
        deserialized_recipe = DatasetRecipe.from_json_file(path_json=path_json)

    assert isinstance(deserialized_recipe, DatasetRecipe)  # type: ignore[misc]
    assert deserialized_recipe == dataset_recipe, "Deserialized recipe does not match original recipe"


@dataclass
class IntegrationTestUseCase:
    """
    This dataclass is used to define the integration test use case for dataset recipes.
    It is used to parameterize the test_recipes_integration_tests function.
    """

    recipe: Any
    short_name: str


TEST_CASES = [
    IntegrationTestUseCase(
        recipe=DatasetRecipe.from_name(name="mnist", version=DATASET_SPEC_MNIST.version, force_redownload=False),
        short_name="mnist",
    ),
    IntegrationTestUseCase(
        recipe=DatasetRecipe.from_path(path_folder=Path(".data/datasets/mnist"), check_for_images=False),
        short_name="'.data-datasets-mnist'",
    ),
    IntegrationTestUseCase(
        recipe=DatasetRecipe.from_merger(
            recipes=[
                DatasetRecipe.from_name(name="mnist", version=DATASET_SPEC_MNIST.version, force_redownload=False),
                DatasetRecipe.from_path(path_folder=Path(".data/datasets/mnist"), check_for_images=False),
            ]
        ),
        short_name="Merger(mnist,'.data-datasets-mnist')",
    ),
    IntegrationTestUseCase(
        recipe=DatasetRecipe.from_merge(
            recipe0=DatasetRecipe.from_path(path_folder=Path(".data/datasets/mnist"), check_for_images=False),
            recipe1=DatasetRecipe.from_name(name="mnist", version=DATASET_SPEC_MNIST.version, force_redownload=False),
        ),
        short_name="Merger('.data-datasets-mnist',mnist)",
    ),
    IntegrationTestUseCase(
        recipe=DatasetRecipe.from_name(name="mnist", version=DATASET_SPEC_MNIST.version, force_redownload=False)
        .select_samples(n_samples=20, shuffle=True, seed=42)
        .shuffle(seed=123),
        short_name="Recipe(mnist,SelectSamples,Shuffle)",
    ),
    IntegrationTestUseCase(
        recipe=DatasetRecipe.from_name(name="mnist", version=DATASET_SPEC_MNIST.version)
        .select_samples(n_samples=30)
        .splits_by_ratios(split_ratios={"train": 0.8, "test": 0.2})
        .split_into_multiple_splits(split_name="test", split_ratios={"val": 0.5, "test": 0.5})
        .define_sample_set_by_size(n_samples=10),
        short_name="Recipe(mnist,SelectSamples,SplitsByRatios,SplitIntoMultipleSplits,DefineSampleSetBySize)",
    ),
    IntegrationTestUseCase(
        recipe=DatasetRecipe.from_name(name="mnist", version=DATASET_SPEC_MNIST.version)
        .class_mapper(
            get_strict_class_mapping_mnist(),
        )
        .rename_task(
            old_task_name=primitives.Classification.default_task_name(),
            new_task_name="digits",
        )
        .select_samples_by_class_name(name="odd"),
        short_name="Recipe(mnist,ClassMapper,RenameTask,SelectSamplesByClassName)",
    ),
]


# ids: Use the name of the test case as the ID for clarity
@pytest.mark.parametrize("test_case", TEST_CASES, ids=lambda test_case: test_case.short_name)
def test_cases_integration_tests(test_case: IntegrationTestUseCase):
    """
    Test that LoadDataset recipe can be created and serialized.
    Even if it can be considered an integration test, we keep it in the unit-test folder
    as it doesn't require any external dependencies and runs fast.
    """

    if not is_hafnia_configured():
        pytest.skip("Hafnia is not configured, skipping build dataset recipe tests.")

    # Smoke test: Convert to short name
    short_name_str = test_case.recipe.as_short_name()
    assert "/" not in short_name_str, f"Short name '{short_name_str}' should not contain '/'."
    assert "\\" not in short_name_str, f"Short name '{short_name_str}' should not contain '\\'."
    assert short_name_str == test_case.short_name, (
        f"Short name '{short_name_str}' does not match expected '{test_case.short_name}'"
    )

    # Smoke test: Convert to code representation
    code_one_liner_str = test_case.recipe.as_python_code()
    assert isinstance(code_one_liner_str, str), "Code representation of dataset recipe is not a string"

    # Smoke test: Convert to JSON representation and back
    json_str = test_case.recipe.as_json_str()
    dataset_recipe_again = DatasetRecipe.from_json_str(json_str=json_str)
    assert dataset_recipe_again == test_case.recipe, "Deserialized recipe does not match original recipe"

    # Build the dataset from the recipe
    dataset: HafniaDataset = test_case.recipe.build()

    # Smoke test: Able to store the dataset to disk
    with tempfile.TemporaryDirectory() as temp_dir:
        path_datasets = Path(temp_dir) / "datasets"

        path_dataset = get_dataset_path_from_recipe(test_case.recipe, path_datasets=path_datasets)

        # To make sure that the dataset path is not nested in the datasets path
        # A dataset path is created from the 'as_short_name()' method. If 'as_short_name()'
        # contain '/' or '\\', it will create a nested folder structure.
        assert path_dataset.parent == path_datasets

        assert not path_dataset.exists(), f"Dataset path shouldn't exist before building: {path_dataset} "
        dataset_again = HafniaDataset.from_recipe_with_cache(test_case.recipe, path_datasets=path_datasets)  # noqa: F841
        assert path_dataset.exists(), f"Dataset path should exist after building: {path_dataset} "
        dataset_again = HafniaDataset.from_recipe_with_cache(test_case.recipe, path_datasets=path_datasets)  # noqa: F841

    assert isinstance(dataset, HafniaDataset), "Dataset is not an instance of HafniaDataset"

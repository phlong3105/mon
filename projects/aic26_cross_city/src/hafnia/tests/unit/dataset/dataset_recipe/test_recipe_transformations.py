import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Type

import polars as pl
import pytest

from hafnia.dataset.dataset_names import TAG_IS_SAMPLE, SampleField
from hafnia.dataset.dataset_recipe.dataset_recipe import DatasetRecipe, FromName
from hafnia.dataset.dataset_recipe.recipe_transforms import (
    ClassMapper,
    DefineSampleSetBySize,
    DropSamplesByClassName,
    RenameTask,
    SelectSamples,
    SelectSamplesByClassName,
    Shuffle,
    SplitIntoMultipleSplits,
    SplitsByRatios,
)
from hafnia.dataset.dataset_recipe.recipe_types import RecipeTransform
from hafnia.dataset.hafnia_dataset import HafniaDataset
from hafnia.dataset.primitives import Bbox
from tests.helper_testing import dict_as_list_of_tuples, get_micro_hafnia_dataset, get_strict_class_mapping_tiny_dataset


@dataclass
class TestCaseRecipeTransform:
    recipe_transform: RecipeTransform
    as_python_code: str
    short_name: str

    def as_dataset_recipe(self) -> DatasetRecipe:
        return DatasetRecipe(creation=FromName(name="test"), operations=[self.recipe_transform])


def get_test_cases() -> list[TestCaseRecipeTransform]:
    return [
        TestCaseRecipeTransform(
            recipe_transform=SelectSamples(n_samples=10, shuffle=True, seed=42),
            as_python_code="select_samples(n_samples=10, shuffle=True, seed=42, with_replacement=False)",
            short_name="SelectSamples",
        ),
        TestCaseRecipeTransform(
            recipe_transform=Shuffle(seed=123),
            as_python_code="shuffle(seed=123)",
            short_name="Shuffle",
        ),
        TestCaseRecipeTransform(
            recipe_transform=SplitsByRatios(split_ratios={"train": 0.5, "val": 0.25, "test": 0.25}, seed=42),
            as_python_code="splits_by_ratios(split_ratios={'train': 0.5, 'val': 0.25, 'test': 0.25}, seed=42)",
            short_name="SplitsByRatios",
        ),
        TestCaseRecipeTransform(
            recipe_transform=DefineSampleSetBySize(n_samples=100),
            as_python_code="define_sample_set_by_size(n_samples=100, seed=42)",
            short_name="DefineSampleSetBySize",
        ),
        TestCaseRecipeTransform(
            recipe_transform=SplitIntoMultipleSplits(split_name="test", split_ratios={"test": 0.5, "val": 0.5}),
            as_python_code="split_into_multiple_splits(split_name='test', split_ratios={'test': 0.5, 'val': 0.5})",
            short_name="SplitIntoMultipleSplits",
        ),
        TestCaseRecipeTransform(
            recipe_transform=ClassMapper(
                class_mapping=get_strict_class_mapping_tiny_dataset(),
                method="strict",
                primitive=None,
                task_name=None,
            ),
            as_python_code=f"class_mapper(class_mapping={dict_as_list_of_tuples(get_strict_class_mapping_tiny_dataset())}, method='strict', primitive=None, task_name=None)",  # noqa: E501
            short_name="ClassMapper",
        ),
        TestCaseRecipeTransform(
            recipe_transform=RenameTask(old_task_name="old_name", new_task_name="new_name"),
            as_python_code="rename_task(old_task_name='old_name', new_task_name='new_name')",
            short_name="RenameTask",
        ),
        TestCaseRecipeTransform(
            recipe_transform=SelectSamplesByClassName(name=["Person"], task_name=None, primitive=None),
            as_python_code="select_samples_by_class_name(name=['Person'], task_name=None, primitive=None)",
            short_name="SelectSamplesByClassName",
        ),
        TestCaseRecipeTransform(
            recipe_transform=DropSamplesByClassName(
                name=["Person"], task_name=None, primitive=None, drop_classes_from_task_info=False
            ),
            as_python_code="drop_samples_by_class_name(name=['Person'], task_name=None, primitive=None, drop_classes_from_task_info=False)",  # noqa: E501
            short_name="DropSamplesByClassName",
        ),
    ]


@pytest.mark.parametrize("recipe_transform", set(RecipeTransform.get_nested_subclasses()))
def test_check_if_recipe_transform_is_missing_a_test_case(recipe_transform: Type[RecipeTransform]):
    """
    Ensure that all recipe transformations are tested.
    This is useful to ensure that new transformations are added to the test suite.
    """
    in_test_recipe_transforms = {tc.recipe_transform.__class__ for tc in get_test_cases()}

    recipe_transform_missing_a_test_case = recipe_transform not in in_test_recipe_transforms

    if recipe_transform_missing_a_test_case:
        error_msg = (
            f"We expect all recipe transformations to have a test case, but found '{RecipeTransform.__name__}' "
            f"classes/subclasses that are not tested. \nPlease add '{recipe_transform.__name__}' as a "
            f"'{TestCaseRecipeTransform.__name__}' in the list of test cases found in '{get_test_cases.__name__}()' "
            "to ensure they are tested."
        )

        raise AssertionError(error_msg)


@pytest.mark.parametrize("test_case", get_test_cases(), ids=lambda tc: tc.as_python_code)
def test_cases_serialization_deserialization_of_recipe_transform(test_case: TestCaseRecipeTransform):
    dataset_recipe = test_case.as_dataset_recipe()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp_file:
        path_json = Path(tmp_file.name)
        dataset_recipe.as_json_file(path_json)
        dataset_recipe_again = DatasetRecipe.from_json_file(path_json)

    assert dataset_recipe_again == dataset_recipe


@pytest.mark.parametrize("test_case", get_test_cases(), ids=lambda tc: tc.as_python_code)
def test_cases_as_python_code(test_case: TestCaseRecipeTransform):
    """
    Test that the `as_python_code` method of the recipe transformation returns the expected string representation.
    """
    code_str = test_case.recipe_transform.as_python_code(keep_default_fields=True, as_kwargs=True)

    assert code_str == test_case.as_python_code


@pytest.mark.parametrize("test_case", get_test_cases(), ids=lambda tc: tc.as_python_code)
def test_cases_as_short_name(test_case: TestCaseRecipeTransform):
    """
    Test that the `as_short_name` method of the recipe transformation returns the expected string representation.
    """
    short_name = test_case.recipe_transform.as_short_name()

    assert short_name == test_case.short_name


def test_sample_transformation():
    dataset: HafniaDataset = get_micro_hafnia_dataset(dataset_name="micro-tiny-dataset")  # type: ignore[annotation-unchecked]
    n_samples = 2
    sample_transformation = SelectSamples(n_samples=n_samples, shuffle=True, seed=42)

    new_dataset = sample_transformation.build(dataset)
    assert isinstance(new_dataset, HafniaDataset), "Sampled dataset is not a HafniaDataset instance"
    assert len(new_dataset) == n_samples, (
        f"Sampled dataset length {len(new_dataset)} does not match expected {n_samples}"
    )


def test_sample_transformation_without_replacement():
    """Without replacement, the number of samples should not exceed the actual dataset size."""
    dataset: HafniaDataset = get_micro_hafnia_dataset(dataset_name="micro-tiny-dataset")  # type: ignore[annotation-unchecked]

    max_actual_number_of_samples = len(dataset)
    n_samples = 100  # A micro dataset is only 3 samples, so this should be capped to 3
    sample_transformation = SelectSamples(n_samples=n_samples, shuffle=True, seed=42, with_replacement=False)

    new_dataset = sample_transformation.build(dataset)
    assert isinstance(new_dataset, HafniaDataset), "Sampled dataset is not a HafniaDataset instance"
    assert len(new_dataset) == max_actual_number_of_samples, (
        f"Sampled dataset length {len(new_dataset)} does not match expected {max_actual_number_of_samples}"
    )


def test_sample_transformation_with_replacement():
    """With replacement, the number of samples can exceed the actual dataset size."""
    dataset: HafniaDataset = get_micro_hafnia_dataset(dataset_name="micro-tiny-dataset")  # type: ignore[annotation-unchecked]

    assert len(dataset) < 100, "The micro dataset should have less than 100 samples for this test to be valid"
    n_samples = 100  # The micro dataset is only 3 samples. With_replacement=True it will duplicate samples
    sample_transformation = SelectSamples(n_samples=n_samples, shuffle=True, seed=42, with_replacement=True)

    new_dataset = sample_transformation.build(dataset)
    assert isinstance(new_dataset, HafniaDataset), "Sampled dataset is not a HafniaDataset instance"
    assert len(new_dataset) == n_samples, (
        f"Sampled dataset length {len(new_dataset)} does not match expected {n_samples}"
    )


def test_class_mapper_preserve_order():
    """
    Class mapping can be specified using either a dict or a list of tuples. When using a dict,
    the order of the classes is not guaranteed to be preserved when serializing to json.

    To handle this, the class mapping is always converted to a list of tuples during validation.
    This test ensures that the order is preserved when using a dict as input.

    And that the class mapping can be serialized and deserialized correctly.
    """
    class_mapping = {
        "Person": "Person",
        "Vehicle*": "Car",
    }

    class_mapper = ClassMapper(class_mapping=class_mapping, method="remove_undefined", primitive=Bbox, task_name=None)

    # Ensure that the class mapping is serialized as a list of tuples to preserve order
    assert isinstance(class_mapper.class_mapping, list), (
        "Class mapping should be serialized as a list to preserve order"
    )
    assert class_mapper.class_mapping == list(class_mapping.items()), (
        "Class mapping should be serialized as a list of tuples to preserve order"
    )
    # Recreate the class mapper from the dumped model to ensure serialization/deserialization works
    class_mapper_recreated = ClassMapper(**class_mapper.model_dump())

    # Apply the class mapper to the dataset and ensure the result is the same
    assert class_mapper_recreated.class_mapping == class_mapper.class_mapping, (
        "Class mapping should be the same after serialization/deserialization"
    )


def test_shuffle_transformation():
    dataset: HafniaDataset = get_micro_hafnia_dataset(dataset_name="micro-tiny-dataset")  # type: ignore[annotation-unchecked]

    shuffle_transformation = Shuffle(seed=123)
    new_dataset = shuffle_transformation.build(dataset)

    shuffle_transformation = Shuffle(seed=123)
    new_dataset2 = shuffle_transformation.build(dataset)
    is_same = all(new_dataset.samples[SampleField.SAMPLE_INDEX] == new_dataset2.samples[SampleField.SAMPLE_INDEX])
    assert is_same, "Shuffled datasets should be equal with the same seed"

    is_same = all(new_dataset.samples[SampleField.FILE_PATH] == dataset.samples[SampleField.FILE_PATH])
    assert not is_same, "Shuffled dataset should not match original dataset"
    assert isinstance(new_dataset, HafniaDataset), "Shuffled dataset is not a HafniaDataset instance"
    assert len(new_dataset) == len(dataset), (
        f"Shuffled dataset length {len(new_dataset)} does not match original {len(dataset)}"
    )


def test_splits_by_ratios_transformation():
    dataset: HafniaDataset = get_micro_hafnia_dataset(dataset_name="micro-tiny-dataset")  # type: ignore[annotation-unchecked]
    # The micro dataset is small, so we duplicate samples up to 100 samples for testing
    dataset = dataset.select_samples(n_samples=100, seed=42, with_replacement=True)

    split_ratios = {"train": 0.5, "val": 0.25, "test": 0.25}
    splits_transformation = SplitsByRatios(split_ratios=split_ratios, seed=42)
    new_dataset = splits_transformation.build(dataset)
    assert isinstance(new_dataset, HafniaDataset), "Splits dataset is not a HafniaDataset instance"

    actual_split_counts = new_dataset.calculate_split_counts()
    expected_split_count = {name: int(ratio * len(dataset)) for name, ratio in split_ratios.items()}
    assert actual_split_counts == expected_split_count


def test_define_sample_by_size_transformation():
    n_samples = 5
    dataset: HafniaDataset = get_micro_hafnia_dataset(dataset_name="micro-tiny-dataset")  # type: ignore[annotation-unchecked]
    # The micro dataset is small, so we duplicate samples up to 100 samples for testing
    dataset = dataset.select_samples(n_samples=n_samples, seed=42, with_replacement=True)

    define_sample_transformation = DefineSampleSetBySize(n_samples=n_samples)
    new_dataset = define_sample_transformation.build(dataset)
    assert isinstance(new_dataset, HafniaDataset), "Sampled dataset is not a HafniaDataset instance"

    n_samples_with_tag = (
        new_dataset.samples[SampleField.TAGS].list.eval(pl.element().filter(pl.element() == TAG_IS_SAMPLE)).list.len()
        > 0
    ).sum()
    assert n_samples_with_tag == n_samples, (
        f"Sampled dataset should have {n_samples} samples, but has {new_dataset.samples[SampleField.TAGS].sum()}"
    )


def test_split_into_multiple_splits():
    n_samples = 100
    dataset: HafniaDataset = get_micro_hafnia_dataset(dataset_name="micro-tiny-dataset")  # type: ignore[annotation-unchecked]
    # The micro dataset is small, so we duplicate samples up to 100 samples for testing
    dataset = dataset.select_samples(n_samples=n_samples, seed=42, with_replacement=True)
    dataset = dataset.splits_by_ratios(split_ratios={"train": 0.5, "test": 0.5}, seed=42)

    divide_split_name = "test"
    split_ratios = {"test": 0.5, "val": 0.5}

    # Create a test split in the dataset

    split_transformation = SplitIntoMultipleSplits(split_name=divide_split_name, split_ratios=split_ratios)
    new_dataset = split_transformation.build(dataset)

    expected_split_counts = {"train": int(0.5 * n_samples), "test": int(0.25 * n_samples), "val": int(0.25 * n_samples)}
    actual_split_counts = new_dataset.calculate_split_counts()
    assert isinstance(new_dataset, HafniaDataset), "New dataset is not a HafniaDataset instance"
    assert actual_split_counts == expected_split_counts, (
        f"Expected split counts {expected_split_counts}, but got {actual_split_counts}"
    )

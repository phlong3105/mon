from pathlib import Path
from typing import Callable

import numpy as np
import pytest

from hafnia.dataset.dataset_names import PrimitiveField, SampleField, StorageFormat
from hafnia.dataset.hafnia_dataset import HafniaDataset
from hafnia.dataset.hafnia_dataset_types import Sample, TaskInfo
from hafnia.dataset.operations.dataset_transformations import (
    expand_class_mapping,
    get_task_info_from_task_name_and_primitive,
)
from hafnia.dataset.primitives import Bbox, Classification
from tests.helper_testing import (
    get_path_micro_hafnia_dataset,
    get_strict_class_mapping_tiny_dataset,
    simulate_hafnia_video_dataset,
)


def test_class_mapper_strict():
    dataset_name = "micro-tiny-dataset"
    path_dataset = get_path_micro_hafnia_dataset(dataset_name=dataset_name, force_update=False)
    dataset = HafniaDataset.from_path(path_dataset)

    dataset_updated = dataset.class_mapper(
        class_mapping=get_strict_class_mapping_tiny_dataset(),
        method="strict",
        primitive=Bbox,
    )

    dataset_updated.check_dataset_tasks()
    task_bbox = dataset_updated.info.get_task_by_primitive(Bbox)
    expected_class_names = ["person", "vehicle", "truck"]
    expected_indices = list(range(len(expected_class_names)))
    assert task_bbox.get_class_names() == expected_class_names
    bboxes = dataset_updated.samples[task_bbox.primitive.column_name()].explode().struct.unnest()
    assert set(bboxes[PrimitiveField.CLASS_NAME]).issubset(expected_class_names)
    assert set(bboxes[PrimitiveField.CLASS_IDX]).issubset(set(expected_indices))
    assert not dataset.samples.equals(dataset_updated.samples), (
        "Samples should be different after class mapping. Verify that original 'dataset.samples' is not mutated."
    )
    assert dataset.info != dataset_updated.info, (
        "Info should be different after class mapping. Verify that original 'dataset.info' is not mutated."
    )


def test_class_mapper_strict_wildcard_mapping():
    dataset_name = "micro-tiny-dataset"
    path_dataset = get_path_micro_hafnia_dataset(dataset_name=dataset_name, force_update=False)
    dataset = HafniaDataset.from_path(path_dataset)

    class_mapping = {
        "Person": "person",
        "Vehicle*": "vehicle",
    }
    dataset_updated = dataset.class_mapper(
        class_mapping=class_mapping,
        method="strict",
        primitive=Bbox,
    )

    dataset_updated.check_dataset_tasks()
    task_bbox = dataset_updated.info.get_task_by_primitive(Bbox)
    expected_class_names = ["person", "vehicle"]
    assert task_bbox.get_class_names() == expected_class_names


def test_class_mapper_remove_undefined():
    dataset_name = "micro-tiny-dataset"
    path_dataset = get_path_micro_hafnia_dataset(dataset_name=dataset_name, force_update=False)
    dataset = HafniaDataset.from_path(path_dataset)

    # Use case 1: Valid class mapping with 'remove_undefined' method
    class_mapping = {
        "Vehicle.Car": "vehicle",
    }
    dataset_updated = dataset.class_mapper(
        class_mapping=class_mapping,
        method="remove_undefined",
        primitive=Bbox,
    )

    dataset_updated.check_dataset_tasks()
    task_bbox = dataset_updated.info.get_task_by_primitive(Bbox)
    expected_class_names = ["vehicle"]
    assert task_bbox.get_class_names() == expected_class_names


def test_class_mapper_exceptions():
    dataset_name = "micro-tiny-dataset"
    path_dataset = get_path_micro_hafnia_dataset(dataset_name=dataset_name, force_update=False)
    dataset = HafniaDataset.from_path(path_dataset)

    class_mapping_bad = {
        "NonExistingClass": "vehicle",
    }

    # Test case 1: Invalid method name
    with pytest.raises(ValueError, match="Method .* is not recognized"):
        dataset.class_mapper(
            class_mapping=class_mapping_bad,
            method="WrongMethodName",
            primitive=Bbox,
        )

    # Test case 2: Class mapping with non-existing class names
    with pytest.raises(ValueError, match="The specified class mapping contains class names .* that do not exist"):
        dataset.class_mapper(
            class_mapping=class_mapping_bad,
            method="remove_undefined",
            primitive=Bbox,
        )


def test_class_mapper_keep_undefined():
    dataset_name = "micro-tiny-dataset"
    path_dataset = get_path_micro_hafnia_dataset(dataset_name=dataset_name, force_update=False)
    dataset = HafniaDataset.from_path(path_dataset)

    task_bbox = dataset.info.get_task_by_primitive(Bbox)
    class_names_original = task_bbox.get_class_names() or []
    rename_class = "Vehicle.Car"
    class_mapping = {
        rename_class: "vehicle",
    }

    dataset_updated = dataset.class_mapper(
        class_mapping=class_mapping,
        method="keep_undefined",
        primitive=Bbox,
    )

    dataset_updated.check_dataset_tasks()
    task_bbox = dataset_updated.info.get_task_by_primitive(Bbox)

    class_names_original.remove(rename_class)
    expected_class_names = ["vehicle"]
    expected_class_names.extend(class_names_original)
    assert task_bbox.get_class_names() == expected_class_names


def test_expand_class_mapping():
    # Task class names
    class_names = [
        "Vehicle",
        "Vehicle.Bicycle",
        "Vehicle.Car.SUV",
        "Vehicle.Car.Sedan",
        "Vehicle.Truck.Small",
        "Vehicle.Truck.Large",
        "Person",
        "Person.Adult",
        "Person.Child",
        "Animal.Dog",
    ]
    # Wildcard mapping to be expanded
    class_mapping_with_wildcard = {
        "Person": "person",
        "Vehicle*": "vehicle",
        "Vehicle.Car*": "car",
        "Vehicle.Truck*": "truck",
        "Person*": "person",
        "Animal*": "animal",
    }

    expanded = expand_class_mapping(wildcard_mapping=class_mapping_with_wildcard, class_names=class_names)

    # Expected expanded mapping
    expected_mapping = {
        "Person": "person",  # Exact match takes precedence
        "Vehicle": "vehicle",
        "Vehicle.Bicycle": "vehicle",
        "Vehicle.Car.SUV": "car",  # More specific wildcard takes precedence
        "Vehicle.Car.Sedan": "car",
        "Vehicle.Truck.Small": "truck",
        "Vehicle.Truck.Large": "truck",
        "Person.Adult": "person",
        "Person.Child": "person",
        "Animal.Dog": "animal",
    }
    assert len(expanded) == len(class_names)
    assert expanded == expected_mapping


def test_rename_task():
    dataset_name = "micro-tiny-dataset"
    path_dataset = get_path_micro_hafnia_dataset(dataset_name=dataset_name, force_update=False)
    dataset = HafniaDataset.from_path(path_dataset)

    old_task_name = Bbox.default_task_name()
    new_task_name = "renamed_bboxes"
    dataset_renamed = dataset.rename_task(old_task_name=old_task_name, new_task_name=new_task_name)

    dataset_renamed.check_dataset_tasks()

    task_bbox = [t for t in dataset_renamed.info.tasks if t.primitive == Bbox][0]
    assert task_bbox.name == new_task_name
    bboxes = dataset_renamed.samples[task_bbox.primitive.column_name()].explode().struct.unnest()
    assert PrimitiveField.TASK_NAME in bboxes.columns
    assert set(bboxes[PrimitiveField.TASK_NAME]) == {new_task_name}
    assert not dataset.samples.equals(dataset_renamed.samples), (
        "Samples should be different after renaming task. Verify that original 'dataset.samples' is not mutated."
    )
    assert dataset.info != dataset_renamed.info, (
        "Info should be different after renaming task. Verify that original 'dataset.info' is not mutated."
    )


def test_merge_datasets():
    dataset_name = "micro-tiny-dataset"
    path_dataset = get_path_micro_hafnia_dataset(dataset_name=dataset_name, force_update=False)
    dataset = HafniaDataset.from_path(path_dataset)

    dataset_0 = dataset.select_samples(n_samples=2, seed=42)
    dataset_1 = dataset.select_samples(n_samples=2, seed=43)

    # Use case 1: Merging two datasets with the same tasks and class names
    dataset_merged = HafniaDataset.from_merge(dataset0=dataset_0, dataset1=dataset_1)
    assert len(dataset_merged.samples) == 4, f"Expected 4 samples, got {len(dataset_merged.samples)}"
    assert len(dataset_merged.info.tasks) == len(dataset_0.info.tasks), "Tasks should be preserved after merging"
    assert len(dataset_merged.info.tasks) == len(dataset_1.info.tasks), "Tasks should be preserved after merging"

    # Use case 2: Merging two datasets with the same tasks but different class names should raise an error
    mapping = get_strict_class_mapping_tiny_dataset()
    dataset_1_changed = dataset_1.class_mapper(
        class_mapping=mapping,
        primitive=Bbox,
    )
    with pytest.raises(ValueError, match="Cannot merge datasets with different class names for the same task name"):
        HafniaDataset.from_merge(dataset0=dataset_0, dataset1=dataset_1_changed)


def test_select_samples_by_class_name():
    dataset_name = "micro-tiny-dataset"
    path_dataset = get_path_micro_hafnia_dataset(dataset_name=dataset_name, force_update=False)
    dataset = HafniaDataset.from_path(path_dataset)

    # Use case 1: Select samples by class name (task_name and primitive are auto-inferred)
    dataset_updated = dataset.select_samples_by_class_name(name="Vehicle.Car")
    assert len(dataset_updated) < len(dataset), "Expected fewer samples after filtering by class name"

    # Use case 2: Select samples by class name with explicit 'primitive'
    dataset_updated = dataset.select_samples_by_class_name(name="Vehicle.Car", primitive=Bbox)
    assert len(dataset_updated) < len(dataset), "Expected fewer samples after filtering by class name"

    # Use case 3: Wrong class name (not found in any task)
    with pytest.raises(ValueError, match="The specified names"):
        dataset_updated = dataset.select_samples_by_class_name(name="NonExistingClass", primitive=Bbox)

    # Use case 4: Class exists but not in the specified task
    with pytest.raises(ValueError, match="The specified names"):
        dataset_updated = dataset.select_samples_by_class_name(name="Vehicle.Car", task_name="Weather")


def test_drop_samples_by_class_name():
    dataset_name = "micro-tiny-dataset"
    path_dataset = get_path_micro_hafnia_dataset(dataset_name=dataset_name, force_update=False)
    dataset = HafniaDataset.from_path(path_dataset)

    selected = dataset.select_samples_by_class_name(name="Vehicle.Car")
    dropped = dataset.drop_samples_by_class_name(name="Vehicle.Car")
    assert len(dropped) == len(dataset) - len(selected), (
        "Expected drop and select to be complementary on the same class name"
    )

    bbox_task = dropped.info.get_task_by_primitive(Bbox)
    bboxes = dropped.samples[bbox_task.primitive.column_name()].explode().struct.unnest()
    bbox_names = bboxes.filter(bboxes[PrimitiveField.TASK_NAME] == bbox_task.name)[PrimitiveField.CLASS_NAME]
    assert "Vehicle.Car" not in set(bbox_names), "Dropped class should not appear in any remaining annotation"

    # drop_classes_from_task_info=True removes the class from TaskInfo and re-indexes
    dropped_with_info = dataset.drop_samples_by_class_name(name="Vehicle.Car", drop_classes_from_task_info=True)
    new_bbox_task = dropped_with_info.info.get_task_by_primitive(Bbox)
    assert "Vehicle.Car" not in (new_bbox_task.get_class_names() or [])
    new_class_names = new_bbox_task.get_class_names() or []
    new_bboxes = dropped_with_info.samples[new_bbox_task.primitive.column_name()].explode().struct.unnest()
    new_bboxes = new_bboxes.filter(new_bboxes[PrimitiveField.TASK_NAME] == new_bbox_task.name)
    for cls_name, cls_idx in zip(new_bboxes[PrimitiveField.CLASS_NAME], new_bboxes[PrimitiveField.CLASS_IDX]):
        assert new_class_names.index(cls_name) == cls_idx, "Class indices should match the new task class ordering"

    # Wrong class name (not found in any task)
    with pytest.raises(ValueError, match="The specified names"):
        dataset.drop_samples_by_class_name(name="NonExistingClass", primitive=Bbox)


def test_get_task_info_from_task_name_and_primitive():
    task_class = TaskInfo.from_class_names(primitive="Classification", class_names=["cat", "dog"])
    task_bbox = TaskInfo.from_class_names(primitive="Bbox", class_names=["car", "bus"])

    ### Test case: No tasks defined
    with pytest.raises(ValueError, match="Dataset has no tasks defined."):
        get_task_info_from_task_name_and_primitive(tasks=[])

    ### Test case: Task name and primitive is not required when there is only one task
    get_task_info_from_task_name_and_primitive(tasks=[task_class])

    ### Test case: Task name is required when there are multiple tasks
    with pytest.raises(ValueError, match="For multiple tasks"):
        get_task_info_from_task_name_and_primitive(tasks=[task_class, task_bbox])

    ### Test case: Task name is found
    task_actual = get_task_info_from_task_name_and_primitive(tasks=[task_class, task_bbox], task_name=task_class.name)
    assert task_actual == task_class

    ### Test case: Task name is not found
    with pytest.raises(ValueError, match="No task found with task_name="):
        get_task_info_from_task_name_and_primitive(tasks=[task_class, task_bbox], task_name="segmentation")

    ### Test case: Non-unique task name
    task_bbox_bad = task_bbox.model_copy()
    task_bbox_bad.name = task_class.name
    with pytest.raises(ValueError, match="Found multiple tasks with task_name="):
        get_task_info_from_task_name_and_primitive(tasks=[task_class, task_bbox_bad], task_name=task_class.name)

    ### Test case: Primitive is found by string
    task_actual = get_task_info_from_task_name_and_primitive(tasks=[task_class, task_bbox], primitive="Classification")
    assert task_actual == task_class

    ### Test case: Primitive is found by type
    task_actual = get_task_info_from_task_name_and_primitive(tasks=[task_class, task_bbox], primitive=Classification)
    assert task_actual == task_class

    ### Test case: Primitive is not found
    with pytest.raises(ValueError, match="No task found with primitive="):
        get_task_info_from_task_name_and_primitive(tasks=[task_class, task_bbox], primitive="Polygon")

    ### Test case: Non-unique Primitive
    task_bbox_bad = task_bbox.model_copy()
    task_bbox_bad.primitive = task_class.primitive
    with pytest.raises(ValueError, match="Found multiple tasks with primitive="):
        get_task_info_from_task_name_and_primitive(tasks=[task_class, task_bbox_bad], primitive=task_class.primitive)

    ### Test case: Both task name and primitive are provided and match
    task_actual = get_task_info_from_task_name_and_primitive(
        tasks=[task_class, task_bbox], task_name=task_class.name, primitive=task_class.primitive
    )
    assert task_actual == task_class

    ### Test case: Both task name and primitive are provided but do not match
    with pytest.raises(ValueError, match="No task found with task_name="):
        get_task_info_from_task_name_and_primitive(
            tasks=[task_class, task_bbox], task_name=task_class.name, primitive=task_bbox.primitive
        )

    ### Test case: Both task name and primitive are provided but do not match
    with pytest.raises(ValueError, match="This should never happen"):
        get_task_info_from_task_name_and_primitive(
            tasks=[task_class, task_class], task_name=task_class.name, primitive=task_class.primitive
        )


def test_video_storage_format_read_image(tmp_path: Path, compare_to_expected_image: Callable) -> None:
    hafnia_dataset = simulate_hafnia_video_dataset(path_dataset=tmp_path, n_frames=10, fps=1)
    for sample_dict in hafnia_dataset:
        sample = Sample(**sample_dict)
        assert sample.storage_format == StorageFormat.VIDEO
        sample.read_image()
        image_annotations = sample.draw_annotations()
        break

    compare_to_expected_image(image_annotations)


def test_convert_to_image_storage_format(tmp_path: Path, compare_to_expected_image: Callable) -> None:
    dataset_video: HafniaDataset = simulate_hafnia_video_dataset(path_dataset=tmp_path, n_frames=10, fps=1)
    path_image_based_dataset = tmp_path / "image_based_dataset"
    dataset_images: HafniaDataset = dataset_video.convert_to_image_storage_format(
        path_output_folder=path_image_based_dataset,
        reextract_frames=False,
    )

    path_image_based_dataset / "data"
    dataset_images.samples[SampleField.FILE_PATH].to_list()
    dataset_images.check_dataset(check_splits=False)


def test_drop_task_by_name():
    dataset_name = "micro-tiny-dataset"
    path_dataset = get_path_micro_hafnia_dataset(dataset_name=dataset_name, force_update=False)
    dataset = HafniaDataset.from_path(path_dataset)

    assert len(dataset.info.get_tasks_by_primitive(Bbox)) == 1, (
        "The test expect that the dataset should have one Bbox task before execution."
    )
    n_classification_tasks = len(dataset.info.get_tasks_by_primitive(Classification))
    assert n_classification_tasks > 1, (
        "The test expect that the dataset should have multiple Classification tasks before execution"
    )
    dataset_org = dataset.copy()  # to verify that original dataset is not mutated

    ## Use case 1: Drop task by name - only task with the primitive
    task_name_to_drop = Bbox.default_task_name()
    dataset_dropped = dataset.drop_task(task_name=task_name_to_drop)
    assert len(dataset_dropped.info.get_tasks_by_primitive(Bbox)) == 0, "Task should be dropped from dataset info"
    assert Bbox.column_name() not in dataset_dropped.samples.columns, "Task column should be dropped from samples"
    assert dataset == dataset_org, "Original dataset have been mutated after operation -- it should not."

    ## Use case 2: Drop primitive - only task with the primitive
    dataset_dropped = dataset.drop_primitive(primitive=Bbox)
    assert len(dataset_dropped.info.get_tasks_by_primitive(Bbox)) == 0, "Task should be dropped from dataset info"
    assert Bbox.column_name() not in dataset_dropped.samples.columns, "Task column should be dropped from samples"
    assert dataset == dataset_org, "Original dataset have been mutated after operation -- it should not."

    ## Use case 3: Drop task by name - multiple tasks with the same primitive
    task_name_to_drop = dataset.info.get_tasks_by_primitive(Classification)[0].name
    dataset_dropped = dataset.drop_task(task_name=task_name_to_drop)
    assert len(dataset_dropped.info.get_tasks_by_primitive(Classification)) == n_classification_tasks - 1, (
        "Only one task should be dropped from dataset info"
    )
    tasks = dataset_dropped.samples[Classification.column_name()].explode().struct.unnest()[PrimitiveField.TASK_NAME]
    assert task_name_to_drop not in set(tasks.unique())
    assert dataset == dataset_org, "Original dataset have been mutated after operation -- it should not."

    ## Use case 4: Drop primitive - multiple tasks with the same primitive
    dataset_dropped = dataset.drop_primitive(primitive=Classification)
    assert len(dataset_dropped.info.get_tasks_by_primitive(Classification)) == 0, (
        "All classification tasks should be dropped from dataset info"
    )
    assert Classification.column_name() not in dataset_dropped.samples.columns, (
        "Task column should be dropped from samples"
    )
    assert dataset == dataset_org, "Original dataset have been mutated after operation -- it should not."


def test_transform_image(tmp_path: Path, compare_to_expected_image: Callable) -> None:
    dataset_name = "micro-tiny-dataset"
    path_dataset = get_path_micro_hafnia_dataset(dataset_name=dataset_name, force_update=False)
    dataset = HafniaDataset.from_path(path_dataset)

    def image_transform(image, sample: Sample) -> np.ndarray:
        image[:10, :10] = 100  # Add black square to the top-left corner of the image
        return image

    dataset_transformed = dataset.transform_images(
        transform=image_transform,
        path_output=tmp_path / "transformed_images",
        use_hash=False,
        subfolder_name="transformed_data",
        select_columns=[
            SampleField.FILE_PATH,
            SampleField.COLLECTION_ID,
            SampleField.STORAGE_FORMAT,
            SampleField.HEIGHT,
            SampleField.WIDTH,
            SampleField.SPLIT,
        ],
    )

    sample = Sample(**dataset_transformed[0])
    image = sample.read_image()
    expected_sum = 100 * 10 * 10 * 1
    assert image[:10, :10, 0].sum() == expected_sum, (
        "The top-left corner of the image should be transformed to black (pixel value 0)"
    )

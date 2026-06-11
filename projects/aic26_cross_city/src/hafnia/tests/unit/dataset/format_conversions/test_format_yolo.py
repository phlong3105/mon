from pathlib import Path
from typing import Callable

import polars as pl
import pytest

from hafnia.dataset import primitives
from hafnia.dataset.dataset_names import SampleField
from hafnia.dataset.format_conversions import format_yolo
from hafnia.dataset.hafnia_dataset import HafniaDataset
from hafnia.dataset.hafnia_dataset_types import Sample
from tests import helper_testing
from tests.helper_testing import get_micro_hafnia_dataset, get_path_test_dataset_formats


def test_import_yolo_format_visualized(compare_to_expected_image: Callable) -> None:
    path_yolo_dataset = get_path_test_dataset_formats() / "format_yolo"
    hafnia_dataset = format_yolo.from_yolo_format(path_yolo_dataset)

    sample_name = "000000000632"
    samples = hafnia_dataset.samples.filter(pl.col(SampleField.FILE_PATH).str.contains(sample_name))
    assert len(samples) == 1, f"Expected to find one sample with name '{sample_name}'"
    sample = Sample(**samples.row(0, named=True))

    sample_visualized = sample.draw_annotations()
    compare_to_expected_image(sample_visualized)


def test_format_yolo_import_export_tiny_dataset(tmp_path: Path, compare_to_expected_image: Callable) -> None:
    dataset: HafniaDataset = get_micro_hafnia_dataset(dataset_name="micro-tiny-dataset")  # type: ignore[annotation-unchecked]

    path_yolo_dataset_exported = tmp_path / "exported_yolo_dataset"
    format_yolo.to_yolo_format(
        dataset=dataset,
        path_output=path_yolo_dataset_exported,
    )

    dataset_reloaded = format_yolo.from_yolo_format(path_yolo_dataset_exported)

    image_hash = "4d5817f0e44da708ce2db0262080f571"
    samples = dataset_reloaded.samples.filter(pl.col(SampleField.FILE_PATH).str.contains(image_hash))
    assert len(samples) == 1, f"Expected to find one sample with index 0, found {len(samples)}"
    sample = Sample(**samples.row(0, named=True))
    sample_visualized = sample.draw_annotations()
    compare_to_expected_image(sample_visualized)


@pytest.mark.parametrize("micro_dataset_name", helper_testing.MICRO_DATASETS)
def test_to_and_from_yolo_format(micro_dataset_name: str, tmp_path: Path) -> None:
    dataset = helper_testing.get_micro_hafnia_dataset(dataset_name=micro_dataset_name)
    n_expected_samples = len(dataset.samples)
    path_output = tmp_path / micro_dataset_name

    # To YOLO format
    dataset.to_yolo_format(path_output=path_output)

    # From YOLO format
    dataset_reloaded = HafniaDataset.from_yolo_format(path_dataset=path_output, dataset_name=micro_dataset_name)

    assert len(dataset_reloaded.samples) == n_expected_samples, (
        "The number of samples before and after COCO format conversion should be the same"
    )


def test_format_yolo_import_export(tmp_path: Path) -> None:
    path_yolo_dataset = get_path_test_dataset_formats() / "format_yolo"

    # Test case 1: Import YOLO dataset
    dataset = format_yolo.from_yolo_format(path_yolo_dataset)

    assert len(dataset) == 3
    assert len(dataset.info.tasks) == 1
    task = dataset.info.tasks[0]
    assert len(task.get_class_names() or []) == 80
    assert task.primitive == primitives.Bbox
    assert task.name == primitives.Bbox.default_task_name()

    # Test case 2: Export yolo dataset
    path_yolo_dataset_exported = tmp_path / "exported_yolo_dataset"
    list_split_paths = format_yolo.to_yolo_format(
        dataset=dataset,
        path_output=path_yolo_dataset_exported,
        task_name=None,
    )
    for split_paths in list_split_paths:
        split_paths.check_paths()
        split_class_names = [n for n in split_paths.path_class_names.read_text().splitlines() if n.strip() != ""]
        assert split_class_names == task.get_class_names(), (
            "The class names in the exported YOLO dataset should match the original dataset task class names."
        )
        image_paths = split_paths.path_images_txt.read_text().splitlines()
        assert len(image_paths) > 0
        for image_path in image_paths:
            path_image_full = split_paths.path_root / image_path
            assert path_image_full.exists()

    # Test case 3: Re-import exported YOLO dataset
    dataset_reimported = format_yolo.from_yolo_format(path_yolo_dataset_exported)

    assert len(dataset_reimported) == len(dataset)
    assert len(dataset_reimported.info.tasks) == len(dataset.info.tasks)
    actual_samples = dataset_reimported.samples.drop(SampleField.FILE_PATH)
    expected_samples = dataset.samples.drop(SampleField.FILE_PATH)
    assert actual_samples.equals(expected_samples)

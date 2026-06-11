from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from hafnia.dataset.dataset_details_uploader import (
    DatasetImageMetadata,
    create_gallery_images,
    dataset_details_from_hafnia_dataset,
)
from hafnia.dataset.dataset_names import SampleField
from hafnia.dataset.hafnia_dataset import HafniaDataset
from hafnia.dataset.hafnia_dataset_types import Sample
from hafnia.dataset.primitives import Classification
from tests import helper_testing


@pytest.mark.parametrize("dataset_name", helper_testing.MICRO_DATASETS)
def test_dataset_details_from_hafnia_dataset(dataset_name: str, tmp_path: Path):
    path_dataset = helper_testing.get_path_micro_hafnia_dataset(dataset_name=dataset_name, force_update=False)
    dataset = HafniaDataset.from_path(path_dataset)
    classification_tasks = dataset.info.get_tasks_by_primitive(Classification)
    distribution_task_names = [task.name or task.primitive.default_task_name() for task in classification_tasks]
    dataset_info = dataset_details_from_hafnia_dataset(
        dataset=dataset,
        path_gallery_images=tmp_path / "gallery_images",
        gallery_samples=dataset.samples.head(2),
        distribution_task_names=distribution_task_names,
    )
    # Check if dataset info can be serialized to JSON
    dataset_info_json = dataset_info.model_dump_json()  # noqa: F841

    # Basic checks on dataset info fields
    assert isinstance(dataset_info.dataset_updated_at, datetime)
    assert isinstance(dataset_info.version, str) and len(dataset_info.version) > 0
    assert isinstance(dataset_info.dataset_format_version, str) and len(dataset_info.dataset_format_version) > 0

    # Check dataset variants
    assert dataset_info.dataset_variants is not None
    assert len(dataset_info.dataset_variants) > 0

    # Check split annotations reports
    assert dataset_info.split_annotations_reports is not None
    assert len(dataset_info.split_annotations_reports) == 8
    full_split_annotations_report = [r for r in dataset_info.split_annotations_reports if r.split == "full"]
    assert len(full_split_annotations_report) == 2, "There should be exactly two 'full' split annotations report"
    full_report = full_split_annotations_report[0]

    # Check annotated object reports
    assert full_report.annotated_object_reports is not None
    assert len(full_report.annotated_object_reports) > 0
    expected_primitives = {t.primitive.__name__ for t in dataset.info.tasks}
    actual_primitives = {r.obj.annotation_type.name for r in full_report.annotated_object_reports}
    assert expected_primitives == actual_primitives

    # Check distribution values
    assert full_report.distribution_values is not None
    actual_dist_names = {d.distribution_category.distribution_type.name for d in full_report.distribution_values}
    assert actual_dist_names == set(distribution_task_names)


def test_dataset_details_extraction():
    path_dataset = helper_testing.get_path_micro_hafnia_dataset(dataset_name="micro-tiny-dataset", force_update=False)
    dataset = HafniaDataset.from_path(path_dataset)

    dataset_info = dataset_details_from_hafnia_dataset(dataset=dataset)
    assert dataset_info.data_captured_end is not None, "Expected data_captured_end to be extracted from dataset"
    assert dataset_info.data_captured_start is not None, "Expected data_captured_start to be extracted from dataset"
    assert dataset_info.data_received_end is not None, "Expected data_received_end to be extracted from dataset"
    assert dataset_info.data_received_start is not None, "Expected data_received_start to be extracted from dataset"

    assert len(dataset_info.dataset_variants) == 2, "Expected two dataset variants (train and validation/test)"
    hidden_variants = [v for v in dataset_info.dataset_variants if v.variant_type == "hidden"]
    assert len(hidden_variants) == 1, "Expected exactly one hidden variant"

    variant_hidden = hidden_variants[0]
    assert variant_hidden.n_cameras is not None, "Expected n_cameras to be extracted from dataset"
    assert variant_hidden.duration is not None, "Expected duration_seconds to be extracted from dataset"
    assert variant_hidden.duration_average is not None, "Expected duration_average to be extracted from dataset"
    assert variant_hidden.frame_rate is not None, "Expected frame_rate to be extracted from dataset"
    assert isinstance(variant_hidden.resolutions, list) and len(variant_hidden.resolutions) > 0, (
        "Expected resolutions to be a list"
    )


@pytest.mark.parametrize(
    "file_path, expected",
    [
        ("C:\\Users\\data\\images\\file.jpg", "data/images/file.jpg"),  # Windows absolute
        ("/opt/ml/input/data/training/file.jpg", "data/training/file.jpg"),  # Linux absolute
        ("data/subdir/file.jpg", "data/subdir/file.jpg"),  # Posix relative
    ],
    ids=["windows", "linux", "posix-relative"],
)
def test_dataset_image_metadata_normalizes_file_path(file_path: str, expected: str) -> None:
    """DatasetImageMetadata.from_sample must produce a forward-slash path of the last 3 parts."""
    sample = Sample(
        file_path=file_path,
        height=10,
        width=10,
        split="train",
        classifications=[Classification(class_name="A", class_idx=0)],
    )
    metadata = DatasetImageMetadata.from_sample(sample)
    assert metadata.meta is not None
    stored_path = metadata.meta[SampleField.FILE_PATH]
    assert "\\" not in stored_path, f"Backslash survived normalisation: {stored_path!r}"
    assert stored_path == expected, f"Unexpected path: {stored_path!r}"


@pytest.mark.parametrize(
    "file_path",
    [
        "C:\\Users\\data\\images\\file.jpg",  # Windows absolute
        "/opt/ml/input/data/training/file.jpg",  # Linux absolute
        "data/subdir/file.jpg",  # Posix relative
    ],
    ids=["windows", "linux", "posix-relative"],
)
def test_create_gallery_images_uses_filename_only(
    file_path: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Gallery images must be saved using only the filename regardless of path style."""
    monkeypatch.setattr(Sample, "draw_annotations", lambda _: np.zeros((10, 10, 3), dtype=np.uint8))

    samples = pl.DataFrame(
        {
            SampleField.FILE_PATH: [file_path],
            SampleField.HEIGHT: [10],
            SampleField.WIDTH: [10],
            SampleField.SPLIT: ["train"],
        }
    )

    path_gallery = tmp_path / "gallery"
    create_gallery_images(gallery_samples=samples, path_gallery_images=path_gallery)

    assert (path_gallery / "file.jpg").exists(), f"Gallery image not saved with correct filename for path {file_path!r}"

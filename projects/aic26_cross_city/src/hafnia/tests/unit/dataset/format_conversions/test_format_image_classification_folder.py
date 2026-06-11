from pathlib import Path

import pytest

from hafnia.dataset.dataset_names import SplitName
from hafnia.dataset.format_conversions.format_image_classification_folder import (
    from_image_classification_folder,
    to_image_classification_folder,
)
from hafnia.dataset.hafnia_dataset import HafniaDataset
from tests.helper_testing import get_micro_hafnia_dataset


def test_import_export_image_classification_from_directory(tmp_path: Path) -> None:
    dataset: HafniaDataset = get_micro_hafnia_dataset(dataset_name="micro-tiny-dataset")

    # Test fix. The 'tiny-dataset' has some duplicate samples with the same file path
    # when stored to disk this will reduce the number of samples from 6 to 3.
    dataset.samples = dataset.samples.unique(subset=["file_path"], keep="first")

    path_exported = tmp_path / "exported"
    with pytest.raises(ValueError, match="Found multiple tasks"):
        to_image_classification_folder(dataset, path_output=path_exported)

    task = dataset.info.get_task_by_task_name_and_primitive(task_name="Time of Day", primitive=None)
    exported_paths = to_image_classification_folder(
        dataset, path_output=path_exported, task_name=task.name, clean_folder=True
    )

    split_names = [p.name for p in exported_paths]
    assert set(split_names) == {SplitName.TRAIN, SplitName.VAL}
    hafnia_dataset_imported = from_image_classification_folder(
        path_folder=path_exported,
        n_samples=None,
    )

    assert len(hafnia_dataset_imported.info.tasks) == 1
    assert hafnia_dataset_imported.info.tasks[0].primitive == task.primitive
    assert len(hafnia_dataset_imported.samples) == len(dataset.samples)

    hafnia_dataset_imported = from_image_classification_folder(
        path_folder=path_exported,
        n_samples=2,
    )

    assert len(hafnia_dataset_imported.info.tasks) == 1
    assert hafnia_dataset_imported.info.tasks[0].primitive == task.primitive
    assert len(hafnia_dataset_imported.samples) == 2

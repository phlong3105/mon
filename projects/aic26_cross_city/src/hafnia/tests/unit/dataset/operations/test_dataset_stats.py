import pytest

import hafnia
from hafnia.dataset.hafnia_dataset import HafniaDataset
from hafnia.dataset.primitives import Bbox
from tests.helper_testing import (
    MICRO_DATASETS,
    get_micro_hafnia_dataset,
    get_path_micro_hafnia_dataset,
    get_path_workspace,
)


@pytest.mark.parametrize("micro_dataset_name", MICRO_DATASETS)
def test_micro_dataset_format_versions(micro_dataset_name: str):
    FORCE_UPDATE = False  # Use this flag to update the micro test datasets if needed
    path_dataset = get_path_micro_hafnia_dataset(dataset_name=micro_dataset_name, force_update=FORCE_UPDATE)
    path_dataset_relative = path_dataset.relative_to(get_path_workspace())
    dataset = HafniaDataset.from_path(path_dataset)
    format_version_match = dataset.info.format_version == hafnia.__dataset_format_version__
    assert format_version_match, (
        f"The micro test dataset '{micro_dataset_name}' (located in '{path_dataset_relative}') is outdated.\n"
        f"The format version for the dataset is '{dataset.info.format_version}', while the current dataset\n"
        f"format version for the hafnia package is  '{hafnia.__dataset_format_version__}'.\n"
        f"Please rerun this test but set 'FORCE_UPDATE = True' to update the micro test dataset."
    )


@pytest.mark.parametrize("micro_dataset_name", MICRO_DATASETS)
def test_class_counts_for_task(micro_dataset_name: str):
    dataset = get_micro_hafnia_dataset(dataset_name=micro_dataset_name, force_update=False)
    counts = dataset.calculate_task_class_counts(primitive=Bbox)
    bbox_task = dataset.info.get_task_by_primitive(Bbox)
    assert isinstance(counts, dict)
    assert len(counts) == len(bbox_task.get_class_names() or [])


@pytest.mark.parametrize("micro_dataset_name", MICRO_DATASETS)
def test_class_counts_all(micro_dataset_name: str):
    dataset = get_micro_hafnia_dataset(dataset_name=micro_dataset_name, force_update=False)
    counts = dataset.calculate_class_counts()
    assert isinstance(counts, list)
    expected_num_classes = sum(len(task.get_class_names() or []) for task in dataset.info.tasks)
    assert len(counts) == expected_num_classes


# NOTE: Below names matches the Dataset Database Model in the backend. Change only if backend fields are updated too
EXPECTED_VIDEO_STATS_NAMES = {
    "n_videos",
    "duration",
    "duration_average",
    "frame_rate",
    "data_received_start",
    "data_received_end",
    "data_captured_start",
    "data_captured_end",
    "n_cameras",
}


@pytest.mark.parametrize(
    "micro_dataset_name, n_stats",
    [
        ("micro-tiny-dataset", len(EXPECTED_VIDEO_STATS_NAMES)),
        ("micro-coco-2017", 0),
    ],
)
def test_calculate_video_stats(micro_dataset_name: str, n_stats: int):
    dataset = get_micro_hafnia_dataset(dataset_name=micro_dataset_name, force_update=False)
    video_stats = dataset.calculate_video_stats()
    assert isinstance(video_stats, dict)
    assert len(video_stats) == n_stats

    assert set(video_stats.keys()).issubset(EXPECTED_VIDEO_STATS_NAMES), (
        f"Unexpected video stats keys found in dataset '{micro_dataset_name}'.\n"
        f"Expected keys are a subset of: {EXPECTED_VIDEO_STATS_NAMES}\n"
        f"Actual keys found: {set(video_stats.keys())}"
    )


@pytest.mark.parametrize("micro_dataset_name", MICRO_DATASETS)
def test_print_basic_stats(micro_dataset_name: str):
    dataset = get_micro_hafnia_dataset(dataset_name=micro_dataset_name, force_update=False)
    dataset.print_basic_stats()


@pytest.mark.parametrize("micro_dataset_name", MICRO_DATASETS)
def test_print_stats(micro_dataset_name: str):
    dataset = get_micro_hafnia_dataset(dataset_name=micro_dataset_name, force_update=False)
    dataset.print_stats()


@pytest.mark.parametrize("micro_dataset_name", MICRO_DATASETS)
def test_print_sample_and_task_counts(micro_dataset_name: str):
    dataset = get_micro_hafnia_dataset(dataset_name=micro_dataset_name, force_update=False)
    dataset.print_sample_and_task_counts()


@pytest.mark.parametrize("micro_dataset_name", MICRO_DATASETS)
def test_print_class_distribution(micro_dataset_name: str):
    dataset = get_micro_hafnia_dataset(dataset_name=micro_dataset_name, force_update=False)
    dataset.print_class_distribution()

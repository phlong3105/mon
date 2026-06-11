from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type

import polars as pl
import rich
from rich import print as rprint
from rich.table import Table

from hafnia.dataset.dataset_names import PrimitiveField, SampleField, SplitName, VideoInfoField
from hafnia.dataset.hafnia_dataset_types import Sample
from hafnia.dataset.operations.table_transformations import create_primitive_table
from hafnia.dataset.primitives import PRIMITIVE_TYPES
from hafnia.log import user_logger
from hafnia.utils import progress_bar

if TYPE_CHECKING:  # Using 'TYPE_CHECKING' to avoid circular imports during type checking
    from hafnia.dataset.hafnia_dataset import HafniaDataset
    from hafnia.dataset.primitives.primitive import Primitive


def calculate_split_counts(dataset: HafniaDataset) -> Dict[str, int]:
    """Return the number of samples per split as a `{split_name: count}` mapping.

    Splits with zero samples are not included; only splits actually present in
    `dataset.samples["split"]` appear as keys.
    """
    return dict(dataset.samples[SampleField.SPLIT].value_counts().iter_rows())


def calculate_task_class_counts(
    dataset: HafniaDataset,
    primitive: Optional[Type[Primitive]] = None,
    task_name: Optional[str] = None,
) -> Dict[str, int]:
    """
    Determines class name counts for a specific task in the dataset.

    The counts are returned as a dictionary where keys are class names and values are their respective counts.
    Note that class names with zero counts are included in the dictionary and that
    the order of the dictionary matches the class idx.

    >>> counts = dataset.class_counts_for_task(primitive=Bbox)
    >>> print(counts)
    {
        'person': 0,  # Note: Zero count classes are included to maintain order with class idx
        'car': 500,
        'bicycle': 0,
        'bus': 100,
        'truck': 150,
        'motorcycle': 50,
    }
    """
    task = dataset.info.get_task_by_task_name_and_primitive(task_name=task_name, primitive=primitive)

    # Initialize counts with zero for all classes to ensure zero-count classes are included
    # and to have class names in the order of class idx
    class_counts = {name: 0 for name in (task.get_class_names() or [])}

    task_column_name = task.primitive.column_name()
    if task_column_name not in dataset.samples.columns:
        return class_counts

    class_counts_df = (
        dataset.samples[task_column_name]
        .explode()
        .struct.unnest()
        .filter(pl.col(PrimitiveField.TASK_NAME) == task.name)[PrimitiveField.CLASS_NAME]
        .value_counts()
    )

    class_counts.update(dict(class_counts_df.iter_rows()))

    return class_counts


def calculate_class_counts(dataset: HafniaDataset) -> List[Dict[str, Any]]:
    """
    Get class counts for all tasks in the dataset.
    The counts are returned as a dictionary where keys are in the format
    '{primitive_name}/{task_name}/{class_name}' and values are their respective counts.

    Example:
    >>> counts = dataset.class_counts_all()
    >>> print(counts)
    [
        {'Primitive': 'Bbox', 'Task Name': 'detection', 'Class Name': 'car', 'Count': 500},
        {'Primitive': 'Bbox', 'Task Name': 'detection', 'Class Name': 'bus', 'Count': 100},
        {'Primitive': 'Classification', 'Task Name': 'scene', 'Class Name': 'indoor', 'Count': 300},
        {'Primitive': 'Classification', 'Task Name': 'scene', 'Class Name': 'outdoor', 'Count': 700},
    ]
    """
    count_info = []
    for task in dataset.info.tasks:
        class_name_counts = dataset.calculate_task_class_counts(task_name=task.name)
        for name, counts in class_name_counts.items():
            count_info.append(
                {
                    "Primitive": task.primitive.__name__,
                    "Task Name": task.name,
                    "Class Name": name,
                    "Count": counts,
                }
            )
    return count_info


def calculate_primitive_counts(dataset: HafniaDataset) -> Dict[str, int]:
    """Count the total number of annotations per task across the whole dataset.

    Each task contributes one entry, keyed by the primitive class name (and task name if it differs
    from the primitive's default), with the value being the total number of annotations of that
    type in the dataset.
    """
    annotation_counts = {}
    for task in dataset.info.tasks:
        objects = dataset.create_primitive_table(task.primitive, task_name=task.name)
        name = task.primitive.__name__
        if task.name != task.primitive.default_task_name():
            name = f"{name}.{task.name}"
        n_objects = 0 if objects is None else len(objects)
        annotation_counts[name] = n_objects
    return annotation_counts


def calculate_split_counts_extended(dataset: HafniaDataset) -> List[Dict[str, Any]]:
    """Return per-split sample counts plus per-task primitive counts, one row per split.

    The returned list contains rows for "All", "Train", "Validation" and "Test", each with a
    sample count and counts for every primitive task in the dataset. Used by
    `print_sample_and_task_counts` to render the dataset overview table.
    """
    splits_sets = {
        "All": SplitName.valid_splits(),
        "Train": [SplitName.TRAIN],
        "Validation": [SplitName.VAL],
        "Test": [SplitName.TEST],
    }
    rows = []
    for split_name, splits in splits_sets.items():
        dataset_split = dataset.create_split_dataset(splits)
        table = dataset_split.samples
        row: Dict[str, Any] = {}
        row["Split"] = split_name
        row["Samples "] = str(len(table))

        primitive_counts = calculate_primitive_counts(dataset_split)
        row.update(primitive_counts)
        rows.append(row)

    return rows


def calculate_video_stats(dataset: HafniaDataset) -> Dict[str, Any]:
    """Aggregate per-video and per-camera statistics for video-derived datasets.

    When the dataset has `video_info` and/or `camera_info` columns, this returns a dictionary with
    the number of unique videos, total/average duration, average frame rate and bit rate, the
    earliest and latest capture/download dates, and the number of unique cameras — depending on
    which fields are populated. Returns an empty dict for datasets without video/camera metadata.
    """
    stats = {}
    samples = dataset.samples

    has_video_info = SampleField.VIDEO_INFO in samples.columns and samples[SampleField.VIDEO_INFO].dtype == pl.Struct
    if has_video_info:
        field_names = [f.name for f in samples.schema[SampleField.VIDEO_INFO].fields if f.dtype is not pl.Null]
        video_data = samples.select(pl.col(SampleField.VIDEO_INFO).struct.unnest()).unique()

        stats["n_videos"] = len(video_data)

        if VideoInfoField.DURATION_SECONDS in field_names:
            stats["duration"] = video_data.select(pl.col(VideoInfoField.DURATION_SECONDS).sum())[0, 0]
            stats["duration_average"] = video_data.select(pl.col(VideoInfoField.DURATION_SECONDS).mean())[0, 0]

        if VideoInfoField.FRAME_RATE in field_names:
            stats["frame_rate"] = video_data.select(pl.col(VideoInfoField.FRAME_RATE).mean())[0, 0]

        if VideoInfoField.BIT_RATE_KBPS in field_names:
            stats["bit_rate"] = video_data.select(pl.col(VideoInfoField.BIT_RATE_KBPS).mean())[0, 0]

        if VideoInfoField.DOWNLOADED_AT in field_names:
            dates = video_data[VideoInfoField.DOWNLOADED_AT].str.to_datetime().drop_nulls().sort()
            if len(dates) > 0:
                stats["data_received_start"] = dates[0]
                stats["data_received_end"] = dates[-1]

        if VideoInfoField.CAPTURED_AT in field_names:
            dates = video_data[VideoInfoField.CAPTURED_AT].str.to_datetime().drop_nulls().sort()
            if len(dates) > 0:
                stats["data_captured_start"] = dates[0]
                stats["data_captured_end"] = dates[-1]

    has_camera_info = SampleField.CAMERA_INFO in samples.columns and samples[SampleField.CAMERA_INFO].dtype == pl.Struct
    if has_camera_info:
        field_names = [f.name for f in samples.schema[SampleField.CAMERA_INFO].fields if f.dtype is not pl.Null]
        camera_data = samples.select(pl.col(SampleField.CAMERA_INFO).struct.unnest()).unique()

        stats["n_cameras"] = len(camera_data)
    return stats


def print_basic_stats(dataset: HafniaDataset) -> None:
    """Print a small `rich` table summarizing dataset name, version, sample count, and per-task class counts.

    Side-effecting only — no return value. Use `print_stats` for a full breakdown including
    per-split sample counts and class distributions.
    """
    table = Table(title="Dataset Statistics", box=rich.box.SIMPLE)
    table.add_column("Property", style="cyan")
    table.add_column("Value")
    table.add_row("Dataset Name", dataset.info.dataset_name)
    table.add_row("Version", dataset.info.version)
    table.add_row("Number of samples", str(len(dataset.samples)))

    for task in dataset.info.tasks:
        class_names = task.get_class_names()
        class_count = len(class_names) if class_names else "N/A"
        table.add_row(f"Task: {task.full_name()}", f"Number of classes: {class_count}")

    rprint(table)


def print_stats(dataset: HafniaDataset) -> None:
    """Print the full dataset overview: basic info, per-split sample/task counts, and per-task class distributions.

    Composes `print_basic_stats`, `print_sample_and_task_counts` and `print_class_distribution`
    into a single call. Side-effecting only — output goes to stdout via `rich`.
    """
    print_basic_stats(dataset)
    print_sample_and_task_counts(dataset)
    print_class_distribution(dataset)


def print_class_distribution(
    dataset: HafniaDataset,
    primitive: Optional[Type[Primitive]] = None,
    task_name: Optional[str] = None,
) -> None:
    """Print one `rich` table per task showing class name, class index and annotation count.

    By default iterates over every task in the dataset; pass `primitive` and/or `task_name` to
    restrict the output to specific tasks. Raises if a task selected by the filter has no class
    names defined.

    Args:
        primitive: Optional primitive type to filter on (e.g. `Bbox`).
        task_name: Optional task name to filter on.
    """
    tasks = dataset.info.get_tasks_by_task_name_and_primitive(task_name=task_name, primitive=primitive)
    for task in tasks:
        if task.get_class_names() is None:
            raise ValueError(f"Task '{task.name}' does not have class names defined.")
        class_counts = dataset.calculate_task_class_counts(primitive=task.primitive, task_name=task.name)

        # Print class distribution
        rich_table = Table(
            title=f"Class Count for '{task.primitive.__name__}/{task.name}'",
            show_lines=False,
        )
        rich_table.add_column("Class Name", style="cyan")
        rich_table.add_column("Class Idx", style="cyan")
        rich_table.add_column("Count", justify="right")
        for class_name, count in class_counts.items():
            class_names = task.get_class_names()
            if class_names is None:
                continue
            class_idx = class_names.index(class_name)  # Get class idx from task info
            rich_table.add_row(class_name, str(class_idx), str(count))
        rprint(rich_table)


def print_sample_and_task_counts(dataset: HafniaDataset) -> None:
    """Print a `rich` table with sample counts and per-task primitive counts, one row per split.

    The rows are: total ("All"), train, validation and test. Backed by
    `calculate_split_counts_extended`.
    """
    rows = calculate_split_counts_extended(dataset)
    rich_table = Table(title="Dataset Statistics", box=rich.box.SIMPLE)
    for i_row, row in enumerate(rows):
        if i_row == 0:
            for column_name in row.keys():
                rich_table.add_column(column_name, justify="left", style="cyan")
        rich_table.add_row(*[str(value) for value in row.values()])
    rprint(rich_table)


def check_dataset(dataset: HafniaDataset, check_splits: bool = True):
    """Run integrity checks on a dataset and raise on the first violation found.

    Verifies that the dataset name is non-empty, that `info.tasks` is consistent with the
    primitive columns in `samples` (delegates to `check_dataset_tasks`), and — when
    `check_splits=True` — that a sample subset exists and that every required split is present.
    Each `Sample` is also instantiated as a sanity check on the row schema.

    Args:
        check_splits: If True, additionally require a non-empty sample subset and the presence of
            every split in `SplitName.valid_splits()`. Disable for partial datasets that
            intentionally lack one or more splits.
    """

    user_logger.info("Checking Hafnia dataset...")
    assert isinstance(dataset.info.dataset_name, str) and len(dataset.info.dataset_name) > 0

    if check_splits:
        sample_dataset = dataset.create_sample_dataset()
        if len(sample_dataset) == 0:
            raise ValueError("The dataset does not include a sample dataset")

        actual_splits = dataset.samples.select(pl.col(SampleField.SPLIT)).unique().to_series().to_list()
        required_splits = SplitName.valid_splits()

        if not set(required_splits).issubset(set(actual_splits)):
            raise ValueError(f"Expected all splits '{required_splits}' in dataset, but got '{actual_splits}'. ")

    dataset.check_dataset_tasks()

    expected_tasks = dataset.info.tasks
    # Check that tasks found in the 'dataset.samples' matches the tasks defined in 'dataset.info.tasks'
    for PrimitiveType in PRIMITIVE_TYPES:
        column_name = PrimitiveType.column_name()
        if column_name not in dataset.samples.columns:
            continue
        objects_df = create_primitive_table(dataset.samples, PrimitiveType=PrimitiveType, keep_sample_data=False)
        if objects_df is None:
            continue
        for (task_name,), object_group in objects_df.group_by(PrimitiveField.TASK_NAME):
            has_task = any([t for t in expected_tasks if t.name == task_name and t.primitive == PrimitiveType])
            if has_task:
                continue
            class_names = object_group[PrimitiveField.CLASS_NAME].unique().to_list()
            raise ValueError(
                f"Task name '{task_name}' for the '{PrimitiveType.__name__}' primitive is missing in "
                f"'dataset.info.tasks' for dataset '{dataset.info.dataset_name}'. Missing task has the following "
                f"classes: {class_names}. "
            )

    for sample_dict in progress_bar(dataset, description="Checking samples in dataset"):
        sample = Sample(**sample_dict)  # noqa: F841


def check_dataset_tasks(dataset: HafniaDataset):
    """Verify that every task in `dataset.info.tasks` is consistent with the rows in `dataset.samples`.

    For each task, the corresponding primitive column must exist in the samples table, contain at
    least one annotation matching the task's name (for non-empty datasets), and use only class
    names that are subsets of `task.class_names` with matching class indices. Also rejects
    duplicate task names. Raises `ValueError` on the first violation found.
    """
    dataset.info.check_for_duplicate_task_names()

    for task in dataset.info.tasks:
        primitive = task.primitive.__name__
        column_name = task.primitive.column_name()

        if column_name not in dataset.samples.columns:
            raise ValueError(
                f"Column '{column_name}' for primitive '{primitive}' and task '{task.name}' is missing in 'dataset.samples' "
                f"for dataset '{dataset.info.dataset_name}'. Please check the dataset."
            )
        primitive_column = dataset.samples[column_name]
        msg_something_wrong = (
            f"Something is wrong with the defined tasks ('info.tasks') in dataset '{dataset.info.dataset_name}'. \n"
            f"For '{primitive=}' and '{task.name=}' "
        )
        if primitive_column.dtype == pl.Null:
            raise ValueError(msg_something_wrong + "the column is 'Null'. Please check the dataset.")

        if len(dataset) > 0:  # Check only performed for non-empty datasets
            primitive_table = (
                primitive_column.explode().struct.unnest().filter(pl.col(PrimitiveField.TASK_NAME) == task.name)
            )
            if primitive_table.is_empty():
                raise ValueError(
                    msg_something_wrong
                    + f"the column '{column_name}' has no {task.name=} objects. Please check the dataset."
                )

            actual_classes = set(primitive_table[PrimitiveField.CLASS_NAME].unique().to_list())
            if task.get_class_names() is None:
                raise ValueError(
                    msg_something_wrong
                    + f"the column '{column_name}' with {task.name=} has no defined classes. Please check the dataset."
                )
            defined_classes = set(task.get_class_names() or [])

            if not actual_classes.issubset(defined_classes):
                raise ValueError(
                    msg_something_wrong
                    + f"the column '{column_name}' with {task.name=} we expected the actual classes in the dataset to \n"
                    f"to be a subset of the defined classes\n\t{actual_classes=} \n\t{defined_classes=}."
                )
            # Check class_indices
            class_names = task.get_class_names() or []
            mapped_indices = primitive_table[PrimitiveField.CLASS_NAME].map_elements(
                lambda x: class_names.index(x), return_dtype=pl.Int64
            )
            table_indices = primitive_table[PrimitiveField.CLASS_IDX]

            error_msg = msg_something_wrong + (
                f"class indices in '{PrimitiveField.CLASS_IDX}' column does not match classes ordering in 'task.class_names'"
            )
            assert mapped_indices.equals(table_indices), error_msg

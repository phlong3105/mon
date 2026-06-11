from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type, Union

import polars as pl
from packaging.version import Version

from hafnia import utils
from hafnia.dataset import dataset_helpers
from hafnia.dataset.benchmark.metrics import (
    classification_metrics,
    object_detection_metrics,
)
from hafnia.dataset.dataset_helpers import is_valid_version_string, version_from_string
from hafnia.dataset.dataset_names import (
    TAG_IS_SAMPLE,
    PrimitiveField,
    SampleField,
    SplitName,
    StorageFormat,
)
from hafnia.dataset.format_conversions import (
    format_coco,
    format_encord,
    format_image_classification_folder,
    format_yolo,
)
from hafnia.dataset.hafnia_dataset_types import (
    DatasetInfo,
    DatasetMetadataFilePaths,
    Sample,
    TaskInfo,
)
from hafnia.dataset.operations import (
    dataset_stats,
    dataset_transformations,
    flattening_attributes,
    table_transformations,
)
from hafnia.dataset.operations.adjust_mask import (
    adjust_bboxes_from_polygon_masks_dataset,
)
from hafnia.dataset.primitives.primitive import Primitive
from hafnia.log import user_logger
from hafnia.platform import s5cmd_utils
from hafnia.platform.datasets import get_read_credentials_by_name
from hafnia.platform.s5cmd_utils import ResourceCredentials
from hafnia.utils import progress_bar
from hafnia_cli.config import Config

if TYPE_CHECKING:
    import boto3


@dataclass
class HafniaDataset:
    info: DatasetInfo
    samples: pl.DataFrame

    # Function mapping: Dataset stats
    calculate_split_counts = dataset_stats.calculate_split_counts
    calculate_split_counts_extended = dataset_stats.calculate_split_counts_extended
    calculate_task_class_counts = dataset_stats.calculate_task_class_counts
    calculate_class_counts = dataset_stats.calculate_class_counts
    calculate_primitive_counts = dataset_stats.calculate_primitive_counts
    calculate_video_stats = dataset_stats.calculate_video_stats

    # Function mapping: Print stats
    print_basic_stats = dataset_stats.print_basic_stats
    print_stats = dataset_stats.print_stats
    print_sample_and_task_counts = dataset_stats.print_sample_and_task_counts
    print_class_distribution = dataset_stats.print_class_distribution

    # Function mapping: Dataset checks
    check_dataset = dataset_stats.check_dataset
    check_dataset_tasks = dataset_stats.check_dataset_tasks

    # Function mapping: Dataset transformations
    transform_images = dataset_transformations.transform_images
    convert_to_image_storage_format = dataset_transformations.convert_to_image_storage_format

    # Function mapping: Benchmark metrics
    calculate_mean_average_precision = object_detection_metrics.calculate_mean_average_precision
    calculate_classification_metrics = classification_metrics.calculate_classification_metrics

    # Import / export functions
    to_yolo_format = format_yolo.to_yolo_format
    from_yolo_format = format_yolo.from_yolo_format
    to_coco_format = format_coco.to_coco_format
    from_coco_format = format_coco.from_coco_format
    to_image_classification_folder = format_image_classification_folder.to_image_classification_folder
    from_image_classification_folder = format_image_classification_folder.from_image_classification_folder
    from_encord_zip_format = format_encord.from_encord_zip_format

    def __getitem__(self, item: int) -> Dict[str, Any]:
        return self.samples.row(index=item, named=True)

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self):
        for row in self.samples.iter_rows(named=True):
            yield row

    def __post_init__(self):
        self.samples, self.info = _dataset_corrections(self.samples, self.info)

    @staticmethod
    def from_path(path_folder: Path, check_for_images: bool = True) -> "HafniaDataset":
        """Load a `HafniaDataset` from a local folder previously written with `HafniaDataset.write`.

        Reads `dataset_info.json` plus the annotations file (Parquet preferred, JSONL fallback) from
        `path_folder`, rewrites sample file paths as absolute paths rooted at the folder, and
        optionally verifies that every referenced image exists.

        Args:
            path_folder: Local directory containing the dataset metadata and (optionally) images.
            check_for_images: If True, raise if any sample's image file is missing on disk.
        """
        path_folder = Path(path_folder)
        metadata_file_paths = DatasetMetadataFilePaths.from_path(path_folder)
        metadata_file_paths.exists(raise_error=True)

        dataset_info = DatasetInfo.from_json_file(Path(metadata_file_paths.dataset_info))
        samples = metadata_file_paths.read_samples()
        samples, dataset_info = _dataset_corrections(samples, dataset_info)

        # Convert from relative paths to absolute paths
        dataset_root = path_folder.absolute().as_posix() + "/"
        samples = samples.with_columns((dataset_root + pl.col(SampleField.FILE_PATH)).alias(SampleField.FILE_PATH))
        if check_for_images:
            table_transformations.check_image_paths(samples)
        return HafniaDataset(samples=samples, info=dataset_info)

    @staticmethod
    def from_name(
        name: str,
        version: Optional[str] = None,
        force_redownload: bool = False,
        download_files: bool = True,
    ) -> "HafniaDataset":
        """Load a dataset by name from the Hafnia platform.

        Returns the **sample dataset** locally and the **full dataset** when running under
        Training-aaS, so the same call works in both environments. The sample dataset is cached
        under `.data/datasets/<name>/` and only re-downloaded when `force_redownload=True` or when
        `version` is ``None``/``"latest"`` (which always re-resolves the latest version).

        Args:
            name: Dataset name as registered on the platform. The ``"name:version"`` shorthand is
                **not** accepted — pass `version` separately.
            version: Specific dataset version (e.g. ``"1.0.0"``), ``"latest"``, or ``None``. If
                ``None``, the function raises with the list of available versions.
            force_redownload: If True, re-download even if a cached copy is present.
            download_files: If True, download image files alongside the annotations. Set to False
                when only annotations are needed (e.g. for stats or filtering).
        """
        if ":" in name:
            name, version = dataset_helpers.dataset_name_and_version_from_string(name)
            raise ValueError(
                "The 'from_name' does not support the 'name:version' format. Please provide the version separately.\n"
                f"E.g., HafniaDataset.from_name(name='{name}', version='{version}')"
            )
        dataset_path = download_or_get_dataset_path(
            dataset_name=name,
            version=version,
            force_redownload=force_redownload,
            download_files=download_files,
        )
        return HafniaDataset.from_path(dataset_path, check_for_images=download_files)

    @staticmethod
    def from_s3(
        bucket_prefix: str,
        path_local: Path,
        session: "boto3.Session",
        version: Optional[str] = None,
        force_redownload: bool = False,
        download_files: bool = True,
    ) -> "HafniaDataset":
        """Download a Hafnia dataset from a user-controlled S3 bucket.

        Args:
            bucket_prefix: S3 location without scheme, e.g. ``"my-bucket/sample"``.
            path_local: Local folder to download into.
            session: Boto3 session used to obtain credentials.
            version: Dataset version string (e.g. ``"0.1.0"``) or ``None`` for latest.
            force_redownload: If ``True``, re-download data files that already exist locally.
            download_files: If ``False``, to only download metadata files. ``True`` by default to also download dataset images/videos.
        """
        from hafnia.dataset.operations.dataset_s3_storage import download_dataset_from_s3

        return download_dataset_from_s3(
            bucket_prefix=bucket_prefix,
            path_local=path_local,
            session=session,
            version=version,
            force_redownload=force_redownload,
            download_files=download_files,
        )

    def to_s3(
        self,
        bucket_prefix: str,
        session: "boto3.Session",
        allow_version_overwrite: bool = False,
        create_bucket_if_missing: bool = True,
        interactive: bool = True,
    ) -> None:
        """Upload this dataset to a user-controlled S3 bucket.

        Args:
            bucket_prefix: S3 location without scheme, e.g. ``"my-bucket"`` or
                ``"my-bucket/sample"``. The leading segment is treated as the
                bucket name for the optional create step.
            session: Boto3 session used to obtain credentials and (optionally)
                create the bucket.
            allow_version_overwrite: If ``True``, overwrite metadata files for an
                existing version in S3.
            create_bucket_if_missing: If ``True`` and the target bucket does not
                exist, create it in the session's region before uploading.
            interactive: If ``True``, prompt for confirmation before uploading.
        """
        from hafnia.dataset.operations.dataset_s3_storage import upload_dataset_to_s3

        upload_dataset_to_s3(
            dataset=self,
            bucket_prefix=bucket_prefix,
            session=session,
            allow_version_overwrite=allow_version_overwrite,
            create_bucket_if_missing=create_bucket_if_missing,
            interactive=interactive,
        )

    @staticmethod
    def from_samples_list(samples_list: List, info: DatasetInfo) -> "HafniaDataset":
        """Create a `HafniaDataset` from a list of `Sample` objects (or sample dicts) and a `DatasetInfo`.

        Used when constructing datasets from custom data — see `examples/example_custom_dataset.py`.
        Adds a sample index and fills in the dataset name on the resulting samples table.

        Args:
            samples_list: List of `Sample` instances or dicts matching the `Sample` schema.
            info: Dataset-level metadata (name, version, tasks, ...).
        """
        sample = samples_list[0]
        if isinstance(sample, Sample):
            json_samples = [sample.model_dump(mode="json") for sample in samples_list]
        elif isinstance(sample, dict):
            json_samples = samples_list
        else:
            raise TypeError(f"Unsupported sample type: {type(sample)}. Expected Sample or dict.")

        # To ensure that the 'file_path' column is of type string even if all samples have 'None' as file_path
        schema_override = {
            SampleField.FILE_PATH: pl.String,
            SampleField.TAGS: pl.List(pl.String),
        }
        table = pl.from_records(json_samples, schema_overrides=schema_override, infer_schema_length=None)
        table = table.drop(pl.selectors.by_dtype(pl.Null))
        table = table_transformations.add_sample_index(table)
        table = table_transformations.add_dataset_name_if_missing(table, dataset_name=info.dataset_name)
        return HafniaDataset(info=info, samples=table)

    @staticmethod
    def from_merge(dataset0: "HafniaDataset", dataset1: "HafniaDataset") -> "HafniaDataset":
        """Merge two datasets into one. Thin wrapper around `HafniaDataset.merge`.

        See `merge` for compatibility checks and the handling of incompatible primitives.
        """
        return HafniaDataset.merge(dataset0, dataset1)

    @staticmethod
    def from_recipe_with_cache(
        dataset_recipe: Any,
        force_redownload: bool = False,
        path_datasets: Optional[Union[Path, str]] = None,
    ) -> "HafniaDataset":
        """Build a dataset from a `DatasetRecipe` with on-disk caching keyed by the recipe contents.

        The first call materializes the recipe and writes the resulting dataset to a cache folder
        named after the recipe's stable hash; subsequent calls with the same recipe load from the
        cache without re-running operations.

        Args:
            dataset_recipe: A `DatasetRecipe` instance describing how to construct the dataset.
            force_redownload: If True, ignore the cache and rebuild from scratch.
            path_datasets: Optional override for the cache root. Defaults to the standard datasets
                folder used by `HafniaDataset.from_name`.
        """

        from hafnia.dataset.dataset_recipe.dataset_recipe import (
            get_or_create_dataset_path_from_recipe,
        )

        path_dataset = get_or_create_dataset_path_from_recipe(
            dataset_recipe,
            path_datasets=path_datasets,
            force_redownload=force_redownload,
        )
        return HafniaDataset.from_path(path_dataset, check_for_images=False)

    @staticmethod
    def from_merger(
        datasets: List[HafniaDataset],
    ) -> "HafniaDataset":
        """Merge a list of `HafniaDataset` instances into one by repeatedly applying `merge`.

        Tasks and primitive columns shared across all inputs are preserved; tasks/primitives that
        appear in only some of the inputs are dropped during the merge with a warning. Returns the
        single dataset unchanged if the list has length one.

        Args:
            datasets: Non-empty list of datasets to merge.
        """
        if len(datasets) == 0:
            raise ValueError("No datasets to merge. Please provide at least one dataset.")

        if len(datasets) == 1:
            return datasets[0]

        merged_dataset = datasets[0]
        remaining_datasets = datasets[1:]
        for dataset in remaining_datasets:
            merged_dataset = HafniaDataset.merge(merged_dataset, dataset)
        return merged_dataset

    @staticmethod
    def from_name_public_dataset(
        name: str,
        force_redownload: bool = False,
        n_samples: Optional[int] = None,
    ) -> HafniaDataset:
        """Load a public dataset (e.g. one of the supported torchvision datasets) as a `HafniaDataset`.

        Unlike `from_name`, this does not require platform credentials — the dataset is fetched
        from its public source and converted into the Hafnia format.

        Args:
            name: Public dataset identifier. Use `torchvision_to_hafnia_converters()` to list the
                supported names.
            force_redownload: If True, re-download even if a local copy is already present.
            n_samples: Optional cap on the number of samples to materialize.
        """
        from hafnia.dataset.format_conversions.torchvision_datasets import (
            torchvision_to_hafnia_converters,
        )

        name_to_torchvision_function = torchvision_to_hafnia_converters()

        if name not in name_to_torchvision_function:
            raise ValueError(
                f"Unknown torchvision dataset name: {name}. Supported: {list(name_to_torchvision_function.keys())}"
            )
        vision_dataset = name_to_torchvision_function[name]
        return vision_dataset(
            force_redownload=force_redownload,
            n_samples=n_samples,
        )

    def shuffle(dataset: HafniaDataset, seed: int = 42) -> HafniaDataset:
        """Return a new dataset with samples shuffled deterministically.

        Args:
            seed: Random seed controlling the shuffle order. The same seed always produces the
                same ordering for a given input dataset.
        """
        table = dataset.samples.sample(n=len(dataset), with_replacement=False, seed=seed, shuffle=True)
        return dataset.update_samples(table)

    def select_samples(
        dataset: "HafniaDataset",
        n_samples: int,
        shuffle: bool = True,
        seed: int = 42,
        with_replacement: bool = False,
    ) -> "HafniaDataset":
        """Return a new dataset containing a random subset of `n_samples` samples.

        Args:
            n_samples: Number of samples to draw. When `with_replacement=False`, this is capped
                to the dataset size.
            shuffle: If True, the drawn samples are shuffled in the returned dataset.
            seed: Random seed for deterministic sampling.
            with_replacement: If True, samples may be drawn more than once (useful for upsampling
                small datasets).
        """
        if not with_replacement:
            n_samples = min(n_samples, len(dataset))
        table = dataset.samples.sample(n=n_samples, with_replacement=with_replacement, seed=seed, shuffle=shuffle)
        return dataset.update_samples(table)

    def splits_by_ratios(dataset: "HafniaDataset", split_ratios: Dict[str, float], seed: int = 42) -> "HafniaDataset":
        """
        Divides the dataset into splits based on the provided ratios.

        Example: Defining split ratios and applying the transformation

        >>> dataset = HafniaDataset.from_path(Path("path/to/dataset"))
        >>> split_ratios = {SplitName.TRAIN: 0.8, SplitName.VAL: 0.1, SplitName.TEST: 0.1}
        >>> dataset_with_splits = splits_by_ratios(dataset, split_ratios, seed=42)
        Or use the function as a
        >>> dataset_with_splits = dataset.splits_by_ratios(split_ratios, seed=42)
        """
        n_items = len(dataset)
        split_name_column = dataset_helpers.create_split_name_list_from_ratios(
            split_ratios=split_ratios, n_items=n_items, seed=seed
        )
        table = dataset.samples.with_columns(pl.Series(split_name_column).alias("split"))
        return dataset.update_samples(table)

    def split_into_multiple_splits(
        dataset: "HafniaDataset",
        split_name: str,
        split_ratios: Dict[str, float],
    ) -> "HafniaDataset":
        """
        Divides a dataset split ('split_name') into multiple splits based on the provided split
        ratios ('split_ratios'). This is especially useful for some open datasets where they have only provide
        two splits or only provide annotations for two splits. This function allows you to create additional
        splits based on the provided ratios.

        Example: Defining split ratios and applying the transformation
        >>> dataset = HafniaDataset.from_path(Path("path/to/dataset"))
        >>> split_name = SplitName.TEST
        >>> split_ratios = {SplitName.TEST: 0.8, SplitName.VAL: 0.2}
        >>> dataset_with_splits = split_into_multiple_splits(dataset, split_name, split_ratios)
        """
        dataset_split_to_be_divided = dataset.create_split_dataset(split_name=split_name)
        if len(dataset_split_to_be_divided) == 0:
            split_counts = dict(dataset.samples.select(pl.col(SampleField.SPLIT).value_counts()).iter_rows())
            raise ValueError(f"No samples in the '{split_name}' split to divide into multiple splits. {split_counts=}")
        assert len(dataset_split_to_be_divided) > 0, f"No samples in the '{split_name}' split!"
        dataset_split_to_be_divided = dataset_split_to_be_divided.splits_by_ratios(split_ratios=split_ratios, seed=42)

        remaining_data = dataset.samples.filter(pl.col(SampleField.SPLIT).is_in([split_name]).not_())
        new_table = pl.concat([remaining_data, dataset_split_to_be_divided.samples], how="vertical")
        dataset_new = dataset.update_samples(new_table)
        return dataset_new

    def define_sample_set_by_size(dataset: "HafniaDataset", n_samples: int, seed: int = 42) -> "HafniaDataset":
        """Tag `n_samples` randomly chosen samples as the dataset's "sample" subset.

        Adds the `sample` tag to the selected rows and removes any pre-existing `sample` tag from
        the others. The resulting dataset can then be reduced to the sample subset via
        `create_sample_dataset`. Used to define the small, anonymized sample dataset that ships
        with full datasets on the platform.

        Args:
            n_samples: Number of samples to mark.
            seed: Random seed for deterministic selection.
        """
        samples = dataset.samples

        # Remove any pre-existing "sample"-tags
        samples = samples.with_columns(
            pl.col(SampleField.TAGS)
            .list.eval(pl.element().filter(pl.element() != TAG_IS_SAMPLE))
            .alias(SampleField.TAGS)
        )

        # Add "sample" to tags column for the selected samples
        is_sample_indices = Random(seed).sample(range(len(dataset)), n_samples)
        samples = samples.with_columns(
            pl.when(pl.int_range(len(samples)).is_in(is_sample_indices))
            .then(pl.col(SampleField.TAGS).list.concat(pl.lit([TAG_IS_SAMPLE])))
            .otherwise(pl.col(SampleField.TAGS))
        )
        return dataset.update_samples(samples)

    def class_mapper(
        dataset: "HafniaDataset",
        class_mapping: Union[Dict[str, str], List[Tuple[str, str]]],
        method: str = "strict",
        primitive: Optional[Type[Primitive]] = None,
        task_name: Optional[str] = None,
    ) -> "HafniaDataset":
        """
        Map class names to new class names using a strict mapping.
        A strict mapping means that all class names in the dataset must be mapped to a new class name.
        If a class name is not mapped, an error is raised.

        The class indices are determined by the order of appearance of the new class names in the mapping.
        Duplicates in the new class names are removed, preserving the order of first appearance.

        E.g.

        mnist = HafniaDataset.from_name("mnist")
        strict_class_mapping = {
            "1 - one": "odd",   # 'odd' appears first and becomes class index 0
            "3 - three": "odd",
            "5 - five": "odd",
            "7 - seven": "odd",
            "9 - nine": "odd",
            "0 - zero": "even",  # 'even' appears second and becomes class index 1
            "2 - two": "even",
            "4 - four": "even",
            "6 - six": "even",
            "8 - eight": "even",
        }

        dataset_new = class_mapper(dataset=mnist, class_mapping=strict_class_mapping)

        """
        return dataset_transformations.class_mapper(
            dataset=dataset,
            class_mapping=class_mapping,
            method=method,
            primitive=primitive,
            task_name=task_name,
        )

    def flattening_by_specification(
        dataset: "HafniaDataset",
        flattening_specification: Dict[TaskInfo, List[List[str]]],
    ) -> "HafniaDataset":
        """Flatten hierarchical class names for one or more tasks according to a specification.

        For each task in `flattening_specification`, applies the listed flatten-types in order to
        collapse hierarchical class names (e.g. ``Vehicle.Car`` → ``Vehicle``) and updates the
        corresponding `TaskInfo`. Raises if the spec references a task that does not exist on this
        dataset.

        Args:
            flattening_specification: Mapping from `TaskInfo` (matched by task name and primitive)
                to a list of flatten-type lists to apply sequentially.
        """
        updated_tasks = []
        samples = dataset.samples
        for task, flatten_types in flattening_specification.items():
            try:
                task_updated = dataset.info.get_task_by_name(task_name=task.name).model_copy(deep=True)
            except ValueError as e:
                task_names = [
                    f"'{task_info.name}' ({task_info.primitive.__name__})" for task_info in dataset.info.tasks
                ]
                raise ValueError(
                    f"A '{task.primitive.__name__}' primitive with task name '{task.name}' in the flattening "
                    "specification does not exist in the dataset.\n"
                    f"Available tasks in the dataset: {task_names}"
                ) from e

            for flatten_types_list in flatten_types:
                samples, task_updated = flattening_attributes.flatten_class_names(
                    samples=samples,
                    task_info=task_updated,
                    flatten_types=flatten_types_list,
                )

            updated_tasks.append(task_updated)

        dataset_info = dataset.info.model_copy(deep=True)
        for task_updated in updated_tasks:
            dataset_info = dataset_info.replace_task(old_task=task_updated, new_task=task_updated)
        return HafniaDataset(info=dataset_info, samples=samples)

    def rename_task(
        dataset: "HafniaDataset",
        old_task_name: str,
        new_task_name: str,
    ) -> "HafniaDataset":
        """Rename a task in `dataset.info` and propagate the new name to every annotation row.

        Args:
            old_task_name: Existing task name to rename. Raises if no task with this name exists.
            new_task_name: New task name to assign.
        """
        return dataset_transformations.rename_task(
            dataset=dataset, old_task_name=old_task_name, new_task_name=new_task_name
        )

    def drop_task(
        dataset: "HafniaDataset",
        task_name: str,
    ) -> "HafniaDataset":
        """Drop a single task from the dataset.

        If the task is the only task using its primitive type (e.g. the only `Bbox` task), the
        primitive column is dropped entirely; otherwise only the matching rows are filtered out
        of the primitive's `List[Struct]` column. The dataset's `info` is updated accordingly.

        Args:
            task_name: Name of the task to drop. Raises if no task with this name exists.
        """
        dataset = copy.copy(dataset)  # To avoid mutating the original dataset. Shallow copy is sufficient
        drop_task = dataset.info.get_task_by_name(task_name=task_name)
        tasks_with_same_primitive = dataset.info.get_tasks_by_primitive(drop_task.primitive)

        no_other_tasks_with_same_primitive = len(tasks_with_same_primitive) == 1
        if no_other_tasks_with_same_primitive:
            return dataset.drop_primitive(primitive=drop_task.primitive)

        column_name = drop_task.primitive.column_name()
        dataset.info = dataset.info.replace_task(old_task=drop_task, new_task=None)

        # If column is already dropped return early
        if column_name not in dataset.samples.columns:
            return dataset

        dataset.samples = dataset.samples.with_columns(
            pl.col(column_name)
            .list.filter(pl.element().struct.field(PrimitiveField.TASK_NAME) != drop_task.name)
            .alias(column_name)
        )

        return dataset

    def drop_primitive(
        dataset: "HafniaDataset",
        primitive: Type[Primitive],
    ) -> "HafniaDataset":
        """Drop every task that uses the given primitive and remove its column from the samples.

        Use this to fully remove a primitive type (e.g. drop all polygons from a dataset). To drop
        a single task while preserving others of the same primitive type, use `drop_task`.

        Args:
            primitive: Primitive class to remove (e.g. `Bbox`, `Polygon`, `Bitmask`,
                `Classification`).
        """
        dataset = copy.copy(dataset)  # To avoid mutating the original dataset. Shallow copy is sufficient
        tasks_to_drop = dataset.info.get_tasks_by_primitive(primitive=primitive)
        for task in tasks_to_drop:
            dataset.info = dataset.info.replace_task(old_task=task, new_task=None)

        # Drop the primitive column from the samples table
        dataset.samples = dataset.samples.drop(primitive.column_name(), strict=False)
        return dataset

    def select_samples_by_class_name(
        dataset: HafniaDataset,
        name: Union[List[str], str],
        task_name: Optional[str] = None,
        primitive: Optional[Type[Primitive]] = None,
    ) -> HafniaDataset:
        """Keep only samples containing at least one annotation with the specified class name(s).

        If neither `task_name` nor `primitive` is provided, the function attempts to infer the
        unique task that contains the given class names; ambiguity raises an error.

        Args:
            name: A class name or list of class names to filter on.
            task_name: Optional task to look up the class names in. Use this to disambiguate when
                the same class name exists across multiple tasks.
            primitive: Optional primitive type to constrain the lookup (e.g. `Bbox`).
        """
        return dataset_transformations.select_samples_by_class_name(
            dataset=dataset, name=name, task_name=task_name, primitive=primitive
        )

    def drop_samples_by_class_name(
        dataset: HafniaDataset,
        name: Union[List[str], str],
        task_name: Optional[str] = None,
        primitive: Optional[Type[Primitive]] = None,
        drop_classes_from_task_info: bool = True,
    ) -> HafniaDataset:
        """Drop samples containing any annotation with the specified class name(s).

        If neither `task_name` nor `primitive` is provided, the function attempts to infer the
        unique task that contains the given class names; ambiguity raises an error.

        Args:
            name: A class name or list of class names whose samples should be removed.
            task_name: Optional task to look up the class names in.
            primitive: Optional primitive type to constrain the lookup.
            drop_classes_from_task_info: If True (default), also remove the class names from the
                matching task's `class_names` list in `dataset.info` and re-assign class indices
                so they remain contiguous. Set to False to keep the task's class definitions
                intact while only filtering samples.
        """
        return dataset_transformations.drop_samples_by_class_name(
            dataset=dataset,
            name=name,
            task_name=task_name,
            primitive=primitive,
            drop_classes_from_task_info=drop_classes_from_task_info,
        )

    def merge(dataset0: "HafniaDataset", dataset1: "HafniaDataset") -> "HafniaDataset":
        """Merge two `HafniaDataset` instances by concatenating samples and combining their `info`.

        Validates that the two datasets are compatible (e.g. same `format_version`) before
        merging. Tasks and primitive columns shared by both inputs are kept; tasks/primitives that
        appear in only one input are dropped from the merged result with a warning logged through
        `user_logger` — when this matters, align the inputs first via `drop_task`/`drop_primitive`.

        Args:
            dataset0: First dataset.
            dataset1: Second dataset.
        """

        # Merges dataset info and checks for compatibility
        merged_info = DatasetInfo.merge(dataset0.info, dataset1.info)

        # Merges samples tables (removes incompatible columns)
        merged_samples = table_transformations.merge_samples(samples0=dataset0.samples, samples1=dataset1.samples)

        # Check if primitives have been removed during the merge_samples
        for task in copy.deepcopy(merged_info.tasks):
            if task.primitive.column_name() not in merged_samples.columns:
                user_logger.warning(
                    f"Task '{task.name}' with primitive '{task.primitive.__name__}' has been removed during the merge. "
                    "This happens if the two datasets do not have the same primitives."
                )
                merged_info = merged_info.replace_task(old_task=task, new_task=None)

        return HafniaDataset(info=merged_info, samples=merged_samples)

    def to_dict_dataset_splits(self) -> Dict[str, "HafniaDataset"]:
        """Return a dict mapping each valid split name to a per-split `HafniaDataset`.

        Iterates over `SplitName.valid_splits()` (typically `train`, `val`, `test`) and produces
        one entry per split using `create_split_dataset`. Splits with no samples are still
        included as empty datasets.

        Raises:
            ValueError: If the dataset's samples table has no `split` column.
        """
        if SampleField.SPLIT not in self.samples.columns:
            raise ValueError(f"Dataset must contain a '{SampleField.SPLIT}' column.")

        splits = {}
        for split_name in SplitName.valid_splits():
            splits[split_name] = self.create_split_dataset(split_name)

        return splits

    def create_sample_dataset(self) -> "HafniaDataset":
        """Return a new dataset containing only the samples tagged as part of the **sample** subset.

        Samples are tagged via `define_sample_set_by_size`. The returned dataset preserves the
        original `info` and only filters the samples table.
        """
        if SampleField.TAGS not in self.samples.columns:
            raise ValueError(f"Dataset must contain an '{SampleField.TAGS}' column.")

        table = self.samples.filter(
            pl.col(SampleField.TAGS).list.eval(pl.element().filter(pl.element() == TAG_IS_SAMPLE)).list.len() > 0
        )
        return self.update_samples(table)

    def create_split_dataset(self, split_name: Union[str | List[str]]) -> "HafniaDataset":
        """Return a new dataset containing only samples whose `split` matches `split_name`.

        Args:
            split_name: A single split name (e.g. ``"train"``) or a list of names. Each must be one
                of the values in `SplitName.all_split_names()`; otherwise a `ValueError` is raised.
        """
        if isinstance(split_name, str):
            split_names = [split_name]
        elif isinstance(split_name, list):
            split_names = split_name

        for name in split_names:
            if name not in SplitName.all_split_names():
                raise ValueError(f"Invalid split name: {split_name}. Valid splits are: {SplitName.valid_splits()}")

        filtered_dataset = self.samples.filter(pl.col(SampleField.SPLIT).is_in(split_names))
        return self.update_samples(filtered_dataset)

    def update_samples(self, table: pl.DataFrame) -> "HafniaDataset":
        """Return a new `HafniaDataset` with the samples replaced by `table` and a deep-copied `info`.

        Used by transformation methods to produce immutable, dataset-shaped results without
        mutating the original.
        """
        dataset = HafniaDataset(info=self.info.model_copy(deep=True), samples=table)
        return dataset

    def has_primitive(dataset: HafniaDataset, PrimitiveType: Type[Primitive]) -> bool:
        """Return True if the dataset contains at least one annotation of the given primitive type."""
        table = dataset.samples if isinstance(dataset, HafniaDataset) else dataset
        return table_transformations.has_primitive(table, PrimitiveType)

    def copy(self) -> "HafniaDataset":
        """Return a deep copy of the dataset (info is deep-copied, samples DataFrame is cloned)."""
        return HafniaDataset(info=self.info.model_copy(deep=True), samples=self.samples.clone())

    def adjust_bboxes_from_polygon_masks(
        self,
        polygon_class_names: List[str],
        run_checks: bool = True,
    ) -> "HafniaDataset":
        """Tighten bounding boxes around the polygon masks of the listed classes.

        For samples that have both `Bbox` and `Polygon` annotations on the same object, this
        replaces each `Bbox` with the tight axis-aligned box around the matching polygon for the
        given class names.

        Args:
            polygon_class_names: Class names whose polygons drive the bbox adjustment.
            run_checks: If True, run consistency checks before/after the adjustment.
        """
        return adjust_bboxes_from_polygon_masks_dataset(
            dataset=self,
            polygon_class_names=polygon_class_names,
            run_checks=run_checks,
        )

    def create_primitive_table(
        self,
        primitive: Type[Primitive],
        task_name: Optional[str] = None,
        keep_sample_data: bool = False,
    ) -> Optional[pl.DataFrame]:
        """Explode the dataset's samples into one row per primitive annotation of the given type.

        Useful for per-annotation analysis (e.g. computing class distributions, per-bbox statistics).

        Args:
            primitive: Primitive class to extract (e.g. `Bbox`, `Classification`, `Polygon`).
            task_name: Optional task name to filter on; if omitted, all annotations of the primitive
                type are returned regardless of task.
            keep_sample_data: If True, retain the per-sample columns (file path, split, ...) on the
                exploded rows. Defaults to False to keep the table compact.

        Returns:
            A Polars DataFrame with one row per matching annotation, or None if the primitive is
            absent from the dataset.
        """
        return table_transformations.create_primitive_table(
            samples_table=self.samples,
            PrimitiveType=primitive,
            task_name=task_name,
            keep_sample_data=keep_sample_data,
        )

    def write(self, path_folder: Path, drop_null_cols: bool = True) -> None:
        """Write the dataset to disk: copies all images into `path_folder/data/` and writes annotations
        to `path_folder`.

        Image files are renamed by content hash so that duplicates are de-duplicated. After copying,
        annotations are written via `write_annotations` (both Parquet and JSONL) and `dataset_info.json`.

        Args:
            path_folder: Destination folder. Created if it does not exist.
            drop_null_cols: If True, drop fully-null columns from the annotations files to keep them
                compact.
        """
        user_logger.info(f"Writing dataset to {path_folder}...")
        path_folder = path_folder.absolute()
        if not path_folder.exists():
            path_folder.mkdir(parents=True)
        hafnia_dataset = self.copy()  # To avoid inplace modifications
        new_paths = []
        org_paths = hafnia_dataset.samples[SampleField.FILE_PATH].to_list()
        for org_path in progress_bar(org_paths, description="- Copy images"):
            new_path = dataset_helpers.copy_and_rename_file_to_hash_value(
                path_source=Path(org_path),
                path_dataset_root=path_folder / "data",
            )
            new_paths.append(new_path.as_posix())
        hafnia_dataset.samples = hafnia_dataset.samples.with_columns(pl.Series(new_paths).alias(SampleField.FILE_PATH))
        hafnia_dataset.write_annotations(path_folder=path_folder, drop_null_cols=drop_null_cols)

    def write_annotations(dataset: HafniaDataset, path_folder: Path, drop_null_cols: bool = True) -> None:
        """Write `dataset_info.json` and the annotations files (JSONL and Parquet) to disk.

        Unlike `write`, this does **not** copy image files — it only persists the metadata. Sample
        file paths are converted to paths relative to `path_folder` if the `file_path` column is
        present (or replaced with empty strings for remote datasets).

        Args:
            path_folder: Destination folder. Created if it does not exist.
            drop_null_cols: If True (default), drop fully-null columns to keep the on-disk files
                compact.
        """

        user_logger.info(f"Writing dataset annotations to {path_folder}...")
        metadata_file_paths = DatasetMetadataFilePaths.from_path(path_folder)
        path_dataset_info = Path(metadata_file_paths.dataset_info)
        path_dataset_info.parent.mkdir(parents=True, exist_ok=True)
        dataset.info.write_json(path_dataset_info)

        samples = dataset.samples
        if drop_null_cols:  # Drops all unused/Null columns
            samples = samples.drop(pl.selectors.by_dtype(pl.Null))

        path_folder = path_folder.absolute()
        # Store only relative paths in the annotations files
        if SampleField.FILE_PATH in samples.columns:  # We drop column for remote datasets
            absolute_paths = samples[SampleField.FILE_PATH].to_list()
            relative_paths = [Path(path).relative_to(path_folder).as_posix() for path in absolute_paths]
            samples = samples.with_columns(pl.Series(relative_paths).alias(SampleField.FILE_PATH))
        else:
            samples = samples.with_columns(pl.lit("").alias(SampleField.FILE_PATH))

        if metadata_file_paths.annotations_jsonl:
            samples.write_ndjson(Path(metadata_file_paths.annotations_jsonl))  # Json for readability
        if metadata_file_paths.annotations_parquet:
            samples.write_parquet(Path(metadata_file_paths.annotations_parquet))  # Parquet for speed

    def delete_on_platform(dataset: HafniaDataset, interactive: bool = True) -> None:
        """
        Delete this dataset from the Hafnia platform.
        This is a thin wrapper around `hafnia.platform.datasets.delete_dataset_completely_by_name`.

        Args:
            dataset (HafniaDataset): The :class:`HafniaDataset` instance to delete from the platform. The
                dataset name is taken from `dataset.info.dataset_name`.
            interactive (bool): If ``True``, perform the deletion in interactive mode (for example,
                prompting the user for confirmation where supported). If ``False``,
                run non-interactively, suitable for automated scripts or CI usage. Defaults to True.
        """
        from hafnia.platform.datasets import delete_dataset_completely_by_name

        delete_dataset_completely_by_name(dataset_name=dataset.info.dataset_name, interactive=interactive)

    def upload_to_platform(
        dataset: HafniaDataset,
        dataset_sample: Optional[HafniaDataset] = None,
        allow_version_overwrite: bool = False,
        interactive: bool = True,
        gallery_samples: Optional[Union[pl.DataFrame, HafniaDataset]] = None,
        distribution_task_names: Optional[List[str]] = None,
        cfg: Optional[Config] = None,
    ) -> dict:
        """
        Upload the dataset and dataset details to the Hafnia platform.
        This method ensures the dataset exists on the platform, synchronizes the
        dataset files to remote storage, and uploads dataset details and optional gallery images
        distributions.
        Args:
            dataset: The full :class:`HafniaDataset` instance that should be uploaded
                to the platform.
            dataset_sample: Optional sample :class:`HafniaDataset` used as a smaller
                preview or subset of the main dataset on the platform. If provided,
                it is uploaded alongside the full dataset for demonstration or
                inspection purposes. Use only this if the sample dataset uses different
                image files than the main dataset. Otherwise it is sufficient to just provide
                the main dataset and the platform will create a sample automatically.
            allow_version_overwrite: If ``True``, allows an existing dataset version
                with the same name to be overwritten on the platform. If ``False``,
                an error or confirmation may be required when a version conflict is
                detected.
            interactive: If ``True``, the upload process may prompt the user for
                confirmation or additional input (for example when overwriting
                existing versions). If ``False``, the upload is performed without
                interactive prompts.
            gallery_samples: Optional :class:`pl.DataFrame` or :class:`HafniaDataset` containing a small set of
                samples to be used as gallery images for the dataset on the platform.
            distribution_task_names: Optional list of task names associated with the
                dataset that should be considered when configuring how the dataset is
                distributed or exposed on the platform.
            cfg: Optional :class:`hafnia_cli.config.Config` instance providing
                configuration for platform access and storage. If not supplied, a
                default configuration is created.
        Returns:
            dict: The response returned by the platform after uploading the dataset
            details. The exact contents depend on the platform API but typically
            include information about the created or updated dataset (such as
            identifiers and status).
        """

        from hafnia.dataset.dataset_details_uploader import (
            upload_dataset_details_to_platform,
        )
        from hafnia.dataset.operations.dataset_s3_storage import (
            sync_dataset_files_to_platform,
        )
        from hafnia.platform.datasets import get_or_create_dataset

        cfg = cfg or Config()
        get_or_create_dataset(dataset.info.dataset_name, cfg=cfg)

        sync_dataset_files_to_platform(
            dataset=dataset,
            sample_dataset=dataset_sample,
            interactive=interactive,
            allow_version_overwrite=allow_version_overwrite,
            cfg=cfg,
        )

        response = upload_dataset_details_to_platform(
            dataset=dataset,
            distribution_task_names=distribution_task_names,
            gallery_samples=gallery_samples,
            cfg=cfg,
        )

        return response

    def __eq__(self, value) -> bool:
        if not isinstance(value, HafniaDataset):
            return False

        if self.info != value.info:
            return False

        if not isinstance(self.samples, pl.DataFrame) or not isinstance(value.samples, pl.DataFrame):
            return False

        if not self.samples.equals(value.samples):
            return False
        return True


def _dataset_corrections(samples: pl.DataFrame, dataset_info: DatasetInfo) -> Tuple[pl.DataFrame, DatasetInfo]:
    format_version_of_dataset = Version(dataset_info.format_version)

    ## Backwards compatibility fixes for older dataset versions
    if format_version_of_dataset < Version("0.2.0"):
        samples = table_transformations.add_dataset_name_if_missing(samples, dataset_info.dataset_name)

        if "file_name" in samples.columns:
            samples = samples.rename({"file_name": SampleField.FILE_PATH})

        if SampleField.SAMPLE_INDEX not in samples.columns:
            samples = table_transformations.add_sample_index(samples)

        # Backwards compatibility: If tags-column doesn't exist, create it with empty lists
        if SampleField.TAGS not in samples.columns:
            tags_column: List[List[str]] = [[] for _ in range(len(samples))]  # type: ignore[annotation-unchecked]
            samples = samples.with_columns(pl.Series(tags_column, dtype=pl.List(pl.String)).alias(SampleField.TAGS))

        if SampleField.STORAGE_FORMAT not in samples.columns:
            samples = samples.with_columns(pl.lit(StorageFormat.IMAGE).alias(SampleField.STORAGE_FORMAT))

        if SampleField.SAMPLE_INDEX in samples.columns and samples[SampleField.SAMPLE_INDEX].dtype != pl.UInt64:
            samples = samples.cast({SampleField.SAMPLE_INDEX: pl.UInt64})

    if format_version_of_dataset <= Version("0.2.0"):
        if SampleField.BITMASKS in samples.columns and samples[SampleField.BITMASKS].dtype == pl.List(pl.Struct):
            struct_schema = samples.schema[SampleField.BITMASKS].inner
            struct_names = [f.name for f in struct_schema.fields]
            if "rleString" in struct_names:
                struct_names[struct_names.index("rleString")] = "rle_string"
                samples = samples.with_columns(
                    pl.col(SampleField.BITMASKS).list.eval(pl.element().struct.rename_fields(struct_names))
                )
    return samples, dataset_info


def check_hafnia_dataset_from_path(path_dataset: Path) -> None:
    dataset = HafniaDataset.from_path(path_dataset, check_for_images=True)
    dataset.check_dataset()


def available_dataset_versions_from_name(
    dataset_name: str,
) -> Dict[Version, "DatasetMetadataFilePaths"]:
    credentials: ResourceCredentials = get_read_credentials_by_name(dataset_name=dataset_name)
    return available_dataset_versions(credentials=credentials)


def available_dataset_versions(
    credentials: ResourceCredentials,
) -> Dict[Version, "DatasetMetadataFilePaths"]:
    envs = credentials.aws_credentials()
    bucket_prefix_sample_versions = f"{credentials.s3_uri()}/versions"
    all_s3_annotation_files = s5cmd_utils.list_bucket(bucket_prefix=bucket_prefix_sample_versions, append_envs=envs)
    available_versions = DatasetMetadataFilePaths.available_versions_from_files_list(all_s3_annotation_files)
    return available_versions


def select_version_from_available_versions(
    available_versions: Dict[Version, "DatasetMetadataFilePaths"],
    version: Optional[str],
) -> "DatasetMetadataFilePaths":
    if len(available_versions) == 0:
        raise ValueError("No versions were found in the dataset.")

    if version is None:
        str_versions = [str(v) for v in available_versions]
        raise ValueError(f"Version must be specified. Available versions: {str_versions}. ")

    if version == "latest":
        version_casted = max(available_versions)
        user_logger.info(f"'latest' version '{version_casted}' has been selected")
    else:
        version_casted = version_from_string(version, raise_error=True)

    if version_casted not in available_versions:
        raise ValueError(f"Selected version '{version}' not found in available versions: {available_versions}")

    return available_versions[version_casted]


def download_meta_dataset_files_from_version(
    resource_credentials: ResourceCredentials,
    version: Optional[str],
    path_dataset: Path,
) -> list[str]:
    envs = resource_credentials.aws_credentials()
    available_versions = available_dataset_versions(credentials=resource_credentials)
    metadata_files = select_version_from_available_versions(available_versions=available_versions, version=version)

    s3_files = metadata_files.as_list()
    path_dataset.mkdir(parents=True, exist_ok=True)
    local_paths = [(path_dataset / filename.split("/")[-1]).as_posix() for filename in s3_files]
    s5cmd_utils.fast_copy_files(
        src_paths=s3_files,
        dst_paths=local_paths,
        append_envs=envs,
        description="Downloading meta dataset files",
    )

    return local_paths


def download_or_get_dataset_path(
    dataset_name: str,
    version: Optional[str],
    cfg: Optional[Config] = None,
    path_datasets_folder: Optional[str] = None,
    force_redownload: bool = False,
    download_files: bool = True,
) -> Path:
    """Download or get the path of the dataset."""
    from hafnia.dataset.operations.dataset_s3_storage import sync_files_from_platform

    path_datasets = path_datasets_folder or utils.PATH_DATASETS
    path_dataset = Path(path_datasets) / dataset_name
    if not is_valid_version_string(version, allow_none=True, allow_latest=True):
        raise ValueError(
            f"Invalid version string: {version}. Should be a valid version (e.g. '0.1.0'), 'latest' or None."
        )

    # Only valid versions (e.g. '0.1.0', '1.0.0') can use local cache. Using either "latest"/None will always redownload
    if is_valid_version_string(version, allow_none=False, allow_latest=False):
        dataset_metadata_files = DatasetMetadataFilePaths.from_path(path_dataset)
        dataset_exists = dataset_metadata_files.exists(version=version, raise_error=False)
        if dataset_exists and not force_redownload:
            user_logger.info("Dataset found locally. Set 'force=True' or add `--force` flag with cli to re-download")
            return path_dataset

    cfg = cfg or Config()
    resource_credentials = get_read_credentials_by_name(dataset_name=dataset_name, cfg=cfg)
    if resource_credentials is None:
        raise ValueError(f"Failed to get read credentials for dataset '{dataset_name}' from the platform.")

    download_meta_dataset_files_from_version(
        resource_credentials=resource_credentials,
        version=version,
        path_dataset=path_dataset,
    )

    if not download_files:
        return path_dataset

    dataset = HafniaDataset.from_path(path_dataset, check_for_images=False)
    dataset = sync_files_from_platform(dataset, path_dataset, force_redownload=True)
    dataset.write_annotations(path_folder=path_dataset)  # Overwrite annotations as files have been re-downloaded
    return path_dataset

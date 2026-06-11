"""
Hafnia dataset transformations that takes and returns a HafniaDataset object.

All functions here will have a corresponding function in both the HafniaDataset class
and a corresponding RecipeTransform class in the `data_recipe/recipe_transformations.py` file.

This allows each function to be used in three ways:

```python
from hafnia.dataset.operations import dataset_transformations
from hafnia.dataset.hafnia_dataset import HafniaDataset
from hafnia.dataset.data_recipe.recipe_transformations import SplitByRatios

splits_by_ratios = {"train": 0.8, "val": 0.1, "test": 0.1}

# Option 1: Using the function directly
dataset = recipe_transformations.splits_by_ratios(dataset, split_ratios=splits_by_ratios)

# Option 2: Using the method of the HafniaDataset class
dataset = dataset.splits_by_ratios(split_ratios=splits_by_ratios)

# Option 3: Using the RecipeTransform class
serializable_transform = SplitByRatios(split_ratios=splits_by_ratios)
dataset = serializable_transform(dataset)
```

Tests will ensure that all functions in this file will have a corresponding function in the
HafniaDataset class and a RecipeTransform class in the `data_recipe/recipe_transformations.py` file and
that the signatures match.
"""

import json
import re
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)

import cv2
import more_itertools
import numpy as np
import polars as pl
from PIL import Image

from hafnia.dataset import dataset_helpers
from hafnia.dataset.dataset_names import (
    OPS_REMOVE_CLASS,
    PrimitiveField,
    SampleField,
    StorageFormat,
)
from hafnia.dataset.hafnia_dataset_types import ClassInfo, Sample, TaskInfo
from hafnia.dataset.operations.table_transformations import update_class_indices
from hafnia.dataset.primitives import get_primitive_type_from_string
from hafnia.dataset.primitives.primitive import Primitive
from hafnia.log import user_logger
from hafnia.utils import progress_bar, remove_duplicates_preserve_order

if TYPE_CHECKING:  # Using 'TYPE_CHECKING' to avoid circular imports during type checking
    from hafnia.dataset.hafnia_dataset import HafniaDataset


### Image transformations ###
class AnonymizeByPixelation:
    def __init__(self, resize_factor: float = 0.10):
        self.resize_factor = resize_factor

    def __call__(self, frame: np.ndarray, sample: Sample) -> np.ndarray:
        org_size = frame.shape[:2]
        frame = cv2.resize(frame, (0, 0), fx=self.resize_factor, fy=self.resize_factor)
        frame = cv2.resize(frame, org_size[::-1], interpolation=cv2.INTER_NEAREST)
        return frame


def transform_images(
    dataset: "HafniaDataset",
    transform: Callable[[np.ndarray, Sample], np.ndarray],
    path_output: Path,
    description: str = "Transform images",
    num_workers: int = 8,
    use_hash: bool = True,
    skip_existing: bool = True,
    subfolder_name: str = "transformed_images",
    select_columns: Optional[List[str]] = None,
) -> "HafniaDataset":
    """Apply an image transform to every sample in parallel and return a new dataset pointing at the outputs.

    Each sample's image is loaded, transformed via `transform(image, sample)`, written to
    `path_output/subfolder_name/`, and the dataset's `file_path` column is updated to the new path.
    The returned dataset shares `info` with the input.

    Args:
        transform: Callable that takes the raw image array and the corresponding `Sample` and
            returns the transformed image array.
        path_output: Output root folder; transformed images are written under
            `path_output/subfolder_name/`.
        description: Label shown on the progress bar.
        num_workers: Number of threads in the pool used to apply the transform.
        use_hash: If True, name output files by content hash for deterministic, deduplicated names.
            If False, reuse the input filename stem with a `.png` extension.
        skip_existing: If True, do not overwrite files that already exist on disk.
        subfolder_name: Subfolder under `path_output` to write images into.
        select_columns: Optional list of columns to keep on the resulting dataset's samples table
            (the rest are dropped before processing).
    """
    path_image_folder = path_output / subfolder_name
    path_image_folder.mkdir(parents=True, exist_ok=True)

    samples = dataset.samples

    if select_columns is not None:
        samples = samples.select(select_columns)

    n_samples = samples.height
    new_paths: List[Optional[str]] = [None] * n_samples

    def _process(i: int, sample_dict: dict) -> Tuple[int, str]:
        sample = Sample(**sample_dict)
        image = sample.read_image()
        image_transformed = transform(image, sample)
        if use_hash:
            new_path = dataset_helpers.save_image_with_hash_name(
                image_transformed,
                path_folder=path_image_folder,
                allow_skip=skip_existing,
            )
        else:
            if sample.file_path is None:
                raise ValueError(f"Sample {sample} does not have a file path.")
            filename = Path(sample.file_path).stem
            new_path = path_image_folder / f"{filename}.png"
            if not skip_existing or not new_path.exists():
                Image.fromarray(image_transformed).save(new_path, format="PNG", compress_level=1)
        if not Path(new_path).exists():
            raise FileNotFoundError(f"Transformed file {new_path} does not exist in the dataset.")
        return i, str(new_path)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_process, i, sd): i for i, sd in enumerate(samples.iter_rows(named=True))}
        for future in progress_bar(as_completed(futures), total=n_samples, description=description):
            i, path = future.result()
            new_paths[i] = path

    table = dataset.samples.with_columns(pl.Series(new_paths).alias(SampleField.FILE_PATH))
    return dataset.update_samples(table)


def convert_to_image_storage_format(
    dataset: "HafniaDataset",
    path_output_folder: Path,
    reextract_frames: bool,
    image_format: str = "png",
    append_transform: Optional[Callable[[np.ndarray, Sample], np.ndarray]] = None,
) -> "HafniaDataset":
    """Convert a video-based dataset to an image-based dataset by extracting referenced frames.

    For each sample with `storage_format == StorageFormat.VIDEO`, decodes the matching frame from
    the video file, optionally runs `append_transform(frame, sample)` to post-process it, writes
    the frame to disk in `image_format`, and rewrites the sample's `file_path` to point at the
    new image. Image-based samples are passed through unchanged. Returns the original dataset
    unchanged when no video-based samples are present.

    Args:
        path_output_folder: Output root; extracted frames are written under
            `path_output_folder/data/`.
        reextract_frames: If True, overwrite existing extracted frames; otherwise reuse on-disk
            frames whose paths already exist.
        image_format: Image extension to use for extracted frames (e.g. ``"png"``, ``"jpg"``).
        append_transform: Optional post-processing callable applied to each extracted frame.
    """
    from hafnia.dataset.hafnia_dataset import HafniaDataset

    path_images = (path_output_folder / "data").absolute()
    path_images.mkdir(parents=True, exist_ok=True)

    # Only video format dataset samples are processed
    video_based_samples = dataset.samples.filter(pl.col(SampleField.STORAGE_FORMAT) == StorageFormat.VIDEO)

    if video_based_samples.is_empty():
        user_logger.info("Dataset has no video-based samples. Returning dataset unchanged.")
        return dataset

    video_grouped_samples = list(video_based_samples.group_by(SampleField.FILE_PATH))
    n_videos = len(video_grouped_samples)
    update_list = []
    for i_video, ((path_video,), video_samples) in enumerate(video_grouped_samples):
        assert Path(path_video).exists(), (
            f"'{path_video}' not found. We expect the video to be downloaded to '{path_output_folder}'"
        )
        video = cv2.VideoCapture(str(path_video))

        video_samples = video_samples.sort(SampleField.COLLECTION_INDEX)
        for sample_dict in progress_bar(
            video_samples.iter_rows(named=True),
            total=video_samples.height,
            description=f"({i_video + 1}/{n_videos}) Extracting frames from '{Path(path_video).name}'",
        ):
            frame_number = sample_dict[SampleField.COLLECTION_INDEX]
            image_name = f"{Path(path_video).stem}_F{frame_number:06d}.{image_format}"
            path_image = path_images / image_name

            update_list.append(
                {
                    SampleField.SAMPLE_INDEX: sample_dict[SampleField.SAMPLE_INDEX],
                    SampleField.COLLECTION_ID: sample_dict[SampleField.COLLECTION_ID],
                    SampleField.COLLECTION_INDEX: frame_number,
                    SampleField.FILE_PATH: path_image.as_posix(),
                    SampleField.STORAGE_FORMAT: StorageFormat.IMAGE,
                }
            )
            if reextract_frames:
                path_image.unlink(missing_ok=True)
            if path_image.exists():
                continue

            video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame_org = video.read()
            if not ret:
                raise RuntimeError(f"Could not read frame {frame_number} from video '{path_video}'")

            if append_transform is not None:
                frame_org = append_transform(frame_org, Sample(**sample_dict))

            cv2.imwrite(str(path_image), frame_org)
    df_updates = pl.DataFrame(update_list)
    samples_as_images = dataset.samples.update(df_updates, on=[SampleField.COLLECTION_ID, SampleField.COLLECTION_INDEX])
    hafnia_dataset = HafniaDataset(samples=samples_as_images, info=dataset.info)

    return hafnia_dataset


def get_task_info_from_task_name_and_primitive(
    tasks: List[TaskInfo],
    task_name: Optional[str] = None,
    primitive: Union[None, str, Type[Primitive]] = None,
) -> TaskInfo:
    if len(tasks) == 0:
        raise ValueError("Dataset has no tasks defined.")

    tasks_str = "\n".join([f"\t{task.__repr__()}" for task in tasks])
    if task_name is None and primitive is None:
        if len(tasks) == 1:
            return tasks[0]
        else:
            raise ValueError(
                "For multiple tasks, you will need to specify 'task_name' or 'type_primitive' "
                "to return a unique task. The dataset contains the following tasks: \n" + tasks_str
            )

    if isinstance(primitive, str):
        primitive = get_primitive_type_from_string(primitive)

    tasks_filtered = tasks
    if primitive is None:
        tasks_filtered = [task for task in tasks if task.name == task_name]

        if len(tasks_filtered) == 0:
            raise ValueError(f"No task found with {task_name=}. Available tasks: \n {tasks_str}")

        unique_primitives = set(task.primitive for task in tasks_filtered)
        if len(unique_primitives) > 1:
            raise ValueError(
                f"Found multiple tasks with {task_name=} using different primitives {unique_primitives=}. "
                "Please specify the primitive type to make it unique. "
                f"The dataset contains the following tasks: \n {tasks_str}"
            )
        primitive = list(unique_primitives)[0]

    if task_name is None:
        tasks_filtered = [task for task in tasks if task.primitive == primitive]
        if len(tasks_filtered) == 0:
            raise ValueError(f"No task found with {primitive=}. Available tasks: \n {tasks_str}")

        unique_task_names = set(task.name for task in tasks_filtered)
        if len(unique_task_names) > 1:
            raise ValueError(
                f"Found multiple tasks with {primitive=} using different task names {unique_task_names=}. "
                "Please specify the 'task_name' to make it unique."
                f"The dataset contains the following tasks: \n {tasks_str}"
            )
        task_name = list(unique_task_names)[0]

    tasks_filtered = [task for task in tasks_filtered if task.primitive == primitive and task.name == task_name]
    if len(tasks_filtered) == 0:
        raise ValueError(f"No task found with {task_name=} and {primitive=}. Available tasks: \n {tasks_str}")

    if len(tasks_filtered) > 1:
        raise ValueError(
            f"Multiple tasks found with {task_name=} and {primitive=}. "
            f"This should never happen. The dataset contains the following tasks: \n {tasks_str}"
        )
    task = tasks_filtered[0]
    return task


def class_mapper(
    dataset: "HafniaDataset",
    class_mapping: Union[Dict[str, str], List[Tuple[str, str]]],
    method: str = "strict",
    primitive: Optional[Type[Primitive]] = None,
    task_name: Optional[str] = None,
) -> "HafniaDataset":
    from hafnia.dataset.hafnia_dataset import HafniaDataset

    if isinstance(class_mapping, list):
        class_mapping = dict(class_mapping)

    allowed_methods = ("strict", "remove_undefined", "keep_undefined")
    if method not in allowed_methods:
        raise ValueError(f"Method '{method}' is not recognized. Allowed methods are: {allowed_methods}")

    task = dataset.info.get_task_by_task_name_and_primitive(task_name=task_name, primitive=primitive)
    current_names = task.get_class_names() or []

    # Expand wildcard mappings e.g. {"Vehicle.*": "Vehicle"} to {"Vehicle.Car": "Vehicle", "Vehicle.Bus": "Vehicle"}
    class_mapping = expand_class_mapping(class_mapping, current_names)

    non_existing_mapping_names = set(class_mapping) - set(current_names)
    if len(non_existing_mapping_names) > 0:
        raise ValueError(
            f"The specified class mapping contains class names {list(non_existing_mapping_names)} "
            f"that do not exist in the dataset task '{task.name}'. "
            f"Available class names: {current_names}"
        )

    missing_class_names = [c for c in current_names if c not in class_mapping]  # List-comprehension to preserve order
    class_mapping = class_mapping.copy()
    if method == "strict":
        pass  # Continue to strict mapping below
    elif method == "remove_undefined":
        for missing_class_name in missing_class_names:
            class_mapping[missing_class_name] = OPS_REMOVE_CLASS
    elif method == "keep_undefined":
        for missing_class_name in missing_class_names:
            class_mapping[missing_class_name] = missing_class_name
    else:
        raise ValueError(f"Method '{method}' is not recognized. Allowed methods are: {allowed_methods}")

    missing_class_names = [c for c in current_names if c not in class_mapping]
    if len(missing_class_names) > 0:
        error_msg = f"""\
        The specified class mapping is not a strict mapping - meaning that all class names have not 
        been mapped to a new class name.
        In the current mapping, the following classes {list(missing_class_names)} have not been mapped.
        The currently specified mapping is:
        {json.dumps(class_mapping, indent=2)}
        A strict mapping will replace all old class names (dictionary keys) to new class names (dictionary values).
        Please update the mapping to include all class names from the dataset task '{task.name}'.
        To keep class map to the same name e.g. 'person' = 'person' 
        or remove class by using the '__REMOVE__' key, e.g. 'person': '__REMOVE__'."""
        raise ValueError(textwrap.dedent(error_msg))

    new_class_names = remove_duplicates_preserve_order(class_mapping.values())

    if OPS_REMOVE_CLASS in new_class_names:
        # Move __REMOVE__ to the end of the list if it exists
        new_class_names.append(new_class_names.pop(new_class_names.index(OPS_REMOVE_CLASS)))

    samples = dataset.samples
    samples_updated = samples.with_columns(
        pl.col(task.primitive.column_name())
        .list.eval(
            pl.element().struct.with_fields(
                pl.when(pl.field(PrimitiveField.TASK_NAME) == task.name)
                .then(pl.field(PrimitiveField.CLASS_NAME).replace_strict(class_mapping, default="Missing"))
                .otherwise(pl.field(PrimitiveField.CLASS_NAME))
                .alias(PrimitiveField.CLASS_NAME)
            )
        )
        .alias(task.primitive.column_name())
    )

    if OPS_REMOVE_CLASS in new_class_names:  # Remove class_names that are mapped to REMOVE_CLASS
        samples_updated = samples_updated.with_columns(
            pl.col(task.primitive.column_name())
            .list.filter(pl.element().struct.field(PrimitiveField.CLASS_NAME) != OPS_REMOVE_CLASS)
            .alias(task.primitive.column_name())
        )

        new_class_names = [c for c in new_class_names if c != OPS_REMOVE_CLASS]

    # Update dataset info with new class names
    name_and_classes_new: Dict[str, List[ClassInfo]] = {new_name: [] for new_name in new_class_names}
    name_and_classes_old = {klass.name: klass.attributes for klass in task.classes or []}
    classes = []
    for new_name in name_and_classes_new:
        previous_names = [name_from for name_from, name_to in class_mapping.items() if name_to == new_name]
        attributes_combined: List[Any] = []
        for previous_name in previous_names:
            attributes_combined.extend(name_and_classes_old[previous_name] or [])
        attributes = None if len(attributes_combined or []) == 0 else attributes_combined
        classes.append(ClassInfo(name=new_name, attributes=attributes))
    new_task = task.model_copy(deep=True)
    new_task.classes = classes
    dataset_info = dataset.info.replace_task(old_task=task, new_task=new_task)

    # Update class indices to match new class names
    samples_updated = update_class_indices(samples_updated, new_task)

    return HafniaDataset(info=dataset_info, samples=samples_updated)


def expand_class_mapping(wildcard_mapping: Dict[str, str], class_names: List[str]) -> Dict[str, str]:
    """
    Expand a wildcard class mapping to a full explicit mapping.

    This function takes a mapping that may contain wildcard patterns (using '*')
    and expands them to match actual class names from a dataset. Exact matches
    take precedence over wildcard patterns.

    Examples:
        >>> from hafnia.dataset.dataset_names import OPS_REMOVE_CLASS
        >>> wildcard_mapping = {
        ...     "Person": "Person",
        ...     "Vehicle.*": "Vehicle",
        ...     "Vehicle.Trailer": OPS_REMOVE_CLASS
        ... }
        >>> class_names = [
        ...     "Person", "Vehicle.Car", "Vehicle.Trailer", "Vehicle.Bus", "Animal.Dog"
        ... ]
        >>> result = expand_wildcard_mapping(wildcard_mapping, class_names)
        >>> print(result)
        {
            "Person": "Person",
            "Vehicle.Car": "Vehicle",
            "Vehicle.Trailer": OPS_REMOVE_CLASS,  # Exact match overrides wildcard
            "Vehicle.Bus": "Vehicle",
            # Note: "Animal.Dog" is not included as it doesn't match any pattern
        }
    """
    expanded_mapping = {}
    for match_pattern, mapping_value in wildcard_mapping.items():
        if "*" in match_pattern:
            # Convert wildcard pattern to regex: Escape special regex characters except *, then replace * with .*
            regex_pattern = re.escape(match_pattern).replace("\\*", ".*")
            class_names_matched = [cn for cn in class_names if re.fullmatch(regex_pattern, cn)]
            expanded_mapping.update({cn: mapping_value for cn in class_names_matched})
        else:
            expanded_mapping.pop(match_pattern, None)
            expanded_mapping[match_pattern] = mapping_value
    return expanded_mapping


def rename_task(
    dataset: "HafniaDataset",
    old_task_name: str,
    new_task_name: str,
) -> "HafniaDataset":
    from hafnia.dataset.hafnia_dataset import HafniaDataset

    old_task = dataset.info.get_task_by_name(task_name=old_task_name)
    new_task = old_task.model_copy(deep=True)
    new_task.name = new_task_name
    samples = dataset.samples.with_columns(
        pl.col(old_task.primitive.column_name())
        .list.eval(
            pl.element().struct.with_fields(
                pl.field(PrimitiveField.TASK_NAME).replace(old_task.name, new_task.name).alias(PrimitiveField.TASK_NAME)
            )
        )
        .alias(new_task.primitive.column_name())
    )

    dataset_info = dataset.info.replace_task(old_task=old_task, new_task=new_task)
    return HafniaDataset(info=dataset_info, samples=samples)


def select_samples_by_class_name(
    dataset: "HafniaDataset",
    name: Union[List[str], str],
    task_name: Optional[str] = None,
    primitive: Optional[Type[Primitive]] = None,
) -> "HafniaDataset":
    task, class_names = _validate_inputs_select_samples_by_class_name(
        dataset=dataset,
        name=name,
        task_name=task_name,
        primitive=primitive,
    )

    samples = dataset.samples.filter(
        pl.col(task.primitive.column_name())
        .list.eval(
            pl.element().struct.field(PrimitiveField.CLASS_NAME).is_in(class_names)
            & (pl.element().struct.field(PrimitiveField.TASK_NAME) == task.name)
        )
        .list.any()
    )

    dataset_updated = dataset.update_samples(samples)
    return dataset_updated


def drop_samples_by_class_name(
    dataset: "HafniaDataset",
    name: Union[List[str], str],
    task_name: Optional[str] = None,
    primitive: Optional[Type[Primitive]] = None,
    drop_classes_from_task_info: bool = True,
) -> "HafniaDataset":
    from hafnia.dataset.hafnia_dataset import HafniaDataset

    task, class_names = _validate_inputs_select_samples_by_class_name(
        dataset=dataset,
        name=name,
        task_name=task_name,
        primitive=primitive,
    )

    contains_dropped_class = (
        pl.col(task.primitive.column_name())
        .list.eval(
            pl.element().struct.field(PrimitiveField.CLASS_NAME).is_in(class_names)
            & (pl.element().struct.field(PrimitiveField.TASK_NAME) == task.name)
        )
        .list.any()
    )
    samples = dataset.samples.filter(~contains_dropped_class)

    if not drop_classes_from_task_info:
        return dataset.update_samples(samples)

    new_classes = [c for c in (task.classes or []) if c.name not in class_names]
    new_task = TaskInfo(primitive=task.primitive, name=task.name, classes=new_classes)
    dataset_info = dataset.info.replace_task(old_task=task, new_task=new_task)
    if new_classes:
        samples = update_class_indices(samples, new_task)
    return HafniaDataset(info=dataset_info, samples=samples)


def _validate_inputs_select_samples_by_class_name(
    dataset: "HafniaDataset",
    name: Union[List[str], str],
    task_name: Optional[str] = None,
    primitive: Optional[Type[Primitive]] = None,
) -> Tuple[TaskInfo, List[str]]:
    if isinstance(name, str):
        name = [name]
    names = list(name)

    # Check that specified names are available in at least one of the tasks
    available_names_across_tasks = set(more_itertools.flatten([t.get_class_names() or [] for t in dataset.info.tasks]))
    missing_class_names_across_tasks = set(names) - available_names_across_tasks
    if len(missing_class_names_across_tasks) > 0:
        raise ValueError(
            f"The specified names {list(names)} have not been found in any of the tasks. "
            f"Available class names: {available_names_across_tasks}"
        )

    # Auto infer task if task_name and primitive are not provided
    if task_name is None and primitive is None:
        tasks_with_names = [t for t in dataset.info.tasks if set(names).issubset(t.get_class_names() or [])]
        if len(tasks_with_names) == 0:
            raise ValueError(
                f"The specified names {names} have not been found in any of the tasks. "
                f"Available class names: {available_names_across_tasks}"
            )
        if len(tasks_with_names) > 1:
            raise ValueError(
                f"Found multiple tasks containing the specified names {names}. "
                f"Specify either 'task_name' or 'primitive' to only select from one task. "
                f"Tasks containing all provided names: {[t.name for t in tasks_with_names]}"
            )

        task = tasks_with_names[0]

    else:
        task = get_task_info_from_task_name_and_primitive(
            tasks=dataset.info.tasks,
            task_name=task_name,
            primitive=primitive,
        )

    task_class_names = set(task.get_class_names() or [])
    missing_class_names = set(names) - task_class_names
    if len(missing_class_names) > 0:
        raise ValueError(
            f"The specified names {list(missing_class_names)} have not been found for the '{task.name}' task. "
            f"Available class names: {task_class_names}"
        )

    return task, names

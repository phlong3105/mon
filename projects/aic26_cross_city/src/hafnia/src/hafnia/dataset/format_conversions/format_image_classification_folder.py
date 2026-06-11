import shutil
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

import more_itertools
import polars as pl
from PIL import Image

from hafnia.dataset.dataset_names import PrimitiveField, SampleField
from hafnia.dataset.format_conversions.format_helpers import SplitNameAndPath, get_splits_from_folder
from hafnia.dataset.hafnia_dataset_types import DatasetInfo, Sample, TaskInfo
from hafnia.dataset.primitives import Classification
from hafnia.utils import is_image_file, progress_bar

if TYPE_CHECKING:  # Using 'TYPE_CHECKING' to avoid circular imports during type checking
    from hafnia.dataset.hafnia_dataset import HafniaDataset


DEFAULT_DATASET_NAME = "ImageClassificationDataset"


def from_image_classification_folder(
    path_folder: Path,
    n_samples: Optional[int] = None,
    dataset_name: str = DEFAULT_DATASET_NAME,
) -> "HafniaDataset":
    """Import an ImageFolder-style classification dataset as a `HafniaDataset`.

    Expects `path_folder` to contain one subfolder per split (e.g. `train/`, `val/`, `test/`),
    each of which contains one subfolder per class with images inside. Class names are unioned
    across splits and assigned consistent indices.

    Args:
        path_folder: Root folder with split-named subfolders.
        n_samples: Optional cap on the total number of samples; samples are interleaved across
            classes so the cap stays balanced across classes.
        dataset_name: Name to assign to the resulting dataset.
    """
    list_split_name_and_path = get_splits_from_folder(path_folder)

    dataset = from_image_classification_folder_by_split_paths(
        n_samples=n_samples,
        dataset_name=dataset_name,
        list_split_paths=list_split_name_and_path,
    )
    return dataset


def from_image_classification_folder_by_split_paths(
    list_split_paths: List[SplitNameAndPath],
    dataset_name: str = DEFAULT_DATASET_NAME,
    n_samples: Optional[int] = None,
) -> "HafniaDataset":
    from hafnia.dataset.hafnia_dataset import HafniaDataset

    class_names_dub = more_itertools.collapse([class_names_from_folder(split.path) for split in list_split_paths])
    class_names = sorted(list(set(class_names_dub)))  # Remove duplicates and sort for determinism

    if n_samples is not None:
        n_samples = n_samples // len(list_split_paths)  # Divide samples evenly across splits
    datasets_per_split = []
    for split in list_split_paths:
        dataset_split = from_image_classification_split_folder(
            path_folder=split.path,
            split=split.name,
            n_samples=n_samples,
            class_names=class_names,
            dataset_name=dataset_name,
        )

        datasets_per_split.append(dataset_split)

    dataset = HafniaDataset.from_merger(datasets=datasets_per_split)
    dataset.info.dataset_name = dataset_name
    return dataset


def from_image_classification_split_folder(
    path_folder: Path,
    split: str,
    dataset_name: str,
    n_samples: Optional[int] = None,
    class_names: Optional[List[str]] = None,
) -> "HafniaDataset":
    from hafnia.dataset.hafnia_dataset import HafniaDataset

    if class_names is None:
        class_names = class_names_from_folder(path_folder)

    # Gather all image paths per class
    path_images_per_class: List[List[Path]] = []
    for class_name in class_names:
        path_class_folder = path_folder / class_name
        per_class_images = []
        for path_image in list(path_class_folder.rglob("*.*")):
            if is_image_file(path_image):
                per_class_images.append(path_image)
        path_images_per_class.append(sorted(per_class_images))

    # Interleave to ensure classes are balanced in the output dataset for n_samples < total
    path_images = list(more_itertools.interleave_longest(*path_images_per_class))

    if n_samples is not None:
        path_images = path_images[:n_samples]

    samples = []
    for path_image_org in progress_bar(
        path_images, description="Convert 'image classification' dataset to Hafnia Dataset"
    ):
        class_name = path_image_org.parent.name

        read_image = Image.open(path_image_org)
        width, height = read_image.size

        classifications = [Classification(class_name=class_name, class_idx=class_names.index(class_name))]
        sample = Sample(
            file_path=str(path_image_org.absolute()),
            width=width,
            height=height,
            split=split,
            classifications=classifications,
        )
        samples.append(sample)

    dataset_info = DatasetInfo(
        dataset_name=dataset_name,
        tasks=[TaskInfo.from_class_names(primitive=Classification, class_names=class_names)],
    )

    hafnia_dataset = HafniaDataset.from_samples_list(samples_list=samples, info=dataset_info)
    return hafnia_dataset


def to_image_classification_folder(
    dataset: "HafniaDataset",
    path_output: Path,
    task_name: Optional[str] = None,
    clean_folder: bool = False,
) -> List[Path]:
    """Export a classification `HafniaDataset` to the ImageFolder layout (one folder per class).

    Creates `path_output/<split>/<class_name>/` and copies each sample's image to the matching
    folder. Requires a `Classification` task in the dataset.

    Args:
        dataset: Dataset to export. Must contain a `Classification` task.
        path_output: Output root folder; one subfolder per split is created beneath it.
        task_name: Optional task name disambiguation when the dataset has multiple
            `Classification` tasks.
        clean_folder: If True, delete the contents of each split folder before writing.

    Returns:
        The list of split-level output folders that were written.
    """
    task = dataset.info.get_task_by_task_name_and_primitive(task_name=task_name, primitive=Classification)

    split_names = dataset.samples[SampleField.SPLIT].unique().to_list()
    split_paths = []
    for split_name in split_names:
        dataset_split = dataset.create_split_dataset(split_name)
        split_path = to_image_classification_split_folder(
            dataset=dataset_split,
            path_output_split=path_output / split_name,
            task_name=task.name,
            clean_folder=clean_folder,
        )
        split_paths.append(split_path)

    return split_paths


def to_image_classification_split_folder(
    dataset: "HafniaDataset",
    path_output_split: Path,
    task_name: Optional[str] = None,
    clean_folder: bool = False,
) -> Path:
    task = dataset.info.get_task_by_task_name_and_primitive(task_name=task_name, primitive=Classification)

    samples = dataset.samples.with_columns(
        pl.col(task.primitive.column_name())
        .list.filter(pl.element().struct.field(PrimitiveField.TASK_NAME) == task.name)
        .alias(task.primitive.column_name())
    )

    classification_counts = samples[task.primitive.column_name()].list.len()
    has_no_classification_samples = (classification_counts == 0).sum()
    if has_no_classification_samples > 0:
        raise ValueError(f"Some samples do not have a classification for task '{task.name}'.")

    has_multi_classification_samples = (classification_counts > 1).sum()
    if has_multi_classification_samples > 0:
        raise ValueError(f"Some samples have multiple classifications for task '{task.name}'.")

    if clean_folder:
        shutil.rmtree(path_output_split, ignore_errors=True)
    path_output_split.mkdir(parents=True, exist_ok=True)

    description = "Export Hafnia Dataset to directory tree"
    for sample_dict in progress_bar(samples.iter_rows(named=True), total=len(samples), description=description):
        classifications = sample_dict[task.primitive.column_name()]
        if len(classifications) != 1:
            raise ValueError("Each sample should have exactly one classification.")
        classification = classifications[0]
        class_name = classification[PrimitiveField.CLASS_NAME].replace("/", "_")  # Avoid issues with subfolders
        path_class_folder = path_output_split / class_name
        path_class_folder.mkdir(parents=True, exist_ok=True)

        path_image_org = Path(sample_dict[SampleField.FILE_PATH])
        path_image_new = path_class_folder / path_image_org.name
        shutil.copy2(path_image_org, path_image_new)

    return path_output_split


def class_names_from_folder(path_folder: Path) -> List[str]:
    class_folder_paths = [path for path in path_folder.iterdir() if path.is_dir()]
    class_names = sorted([folder.name for folder in class_folder_paths])  # Sort for determinism
    return class_names

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

from PIL import Image

from hafnia.dataset import primitives
from hafnia.dataset.dataset_names import SampleField, SplitName
from hafnia.dataset.format_conversions import format_helpers
from hafnia.dataset.hafnia_dataset_types import DatasetInfo, Sample, TaskInfo
from hafnia.utils import progress_bar

if TYPE_CHECKING:  # Using 'TYPE_CHECKING' to avoid circular imports during type checking
    from hafnia.dataset.hafnia_dataset import HafniaDataset

FILENAME_YOLO_CLASS_NAMES = "obj.names"
FILENAME_YOLO_IMAGES_TXT = "images.txt"


def get_image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as img:
        return img.size  # (width, height)


@dataclass
class YoloSplitPaths:
    split: str
    path_root: Path
    path_images_txt: Path
    path_class_names: Path

    def check_paths(self):
        if not self.path_root.exists():
            raise FileNotFoundError(f"YOLO dataset root path not found at '{self.path_root.resolve()}'")
        if not self.path_images_txt.exists():
            raise FileNotFoundError(f"File with images not found at '{self.path_images_txt.resolve()}'")
        if not self.path_class_names.exists():
            raise FileNotFoundError(f"File with class names not found at '{self.path_class_names.resolve()}'")


def from_yolo_format(
    path_dataset: Path,
    dataset_name: str = "yolo-dataset",
    filename_class_names: str = FILENAME_YOLO_CLASS_NAMES,
    filename_images_txt: str = FILENAME_YOLO_IMAGES_TXT,
) -> "HafniaDataset":
    """Import a YOLO (Darknet) formatted dataset directory as a `HafniaDataset`.

    Discovers split subfolders under `path_dataset`, reads the shared class-names file and a
    per-split image list, and assembles them into a single `HafniaDataset` with a `Bbox` task.

    Args:
        path_dataset: Root folder containing one subfolder per split.
        dataset_name: Name to assign to the resulting dataset.
        filename_class_names: Filename of the class-names file at the dataset root.
        filename_images_txt: Filename of the images list inside each split folder.
    """
    per_split_paths: List[YoloSplitPaths] = get_split_definitions_for_coco_dataset_formats(
        path_dataset=path_dataset,
        filename_class_names=filename_class_names,
        filename_images_txt=filename_images_txt,
    )

    hafnia_dataset = from_yolo_format_by_split_paths(splits=per_split_paths, dataset_name=dataset_name)
    return hafnia_dataset


def from_yolo_format_by_split_paths(splits: List[YoloSplitPaths], dataset_name: str) -> "HafniaDataset":
    from hafnia.dataset.hafnia_dataset import HafniaDataset

    dataset_splits = []
    for split_paths in splits:
        dataset_split = dataset_split_from_yolo_format(split_paths=split_paths, dataset_name=dataset_name)
        dataset_splits.append(dataset_split)

    hafnia_dataset = HafniaDataset.from_merger(dataset_splits)
    return hafnia_dataset


def get_split_definitions_for_coco_dataset_formats(
    path_dataset: Path,
    filename_class_names: str = FILENAME_YOLO_CLASS_NAMES,
    filename_images_txt: str = FILENAME_YOLO_IMAGES_TXT,
) -> List[YoloSplitPaths]:
    splits = []

    for split_def in format_helpers.get_splits_from_folder(path_dataset):
        split_path = YoloSplitPaths(
            split=split_def.name,
            path_root=split_def.path,
            path_images_txt=split_def.path / filename_images_txt,
            path_class_names=path_dataset / filename_class_names,
        )
        splits.append(split_path)

    return splits


def dataset_split_from_yolo_format(
    split_paths: YoloSplitPaths,
    dataset_name: str,
) -> "HafniaDataset":
    """
    Imports a YOLO (Darknet) formatted dataset as a HafniaDataset.
    """
    from hafnia.dataset.hafnia_dataset import HafniaDataset

    path_class_names = split_paths.path_class_names
    if split_paths.split not in SplitName.all_split_names():
        raise ValueError(f"Invalid split name: {split_paths.split}. Must be one of {SplitName.all_split_names()}")
    if not path_class_names.exists():
        raise FileNotFoundError(f"File with class names not found at '{path_class_names.resolve()}'.")

    class_names_text = path_class_names.read_text()
    if class_names_text.strip() == "":
        raise ValueError(f"File with class names not found at '{path_class_names.resolve()}' is empty")

    class_names = [class_name for class_name in class_names_text.splitlines() if class_name.strip() != ""]

    if len(class_names) == 0:
        raise ValueError(f"File with class names not found at '{path_class_names.resolve()}' has no class names")

    path_images_txt = split_paths.path_images_txt
    if not path_images_txt.exists():
        raise FileNotFoundError(f"File with images not found at '{path_images_txt.resolve()}'")

    images_txt_text = path_images_txt.read_text()
    if len(images_txt_text.strip()) == 0:
        raise ValueError(f"File is empty at '{path_images_txt.resolve()}'")

    image_paths_raw = [line.strip() for line in images_txt_text.splitlines()]

    samples: List[Sample] = []
    for image_path_raw in progress_bar(image_paths_raw, description=f"Import YOLO '{split_paths.split}' split"):
        path_image = split_paths.path_root / image_path_raw
        if not path_image.exists():
            raise FileNotFoundError(f"File with image not found at '{path_image.resolve()}'")
        width, height = get_image_size(path_image)

        path_label = path_image.with_suffix(".txt")
        if not path_label.exists():
            raise FileNotFoundError(f"File with labels not found at '{path_label.resolve()}'")

        boxes: List[primitives.Bbox] = []
        bbox_strings = path_label.read_text().splitlines()
        for bbox_string in bbox_strings:
            parts = bbox_string.strip().split()
            if len(parts) != 5:
                raise ValueError(f"Invalid bbox format in file {path_label.resolve()}: {bbox_string}")

            class_idx = int(parts[0])
            x_center, y_center, bbox_width, bbox_height = (float(value) for value in parts[1:5])

            top_left_x = x_center - bbox_width / 2
            top_left_y = y_center - bbox_height / 2

            bbox = primitives.Bbox(
                top_left_x=top_left_x,
                top_left_y=top_left_y,
                width=bbox_width,
                height=bbox_height,
                class_idx=class_idx,
                class_name=class_names[class_idx] if 0 <= class_idx < len(class_names) else None,
            )
            boxes.append(bbox)

        sample = Sample(
            file_path=path_image.absolute().as_posix(),
            height=height,
            width=width,
            split=split_paths.split,
            bboxes=boxes,
        )
        samples.append(sample)

    tasks = [TaskInfo.from_class_names(primitive=primitives.Bbox, class_names=class_names)]
    info = DatasetInfo(dataset_name=dataset_name, tasks=tasks)
    hafnia_dataset = HafniaDataset.from_samples_list(samples, info=info)
    return hafnia_dataset


def to_yolo_format(
    dataset: "HafniaDataset",
    path_output: Path,
    task_name: Optional[str] = None,
    filename_images_txt: str = FILENAME_YOLO_IMAGES_TXT,
    filename_class_names: str = FILENAME_YOLO_CLASS_NAMES,
) -> List[YoloSplitPaths]:
    """Export a `HafniaDataset` to the YOLO (Darknet) on-disk format.

    Writes one subfolder per split under `path_output`, each containing the YOLO-style label
    text files and an `images.txt` listing, plus a shared class-names file at the dataset root.

    Args:
        dataset: Dataset to export. Must contain a `Bbox` task.
        path_output: Output root folder; one subfolder per split is created beneath it.
        task_name: Optional task name disambiguation when the dataset has multiple `Bbox` tasks.
        filename_images_txt: Filename of the per-split image listing.
        filename_class_names: Filename of the shared class-names file at the dataset root.

    Returns:
        A list of `YoloSplitPaths` describing the files written for each split.
    """

    split_names = dataset.samples[SampleField.SPLIT].unique().to_list()

    per_split_paths: List[YoloSplitPaths] = []
    for split_name in split_names:
        dataset_split = dataset.create_split_dataset(split_name)

        yolo_split_paths = YoloSplitPaths(
            split=split_name,
            path_root=path_output / split_name,
            path_images_txt=path_output / split_name / filename_images_txt,
            path_class_names=path_output / filename_class_names,
        )

        to_yolo_split_format(
            dataset=dataset_split,
            split_paths=yolo_split_paths,
            task_name=task_name,
        )
        per_split_paths.append(yolo_split_paths)
    return per_split_paths


def to_yolo_split_format(
    dataset: "HafniaDataset",
    split_paths: YoloSplitPaths,
    task_name: Optional[str],
):
    """Exports a HafniaDataset as YOLO (Darknet) format."""

    bbox_task = dataset.info.get_task_by_task_name_and_primitive(task_name=task_name, primitive=primitives.Bbox)

    class_names = bbox_task.get_class_names() or []
    if len(class_names) == 0:
        raise ValueError(
            f"Hafnia dataset task '{bbox_task.name}' has no class names defined. This is required for YOLO export."
        )
    split_paths.path_root.mkdir(parents=True, exist_ok=True)
    split_paths.path_class_names.parent.mkdir(parents=True, exist_ok=True)
    split_paths.path_class_names.write_text("\n".join(class_names))

    path_data_folder = split_paths.path_root / "data"
    path_data_folder.mkdir(parents=True, exist_ok=True)
    image_paths: List[str] = []
    for sample_dict in dataset:
        sample = Sample(**sample_dict)
        if sample.file_path is None:
            raise ValueError("Sample has no file_path defined.")
        path_image_src = Path(sample.file_path)
        path_image_dst = path_data_folder / path_image_src.name
        shutil.copy2(path_image_src, path_image_dst)
        image_paths.append(path_image_dst.relative_to(split_paths.path_root).as_posix())
        path_label = path_image_dst.with_suffix(".txt")
        bboxes = sample.bboxes or []
        bbox_strings = [bbox_to_yolo_format(bbox) for bbox in bboxes]
        path_label.write_text("\n".join(bbox_strings))

    split_paths.path_images_txt.parent.mkdir(parents=True, exist_ok=True)
    split_paths.path_images_txt.write_text("\n".join(image_paths))


def bbox_to_yolo_format(bbox: primitives.Bbox) -> str:
    """
    From hafnia bbox to yolo bbox string conversion
    Both yolo and hafnia use normalized coordinates [0, 1]
        Hafnia: top_left_x, top_left_y, width, height
        Yolo (darknet): "<object-class> <x_center> <y_center> <width> <height>"
    Example (3 bounding boxes):
        1 0.716797 0.395833 0.216406 0.147222
        0 0.687109 0.379167 0.255469 0.158333
        1 0.420312 0.395833 0.140625 0.166667
    """
    x_center = bbox.top_left_x + bbox.width / 2
    y_center = bbox.top_left_y + bbox.height / 2
    return f"{bbox.class_idx} {x_center} {y_center} {bbox.width} {bbox.height}"

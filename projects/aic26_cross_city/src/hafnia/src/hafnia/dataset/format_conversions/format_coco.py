import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import polars as pl
from pycocotools import mask as coco_utils

from hafnia.dataset import license_types
from hafnia.dataset.dataset_names import SampleField, SplitName
from hafnia.dataset.format_conversions import format_coco, format_helpers
from hafnia.utils import progress_bar

if TYPE_CHECKING:  # Using 'TYPE_CHECKING' to avoid circular imports during type checking
    from hafnia.dataset.hafnia_dataset import HafniaDataset

from hafnia.dataset.hafnia_dataset_types import Attribution, DatasetInfo, License, Sample, TaskInfo
from hafnia.dataset.primitives import Bbox, Bitmask
from hafnia.log import user_logger

COCO_KEY_FILE_NAME = "file_name"

HAFNIA_TO_ROBOFLOW_SPLIT_NAME = {
    SplitName.TRAIN: "train",
    SplitName.VAL: "valid",
    SplitName.TEST: "test",
}
ROBOFLOW_ANNOTATION_FILE_NAME = "_annotations.coco.json"


@dataclass
class CocoSplitPaths:
    split: str
    path_images: Path
    path_instances_json: Path


def from_coco_format(
    path_dataset: Path,
    coco_format_type: str = "roboflow",
    max_samples: Optional[int] = None,
    dataset_name: str = "coco-2017",
):
    """Import a COCO-formatted dataset as a `HafniaDataset`.

    Resolves split-level image folders and `instances.json` files according to the chosen layout
    and merges them into one dataset. Currently supports the Roboflow on-disk layout (one
    subfolder per split, each containing a `_annotations.coco.json` instances file).

    Args:
        path_dataset: Root folder containing the COCO splits.
        coco_format_type: Layout type; only ``"roboflow"`` is currently supported.
        max_samples: Optional cap on the total number of samples (split evenly across splits).
        dataset_name: Name to assign to the resulting dataset.
    """
    split_definitions = get_split_paths_for_coco_dataset_formats(
        path_dataset=path_dataset, coco_format_type=coco_format_type
    )

    hafnia_dataset = from_coco_dataset_by_split_definitions(
        split_definitions=split_definitions,
        max_samples=max_samples,
        dataset_name=dataset_name,
    )

    return hafnia_dataset


def get_split_paths_for_coco_dataset_formats(
    path_dataset: Path,
    coco_format_type: str,
) -> List[CocoSplitPaths]:
    splits = []
    if coco_format_type == "roboflow":
        for split_def in format_helpers.get_splits_from_folder(path_dataset):
            splits.append(
                CocoSplitPaths(
                    split=split_def.name,
                    path_images=split_def.path,
                    path_instances_json=split_def.path / ROBOFLOW_ANNOTATION_FILE_NAME,
                )
            )
        return splits

    raise ValueError(f"The specified '{coco_format_type=}' is not supported.")


def from_coco_dataset_by_split_definitions(
    split_definitions: List[CocoSplitPaths],
    max_samples: Optional[int],
    dataset_name: str,
) -> "HafniaDataset":
    from hafnia.dataset.hafnia_dataset import HafniaDataset

    if max_samples is None:
        max_samples_per_split = None
    else:
        max_samples_per_split = max_samples // len(split_definitions)
    samples = []
    tasks: List[TaskInfo] = []
    for split_definition in split_definitions:
        if split_definition.path_instances_json is None or not split_definition.path_instances_json.exists():
            raise FileNotFoundError(
                f"Expected COCO dataset files not found for split '{split_definition.split}'. "
                f"Label file doesn't exist: {split_definition.path_instances_json}"
            )
        if not split_definition.path_images.exists():
            raise FileNotFoundError(
                f"Expected COCO dataset files not found for split '{split_definition.split}'. "
                f"Images folder doesn't exist: {split_definition.path_images}"
            )

        samples_in_split, tasks_in_split = coco_format_folder_with_split_to_hafnia_samples(
            path_label_file=split_definition.path_instances_json,
            max_samples_per_split=max_samples_per_split,
            path_images=split_definition.path_images,
            split_name=split_definition.split,
        )

        for task_in_split in tasks_in_split:
            matching_tasks = [task for task in tasks if task.name == task_in_split.name]

            add_missing_task = len(matching_tasks) == 0
            if add_missing_task:
                tasks.append(task_in_split)
                continue

            if len(matching_tasks) != 1:
                raise ValueError("Duplicate task names found across splits in the COCO dataset.")
            match_task = matching_tasks[0]
            if task_in_split != match_task:
                raise ValueError(
                    f"Inconsistent task found across splits in the COCO dataset for task name '{task_in_split.name}'. "
                )

        samples.extend(samples_in_split)

    dataset_info = DatasetInfo(
        dataset_name=dataset_name,
        tasks=tasks,
    )

    hafnia_dataset = HafniaDataset.from_samples_list(samples, info=dataset_info)
    return hafnia_dataset


def coco_format_folder_with_split_to_hafnia_samples(
    path_label_file: Path,
    path_images: Path,
    split_name: str,
    max_samples_per_split: Optional[int],
) -> Tuple[List[Sample], List[TaskInfo]]:
    if not path_label_file.exists():
        raise FileNotFoundError(f"Expected label file not found: {path_label_file}")
    user_logger.info("Loading coco label file as json")
    image_and_annotation_dict = json.loads(path_label_file.read_text())
    user_logger.info("Converting coco dataset to HafniaDataset samples")

    id_to_category, class_names = get_coco_id_category_mapping(image_and_annotation_dict.get("categories", []))
    tasks = [
        TaskInfo.from_class_names(primitive=Bbox, class_names=class_names),
        TaskInfo.from_class_names(primitive=Bitmask, class_names=class_names),
    ]

    coco_licenses = image_and_annotation_dict.get("licenses", [])
    id_to_license_mapping = {lic["id"]: license_types.get_license_by_url(lic["url"]) for lic in coco_licenses}

    coco_images = image_and_annotation_dict.get("images", [])
    if max_samples_per_split is not None:
        coco_images = coco_images[:max_samples_per_split]
    id_to_image = {img["id"]: img for img in coco_images}

    img_id_to_annotations: Dict[int, List[dict]] = {}
    coco_annotations = image_and_annotation_dict.get("annotations", [])
    for annotation in coco_annotations:
        img_id = annotation["image_id"]
        if img_id not in img_id_to_annotations:
            img_id_to_annotations[img_id] = []
        img_id_to_annotations[img_id].append(annotation)

    samples = []
    for img_id, image_dict in progress_bar(
        id_to_image.items(), description=f"Convert coco to hafnia sample '{split_name}'"
    ):
        image_annotations = img_id_to_annotations.get(img_id, [])

        sample = fiftyone_coco_to_hafnia_sample(
            path_images=path_images,
            image_dict=image_dict,
            image_annotations=image_annotations,
            id_to_category=id_to_category,
            class_names=class_names,
            id_to_license_mapping=id_to_license_mapping,
            split_name=split_name,
        )
        samples.append(sample)

    return samples, tasks


def get_coco_id_category_mapping(
    coco_categories: List[dict],
) -> Tuple[Dict[int, dict], List[str]]:
    category_mapping = {}
    for i_cat, category in enumerate(coco_categories):
        category = category.copy()  # Create a copy to avoid modifying the original dictionary.
        category["class_idx"] = i_cat  # Add an index to the category for easier access.
        category_mapping[category["id"]] = category  # Map the category ID to the category dictionary.
    sorted_category_mapping = dict(sorted(category_mapping.items(), key=lambda item: item[1]["class_idx"]))
    class_names = [cat_data["name"] for cat_data in sorted_category_mapping.values()]
    return sorted_category_mapping, class_names


def convert_segmentation_to_rle_list(segmentation: Union[Dict, List], height: int, width: int) -> List[Dict]:
    is_polygon_format = isinstance(segmentation, list)
    if is_polygon_format:  # Multiple polygons format
        rles = coco_utils.frPyObjects(segmentation, height, width)
        return rles

    is_rle_format = isinstance(segmentation, dict) and "counts" in segmentation
    if is_rle_format:  # RLE format
        counts = segmentation["counts"]  # type: ignore
        uncompressed_list_of_ints = isinstance(counts, list)
        if uncompressed_list_of_ints:  # Uncompressed RLE. Counts is List[int]
            rles = coco_utils.frPyObjects([segmentation], height, width)
            return rles

        is_compressed_str_or_bytes = isinstance(counts, str | bytes)
        if is_compressed_str_or_bytes:  # Compressed RLE. Counts is str
            rles = [segmentation]
            return rles

    raise ValueError("Segmentation format not recognized for conversion to RLE.")


def fiftyone_coco_to_hafnia_sample(
    path_images: Path,
    image_dict: Dict,
    image_annotations: List[Dict],
    id_to_category: Dict,
    class_names: List[str],
    id_to_license_mapping: Dict[int, License],
    split_name: str,
) -> Sample:
    image_dict = image_dict.copy()  # Create a copy to avoid modifying the original dictionary.
    file_name_relative = image_dict.pop(COCO_KEY_FILE_NAME)
    file_name = path_images / file_name_relative
    if not file_name.exists():
        raise FileNotFoundError(f"Expected image file not found: {file_name}. Please check the dataset structure.")

    img_width = image_dict.pop("width")
    img_height = image_dict.pop("height")
    bitmasks: List[Bitmask] = []
    bboxes: List[Bbox] = []
    for obj_instance in image_annotations:
        category_data = id_to_category[obj_instance["category_id"]]
        class_name = category_data["name"]  # Get the name of the category.
        class_idx = class_names.index(class_name)
        bbox_list = obj_instance["bbox"]
        if isinstance(bbox_list[0], float):  # Polygon coordinates are often floats.
            bbox_ints = [int(coord) for coord in bbox_list]
        else:
            bbox_ints = bbox_list
        rle_list = convert_segmentation_to_rle_list(obj_instance["segmentation"], height=img_height, width=img_width)
        rle = coco_utils.merge(rle_list)
        rle_string = rle["counts"]
        if isinstance(rle_string, bytes):
            rle_string = rle_string.decode("utf-8")

        if "area" in obj_instance and obj_instance["area"] is not None:
            area_px = obj_instance["area"]
        else:
            area_px = coco_utils.area(rle).item()
        area = float(area_px) / (img_height * img_width)
        bitmask = Bitmask(
            top=bbox_ints[1],
            left=bbox_ints[0],
            height=bbox_ints[3],
            width=bbox_ints[2],
            area=area,
            rle_string=rle_string,
            class_name=class_name,
            class_idx=class_idx,
            object_id=str(obj_instance["id"]),
            meta={"iscrowd": obj_instance["iscrowd"]},
        )
        bitmasks.append(bitmask)

        bbox = Bbox.from_coco(bbox=bbox_list, height=img_height, width=img_width)
        bbox.class_name = class_name
        bbox.class_idx = class_idx
        bbox.object_id = str(obj_instance["id"])  # Use the ID from the instance if available.
        bbox.meta = {"iscrowd": obj_instance["iscrowd"]}
        bbox.area = bbox.calculate_area(image_height=img_height, image_width=img_width)
        bboxes.append(bbox)

    if "license" in image_dict:
        license_data: License = id_to_license_mapping[image_dict["license"]]

        capture_date = datetime.fromisoformat(image_dict["date_captured"])
        source_url = image_dict["flickr_url"] if "flickr_url" in image_dict else image_dict.get("coco_url")
        attribution = Attribution(
            date_captured=capture_date,
            licenses=[license_data],
            source_url=source_url,
        )
    else:
        attribution = None

    return Sample(
        file_path=str(file_name),
        width=img_width,
        height=img_height,
        split=split_name,
        bboxes=bboxes,  # Bboxes will be added later if needed.
        bitmasks=bitmasks,  # Add the bitmask to the sample.
        attribution=attribution,
        meta=image_dict,
    )


def to_coco_format(
    dataset: "HafniaDataset",
    path_output: Path,
    task_name: Optional[str] = None,
    coco_format_type: str = "roboflow",
) -> List[CocoSplitPaths]:
    """Export a `HafniaDataset` to COCO format on disk and return the paths of each split.

    Writes one `instances.json` per split, plus the corresponding image folder layout. If the
    dataset has both `Bitmask` and `Bbox` tasks and `task_name` is omitted, segmentation masks are
    preferred over bounding boxes.

    Args:
        dataset: Dataset to export.
        path_output: Output root folder; one subfolder per split is created beneath it.
        task_name: Specific task to export. If None, a single eligible task is auto-selected
            (Bitmask first, then Bbox).
        coco_format_type: Output layout. Currently only ``"roboflow"`` is supported.

    Returns:
        A list of `CocoSplitPaths` describing the files written for each split.
    """
    samples_modified_all = dataset.samples.with_row_index("id")

    if SampleField.ATTRIBUTION in samples_modified_all.columns:
        samples_modified_all = samples_modified_all.unnest(SampleField.ATTRIBUTION)
        license_table = (
            samples_modified_all["licenses"]
            .explode()
            .struct.unnest()
            .unique()
            .with_row_index("id")
            .select(["id", "name", "url"])
        )
        license_mapping = {lic["name"]: lic["id"] for lic in license_table.iter_rows(named=True)}
    else:
        license_mapping = None
        license_table = None

    if task_name is not None:
        task_info = dataset.info.get_task_by_name(task_name)
    else:
        # Auto derive the task to be used for COCO conversion as only one Bitmask/Bbox task can be present
        # in the coco format. Will first search for Bitmask (because COCO supports segmentation), then Bbox afterwards.
        tasks_info = dataset.info.get_tasks_by_primitive(Bitmask)
        if len(tasks_info) == 0:
            tasks_info = dataset.info.get_tasks_by_primitive(Bbox)
        if len(tasks_info) == 0:
            raise ValueError("No 'Bitmask' or 'Bbox' primitive found in dataset tasks for COCO conversion")
        if len(tasks_info) > 1:
            task_names = [task.name for task in tasks_info]
            raise ValueError(
                f"Found multiple tasks {task_names} for 'Bitmask'/'Bbox' primitive in dataset."
                " Please specify 'task_name'."
            )
        task_info = tasks_info[0]

    categories_list_dict = [
        {"id": i, "name": c, "supercategory": "NotDefined"} for i, c in enumerate(task_info.get_class_names() or [])
    ]
    category_mapping = {cat["name"]: cat["id"] for cat in categories_list_dict}

    split_names = samples_modified_all[SampleField.SPLIT].unique().to_list()

    list_split_paths = []
    for split_name in split_names:
        if coco_format_type == "roboflow":
            path_split = path_output / HAFNIA_TO_ROBOFLOW_SPLIT_NAME[split_name]
            split_paths = format_coco.CocoSplitPaths(
                split=split_name,
                path_images=path_split,
                path_instances_json=path_split / ROBOFLOW_ANNOTATION_FILE_NAME,
            )
        else:
            raise ValueError(f"The specified '{coco_format_type=}' is not supported.")
        samples_in_split = samples_modified_all.filter(pl.col(SampleField.SPLIT) == split_name)
        images_table, annotation_table = _convert_bbox_bitmask_to_coco_format(
            samples_modified=samples_in_split,
            license_mapping=license_mapping,
            task_info=task_info,
            category_mapping=category_mapping,  # type: ignore[arg-type]
        )

        split_paths.path_images.mkdir(parents=True, exist_ok=True)
        src_paths = images_table[COCO_KEY_FILE_NAME].to_list()
        new_relative_image_path = []
        for src_path in src_paths:
            dst_path = split_paths.path_images / Path(src_path).name
            new_relative_image_path.append(dst_path.relative_to(split_paths.path_images).as_posix())
            if dst_path.exists():
                continue

            shutil.copy2(src_path, dst_path)

        images_table_files_moved = images_table.with_columns(
            pl.Series(new_relative_image_path).alias(COCO_KEY_FILE_NAME)
        )
        split_labels = {
            "info": dataset.info.model_dump(mode="json"),
            "images": list(images_table_files_moved.iter_rows(named=True)),
            "categories": categories_list_dict,
            "annotations": list(annotation_table.iter_rows(named=True)),
        }
        if license_table is not None:
            split_labels["licenses"] = list(license_table.iter_rows(named=True))
        split_paths.path_instances_json.parent.mkdir(parents=True, exist_ok=True)
        split_paths.path_instances_json.write_text(json.dumps(split_labels))

        list_split_paths.append(split_paths)

    return list_split_paths


def _convert_bbox_bitmask_to_coco_format(
    samples_modified: pl.DataFrame,
    license_mapping: Optional[Dict[str, int]],
    task_info: TaskInfo,
    category_mapping: Dict[str, int],
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    if task_info.primitive not in [Bbox, Bitmask]:
        raise ValueError(f"Unsupported primitive '{task_info.primitive}' for COCO conversion")

    task_sample_field = task_info.primitive.column_name()
    select_image_table_columns = [
        pl.col("id"),
        pl.col(SampleField.WIDTH).alias("width"),
        pl.col(SampleField.HEIGHT).alias("height"),
        pl.col(SampleField.FILE_PATH).alias(COCO_KEY_FILE_NAME),
    ]

    if license_mapping is not None:
        samples_modified = samples_modified.with_columns(pl.col("licenses").list.first().struct.unnest())
        select_image_table_columns = select_image_table_columns + [
            pl.col("name").replace_strict(license_mapping, return_dtype=pl.Int64).alias("license"),
            pl.col("source_url").alias("flickr_url"),
            pl.col("source_url").alias("coco_url"),
            pl.col("date_captured"),
        ]

    images_table = samples_modified.select(select_image_table_columns)

    annotation_table_full = (
        samples_modified.select(
            pl.col("id").alias("image_id"),
            pl.col(SampleField.HEIGHT).alias("image_height"),
            pl.col(SampleField.WIDTH).alias("image_width"),
            pl.col(task_sample_field),
        )
        .explode(task_sample_field)
        .with_row_index("id")
        .unnest(task_sample_field)
    )

    iscrowd_list = [0 if row is None else row.get("iscrowd", 0) for row in annotation_table_full["meta"]]
    annotation_table_full = annotation_table_full.with_columns(pl.Series(iscrowd_list).alias("iscrowd"))

    if task_info.primitive == Bitmask:
        annotation_table = annotation_table_full.select(
            pl.col("id"),
            pl.col("image_id"),
            category_id=pl.col("class_name").replace_strict(category_mapping, return_dtype=pl.Int64),
            segmentation=pl.struct(
                counts=pl.col("rle_string"),
                size=pl.concat_arr(
                    pl.col("image_height"),
                    pl.col("image_width"),
                ),
            ),
            area=pl.col("area") * pl.col("image_height") * pl.col("image_width"),
            bbox=pl.concat_arr(
                pl.col("left"),  # bbox x coordinate
                pl.col("top"),  # bbox y coordinate
                pl.col("width"),  # bbox width
                pl.col("height"),  # bbox height
            ),
            iscrowd=pl.col("iscrowd"),
        )

    elif task_info.primitive == Bbox:
        annotation_table = annotation_table_full.select(
            pl.col("id"),
            pl.col("image_id"),
            category_id=pl.col("class_name").replace_strict(category_mapping, return_dtype=pl.Int64),
            segmentation=pl.lit([]),
            area=pl.col("height") * pl.col("width") * pl.col("image_height") * pl.col("image_width"),
            bbox=pl.concat_arr(
                pl.col("top_left_x") * pl.col("image_width"),  # x coordinate
                pl.col("top_left_y") * pl.col("image_height"),  # y coordinate
                pl.col("width") * pl.col("image_width"),  # width
                pl.col("height") * pl.col("image_height"),  # height
            ),
            iscrowd=pl.col("iscrowd"),
        )

    return images_table, annotation_table

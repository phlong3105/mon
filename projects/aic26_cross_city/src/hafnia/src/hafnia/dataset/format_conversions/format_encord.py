from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Type

import numpy as np
from pycocotools import mask as mask_utils

from hafnia import utils
from hafnia.dataset.dataset_names import SplitName, StorageFormat
from hafnia.dataset.hafnia_dataset_types import ClassInfo, DatasetInfo, Sample, TaskInfo
from hafnia.dataset.primitives import (
    Bbox,
    Bitmask,
    Classification,
    Point,
    Polygon,
    Primitive,
    Segmentation,
)
from hafnia.log import user_logger
from hafnia.utils import progress_bar, title_to_name

if TYPE_CHECKING:  # Using 'TYPE_CHECKING' to avoid circular imports during type checking
    import encord
    from encord.project import Project

    from hafnia.dataset.hafnia_dataset import HafniaDataset

FILENAME_ENCORD_ANNOTATIONS = "encord_annotations.json.gz"


@dataclass
class EncordNaming:
    primitive: Type[Primitive]
    encord_name: str
    encord_shape_name: str


ENCORD_NAMES = [
    EncordNaming(
        primitive=Classification,
        encord_name="classification",
        encord_shape_name="classification",
    ),
    EncordNaming(primitive=Bbox, encord_name="boundingBox", encord_shape_name="bounding_box"),
    EncordNaming(primitive=Bitmask, encord_name="bitmask", encord_shape_name="bitmask"),
    EncordNaming(primitive=Polygon, encord_name="polygon", encord_shape_name="polygon"),
    EncordNaming(
        primitive=Segmentation,
        encord_name="segmentation",
        encord_shape_name="segmentation",
    ),
]

MAPPING_GET_PRIMITIVE_FROM_ENCORD_NAME = {item.encord_name: item.primitive for item in ENCORD_NAMES}
MAPPING_GET_PRIMITIVE_FROM_ENCORD_SHAPE_NAME = {item.encord_shape_name: item.primitive for item in ENCORD_NAMES}
MAPPING_GET_ENCORD_NAME_FROM_PRIMITIVE = {item.primitive: item.encord_name for item in ENCORD_NAMES}


def from_encord_zip_format(path_compressed_data: Path, max_samples: Optional[int] = None) -> "HafniaDataset":
    """Import an Encord project export (compressed JSON) as a `HafniaDataset`.

    Reads the Encord ontology, label rows and project info from the export and converts them
    into Hafnia primitives (`Bbox`, `Polygon`, `Bitmask`, `Classification`, ...) and tasks.

    Args:
        path_compressed_data: Path to the Encord export (typically produced by
            `dump_encord_project_from_id`).
        max_samples: Optional cap on the number of samples to import.
    """
    metadata_dict = utils.read_compressed_json(path_compressed_data)
    encord_info = metadata_dict["encord_info"]
    encord_labels = metadata_dict["labels"]
    encord_ontology_dict = metadata_dict["ontology"]

    hafnia_dataset = hafnia_dataset_from_encord_data(
        encord_info=encord_info,
        encord_labels=encord_labels,
        encord_ontology_dict=encord_ontology_dict,
        max_samples=max_samples,
    )
    return hafnia_dataset


def dump_encord_project_from_id(
    project_id: str,
    encord_client: "encord.EncordUserClient",
    path_output_file: Path,
    select_rows: Optional[List[str]] = None,
    max_rows: Optional[int] = None,
) -> Path:
    encord_project = encord_client.get_project(project_id)

    encord_data = dump_encord_data(
        encord_project=encord_project,
        select_rows=select_rows,
        max_rows=max_rows,
    )

    path_output_file.parent.mkdir(parents=True, exist_ok=True)
    utils.save_compressed_json(path_output_file, encord_data)
    return path_output_file


def get_encord_dataset_items(
    project: "Project",
    select_rows: Optional[List[str]],
    max_rows: Optional[int] = None,
) -> List[Dict]:
    dataset_items = project.list_label_rows_v2()

    # Sorted by creation date to ensure consistent ordering even if new labels are added later.
    dataset_items = sorted(dataset_items, key=lambda x: x.created_at)

    if select_rows is not None:
        dataset_items = [item for item in dataset_items if item.data_title in select_rows]
    if max_rows is not None:
        dataset_items = dataset_items[:max_rows]
    n_items = len(dataset_items)
    if n_items == 0:
        raise ValueError("No dataset items found for the given project and selection criteria.")
    bundle_size = min(100, n_items)
    user_logger.info("Downloading labels from encord")
    with project.create_bundle(bundle_size=bundle_size) as bundle:
        for label_row in progress_bar(dataset_items, description="Initializing Labels"):
            label_row.initialise_labels(bundle=bundle)
        user_logger.info("This can take several minutes for large datasets")

    user_logger.info("Convert to dictionary format")
    encord_annotation_items = []
    for data_item in progress_bar(dataset_items, description="Loading Encord Annotations"):
        if (select_rows is not None) and data_item.data_title not in select_rows:
            user_logger.info(f"NOTE: Skipping row '{data_item.data_title}'")
            continue
        if not data_item.is_labelling_initialised:
            data_item.initialise_labels()
        encord_data_as_dict = data_item.to_encord_dict()
        encord_annotation_items.append(encord_data_as_dict)

    return encord_annotation_items


def dump_encord_data(
    encord_project: "encord.Project",
    select_rows: Optional[List[str]] = None,
    max_rows: Optional[int] = None,
) -> Dict:
    collection_items = get_encord_dataset_items(encord_project, select_rows=select_rows, max_rows=max_rows)

    encord_ontology_as_dict = encord_project.ontology_structure.to_dict()

    encord_keys = ["created_at", "last_edited_at", "title", "ontology_hash"]
    project_level_info_from_encord = {key: str(getattr(encord_project, key)) for key in encord_keys}

    # Merge dataset info
    dataset_metadata = {
        "dump_date": datetime.now().isoformat(),
        "git_commit": utils.get_git_revision_hash(),
        "labels": collection_items,
        "encord_info": project_level_info_from_encord,
        "ontology": encord_ontology_as_dict,
    }

    return dataset_metadata


def hafnia_dataset_from_encord_data(
    encord_info: Dict,
    encord_labels: List[Dict],
    encord_ontology_dict: Dict,
    max_samples: Optional[int] = None,
) -> "HafniaDataset":
    from hafnia.dataset.hafnia_dataset import HafniaDataset  # Avoid circular import

    tasks = TaskInfo.from_encord_ontology_dict(encord_ontology_dict)
    encord_dataset_title = encord_info["title"]
    dataset_name = title_to_name(encord_dataset_title)
    dataset_info = DatasetInfo(
        dataset_name=dataset_name,
        dataset_title=encord_dataset_title,
        tasks=tasks,
    )
    if max_samples is not None:
        encord_labels = encord_labels[:max_samples]
    samples_list: List[Sample] = []
    for data_item in progress_bar(
        encord_labels, description="Processing Encord Labels"
    ):  # Labels per video / annotator
        samples = _get_sample_from_encord_item(
            data_item,
            tasks=tasks,
        )
        samples_list.extend(samples)

    # Per video metadata as polars DataFrame
    hafnia_dataset = HafniaDataset.from_samples_list(samples_list, info=dataset_info)

    return hafnia_dataset


def _get_nested_attributes(object_attributes: List[Dict], class_info: ClassInfo) -> List[Classification]:
    """Recursively extract nested object attribute answers into a list of Classification objects.

    Args:
        object_attributes: Flat list of all attribute-task answer dicts for an encord annotation.
        class_info: ClassInfo whose *attributes* define which task names to look for at this level.

    Returns:
        A list of Classification objects, each potentially carrying further nested
        Classification objects via their *classifications* field.
    """
    # Build a name-keyed lookup
    attr_by_name: Dict[str, Dict] = {obj_attr["name"]: obj_attr for obj_attr in object_attributes}
    attributes: List[Classification] = []
    for attribute_task_info in class_info.attributes or []:
        if attribute_task_info.name not in attr_by_name:
            continue  # This attribute was not answered
        obj_attr = attr_by_name[attribute_task_info.name]

        for answer in obj_attr.get("answers", []):
            answer_class_name: str = answer["name"]
            answer_class_idx: int = attribute_task_info.get_class_index(answer_class_name)

            if attribute_task_info.classes is None:
                raise ValueError(f"Expected task '{attribute_task_info.name}' to have classes defined in the ontology.")
            answer_class_info: ClassInfo = attribute_task_info.classes[answer_class_idx]

            # Recurse using the same flat list — the selected class may own further sub-attributes
            nested_classifications: Optional[List[Classification]] = None
            if answer_class_info.attributes:
                nested_classifications = _get_nested_attributes(object_attributes, answer_class_info)

            attributes.append(
                Classification(
                    class_name=answer_class_name,
                    class_idx=answer_class_idx,
                    task_name=attribute_task_info.name,
                    classifications=nested_classifications or None,
                )
            )
    return attributes


def _get_sample_from_encord_item(label_row: Dict, tasks: List[TaskInfo]) -> List[Sample]:
    data_type = label_row["data_type"]  # E.g. "video"
    data_item_title = label_row["data_title"]
    encord_data_fields = [
        "data_title",
        "label_hash",
        "dataset_hash",
        "created_at",
        "last_edited_at",
    ]
    encord_meta_data = {key: label_row[key] for key in encord_data_fields}
    object_answers = label_row["object_answers"]
    classification_answers = label_row["classification_answers"]
    data_units = label_row["data_units"]
    assert len(data_units) == 1, (
        f"Expected one data item, but got {len(data_units)}. Investigate why there are multiple data items and adjust the code accordingly."
    )
    data_unit_hash, data_item = next(iter(data_units.items()))
    classification_tasks = [t for t in tasks if issubclass(t.primitive, Classification)]
    object_tasks = {t.primitive: t for t in tasks if not issubclass(t.primitive, Classification)}

    encord_meta_data["id"] = data_item_title
    encord_meta_data["data_unit_hash"] = data_unit_hash
    encord_general_data_fields = [
        "data_hash",
        "data_link",  # e.g. 's3://mdi-encord-dubuque-traffic/4068.mp4'
        "data_type",  # e.g. "video/mp4"
    ]
    encord_meta_data.update({key: data_item[key] for key in encord_general_data_fields})
    samples: List[Sample] = []
    if data_type == "video":
        video_specific_fields = ["data_fps", "data_duration", "height", "width"]
        encord_meta_data.update({f"video.{key}": data_item[key] for key in video_specific_fields})

        labels = data_item["labels"]
        encord_meta_data["number_of_frames"] = len(labels)
        storage_format = StorageFormat.VIDEO
    elif data_type == "image":
        frame_labels = data_item["labels"]
        frame_number = str(data_item["data_sequence"])
        labels = {frame_number: frame_labels}
        storage_format = StorageFormat.IMAGE
    else:
        raise NotImplementedError(f"Data type '{data_type}' not implemented")

    for frame_as_str, annotations in labels.items():
        frame = int(frame_as_str)
        sample = Sample(
            file_path=None,
            height=data_item["height"],
            width=data_item["width"],
            collection_id=data_item_title,
            collection_index=frame,
            storage_format=storage_format,
            split=SplitName.UNDEFINED,
            remote_path=data_item["data_link"],
        )

        if "objects" in annotations:  # Iterate any primitives (e.g., bboxes, masks etc) in the frame
            objects = annotations["objects"]
            for obj_annotation in objects:
                obj_annotation = obj_annotation.copy()
                obj_annotation["confidence"] = float(
                    obj_annotation["confidence"]
                )  # To avoid mixing float and int values
                object_hash = obj_annotation["objectHash"]
                encord_primitive_shape = obj_annotation.pop("shape")

                created_at = parse_encord_date_field(obj_annotation, "createdAt")
                updated_at = parse_encord_date_field(obj_annotation, "lastEditedAt")

                Primitive = primitive_from_encord_shape_name(encord_primitive_shape)
                task_info: TaskInfo = object_tasks[Primitive]

                class_name = obj_annotation["name"]
                class_idx = task_info.get_class_index(class_name)
                if task_info.classes is None:
                    raise ValueError(f"Expected task '{task_info.name}' to have classes defined in the ontology.")
                class_info: ClassInfo = task_info.classes[class_idx]

                object_attributes = object_answers[object_hash]["classifications"]

                attributes = _get_nested_attributes(object_attributes, class_info)
                primitive_attributes = attributes or None
                if Primitive == Bbox:
                    if sample.bboxes is None:
                        sample.bboxes = []
                    encord_primitive_data = obj_annotation.pop("boundingBox")
                    sample.bboxes.append(
                        Bbox(
                            height=float(encord_primitive_data["h"]),
                            width=float(encord_primitive_data["w"]),
                            top_left_x=float(encord_primitive_data["x"]),
                            top_left_y=float(encord_primitive_data["y"]),
                            object_id=object_hash,
                            class_name=class_name,
                            class_idx=class_idx,
                            meta=obj_annotation,
                            classifications=primitive_attributes,
                            created_at=created_at,
                            updated_at=updated_at,
                        )
                    )
                elif Primitive == Polygon:
                    if sample.polygons is None:
                        sample.polygons = []
                    obj_annotation.pop("polygons")  # Remove duplicated field
                    encord_primitive_data = obj_annotation.pop("polygon")
                    polygon_points = [
                        Point(x=float(point["x"]), y=float(point["y"])) for point in encord_primitive_data.values()
                    ]
                    sample.polygons.append(
                        Polygon(
                            points=polygon_points,
                            object_id=object_hash,
                            class_name=class_name,
                            class_idx=class_idx,
                            meta=obj_annotation,
                            classifications=primitive_attributes,
                            created_at=created_at,
                            updated_at=updated_at,
                        )
                    )
                elif Primitive == Bitmask:
                    if sample.bitmasks is None:
                        sample.bitmasks = []
                    encord_primitive_data = obj_annotation.pop("bitmask")
                    rle_string = from_encord_bitmask_to_coco_rle_string(encord_primitive_data)
                    left, top, width, height = mask_utils.toBbox(rle_string)
                    tmp_bitmask = Bitmask(
                        top=int(top),
                        left=int(left),
                        height=int(height),
                        width=int(width),
                        area=mask_utils.area(rle_string) / (sample.height * sample.width),
                        rle_string=rle_string["counts"].decode("utf-8"),
                        object_id=object_hash,
                        class_name=class_name,
                        class_idx=class_idx,
                        meta=obj_annotation,
                        classifications=primitive_attributes,
                        created_at=created_at,
                        updated_at=updated_at,
                    )

                    sample.bitmasks.append(tmp_bitmask)
                else:
                    raise NotImplementedError(
                        f"Encord shape '{encord_primitive_shape}' to Hafnia primitive not implemented yet"
                    )

        if "classifications" in annotations:
            classifications = []
            for classification in annotations["classifications"]:
                classification_hash = classification["classificationHash"]
                classification_attributes = classification_answers[classification_hash]["classifications"]

                class_info = ClassInfo(name=classification["name"], attributes=classification_tasks)
                attributes_: List[Classification] = _get_nested_attributes(classification_attributes, class_info)
                classifications.extend(attributes_)

            sample.classifications = classifications or None
        sample.meta = encord_meta_data
        samples.append(sample)

    return samples


def parse_encord_date_field(obj_annotation: dict, date_field: str) -> Optional[datetime]:
    if date_field not in obj_annotation:
        return None

    if obj_annotation[date_field] is None:
        return None

    encord_dateformat = "%a, %d %b %Y %H:%M:%S"  # E.g. "Wed, 27 Sep 2023 14:48:00" or "Wed, 27 Sep 2023 14:48:00 UTC"
    created_at_str = obj_annotation.pop(date_field).strip().removesuffix("UTC").strip()
    created_at = datetime.strptime(created_at_str, encord_dateformat)
    return created_at


def from_encord_bitmask_to_coco_rle_string(encord_bitmask_dict: Dict) -> Dict:
    """
    Function for converting encord bitmask to coco rle string.
    Note: Encord and coco are both using rle encoding for bitmasks.
    Unfortunately, they are not compatible as rle (encord) is column-major, while rle (coco) is row-major.
    Instead we convert from "rle (encord) -> mask -> rle (coco)" to ensure compatibility.

    Consider using a different approach if performance becomes an issue.
    """
    # Encord has a function for converting rle string to mask 'mask = BitmaskCoordinates.from_dict(encord_bitmask_dict).to_numpy_array()'
    # But it is 100x slower than using pycocotools directly.
    rle_string_flipped = {
        "counts": encord_bitmask_dict["rleString"],
        "size": [encord_bitmask_dict["width"], encord_bitmask_dict["height"]],
    }
    mask_flipped = mask_utils.decode(rle_string_flipped)
    mask = np.transpose(mask_flipped)

    rle_string = mask_utils.encode(np.asfortranarray(mask))
    return rle_string


def primitive_from_encord_name(encord_name: str) -> Type[Primitive]:  # Remove
    if encord_name not in MAPPING_GET_PRIMITIVE_FROM_ENCORD_NAME:
        raise NotImplementedError(f"Encord name {encord_name} is not supported or defined for primitive mapping.")
    return MAPPING_GET_PRIMITIVE_FROM_ENCORD_NAME[encord_name]


def primitive_from_encord_shape_name(
    encord_shape_name: str,
) -> Type[Primitive]:  # Remove
    if encord_shape_name not in MAPPING_GET_PRIMITIVE_FROM_ENCORD_SHAPE_NAME:
        raise NotImplementedError(
            f"Encord shape name {encord_shape_name} is not supported or defined for primitive mapping."
        )
    return MAPPING_GET_PRIMITIVE_FROM_ENCORD_SHAPE_NAME[encord_shape_name]


def encord_name_from_primitive(primitive: Type[Primitive]) -> str:  # Remove
    if primitive not in MAPPING_GET_ENCORD_NAME_FROM_PRIMITIVE:
        raise NotImplementedError(f"Primitive {primitive} is not supported or defined for Encord name mapping.")
    return MAPPING_GET_ENCORD_NAME_FROM_PRIMITIVE[primitive]


def primitive_from_encord_dict(  # Remove
    coordinate_dict: Dict,
    class_name: Optional[str],
    class_idx: Optional[int],
    object_id: Optional[str],
) -> Primitive:
    from hafnia.dataset.primitives import Bbox, Bitmask, Point, Polygon

    if len(coordinate_dict) != 1:
        raise ValueError(f"Expected only one key in the coordinates dict, but got {coordinate_dict.keys()}")
    encord_primitive_name, encord_primitive_data = next(iter(coordinate_dict.items()))
    PrimitiveType = primitive_from_encord_name(encord_primitive_name)

    if PrimitiveType == Bbox:
        if class_idx is not None:
            class_idx = int(class_idx)
        return Bbox(
            height=encord_primitive_data["h"],
            width=encord_primitive_data["w"],
            top_left_x=encord_primitive_data["x"],
            top_left_y=encord_primitive_data["y"],
            class_name=class_name,
            class_idx=class_idx,
            object_id=object_id,
        )
    elif PrimitiveType == Polygon:
        polygon_points = [Point(x=point["x"], y=point["y"]) for point in encord_primitive_data.values()]
        if class_idx is not None:
            class_idx = int(class_idx)
        return Polygon(
            points=polygon_points,
            class_name=class_name,
            class_idx=class_idx,
            object_id=object_id,
        )

    elif PrimitiveType == Bitmask:
        if class_idx is not None:
            class_idx = int(class_idx)
        return Bitmask(
            top=encord_primitive_data["top"],
            left=encord_primitive_data["left"],
            height=encord_primitive_data["height"],
            width=encord_primitive_data["width"],
            rleString=encord_primitive_data["rleString"],
            class_name=class_name,
            class_idx=class_idx,
            object_id=object_id,
        )

    raise NotImplementedError(f"Primitive type {PrimitiveType} is not implemented for Encord export.")

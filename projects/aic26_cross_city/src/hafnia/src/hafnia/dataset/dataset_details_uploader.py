from __future__ import annotations

import base64
from datetime import datetime
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional, Type, Union

import boto3
import polars as pl
from PIL import Image
from pydantic import BaseModel, ConfigDict, field_validator

from hafnia.dataset.dataset_names import (
    DatasetVariant,
    PrimitiveField,
    SampleField,
    SplitName,
)
from hafnia.dataset.hafnia_dataset import HafniaDataset
from hafnia.dataset.hafnia_dataset_types import Attribution, License, Sample, TaskInfo
from hafnia.dataset.operations import table_transformations
from hafnia.dataset.primitives import (
    PRIMITIVE_COLUMN_NAMES,
    Bbox,
    Bitmask,
    Classification,
    Polygon,
    Segmentation,
)
from hafnia.dataset.primitives.primitive import Primitive
from hafnia.platform.datasets import upload_dataset_details
from hafnia.utils import get_path_dataset_gallery_images
from hafnia_cli.config import Config


class DatasetDetails(BaseModel, validate_assignment=True):  # type: ignore[call-arg]
    model_config = ConfigDict(use_enum_values=True)  # To parse Enum values as strings
    name: Optional[str] = None
    title: Optional[str] = None
    overview: Optional[str] = None
    purpose: Optional[str] = None
    licenses: Optional[List[License]] = None
    data_captured_start: Optional[datetime] = None
    data_captured_end: Optional[datetime] = None
    data_received_start: Optional[datetime] = None
    data_received_end: Optional[datetime] = None
    dataset_updated_at: Optional[datetime] = None
    license_citation: Optional[str] = None
    version: Optional[str] = None
    dataset_format_version: Optional[str] = None
    annotation_date: Optional[datetime] = None
    annotation_project_id: Optional[str] = None
    annotation_dataset_id: Optional[str] = None
    annotation_ontology: Optional[str] = None
    annotation_format: Optional[NamedModel] = None
    annotation_provider: Optional[NamedModel] = None
    annotation_reviewer: Optional[NamedModel] = None
    language: Optional[str] = None
    tags: Optional[List[NamedModel]] = None
    data_types: Optional[List[NamedModel]] = None
    use_cases: Optional[List[NamedModel]] = None
    industries: Optional[List[NamedModel]] = None
    # categories: Optional[List[NamedModel]] = None # Not used
    # labeling_model: Optional[List[NamedModel]] = None # Not used
    annotation_methods: Optional[List[NamedModel]] = None
    annotation_types: Optional[List[NamedModel]] = None
    dataset_variants: Optional[List[DbDatasetVariant]] = None
    split_annotations_reports: Optional[List[DbSplitAnnotationsReport]] = None
    sensitivity: Optional[int] = None
    sensitivity_reason: Optional[str] = None
    imgs: Optional[List[DatasetImage]] = None


class DbDatasetVariant(BaseModel, validate_assignment=True):  # type: ignore[call-arg]
    model_config = ConfigDict(use_enum_values=True)  # To parse Enum values as strings
    variant_type: VariantTypeChoices  # Required
    upload_date: Optional[datetime] = None
    size_bytes: Optional[int] = None
    data_type: Optional[str] = None
    number_of_data_items: Optional[int] = None
    resolutions: Optional[List[DbResolution]] = None
    duration: Optional[float] = None
    duration_average: Optional[float] = None
    frame_rate: Optional[float] = None
    bit_rate: Optional[float] = None
    n_cameras: Optional[int] = None


class DbAnnotatedObject(BaseModel, validate_assignment=True):  # type: ignore[call-arg]
    model_config = ConfigDict(use_enum_values=True)  # To parse Enum values as strings
    name: str
    entity_type: EntityTypeChoices
    annotation_type: DbAnnotationType
    task_name: Optional[str] = None  # Not sure if adding task_name makes sense.


class DbAnnotatedObjectReport(BaseModel, validate_assignment=True):  # type: ignore[call-arg]
    model_config = ConfigDict(use_enum_values=True)  # To parse Enum values as strings
    obj: DbAnnotatedObject
    unique_obj_ids: Optional[int] = None
    obj_instances: Optional[int] = None
    images_with_obj: Optional[int] = None

    average_count_per_image: Optional[float] = None

    area_avg_ratio: Optional[float] = None
    area_min_ratio: Optional[float] = None
    area_max_ratio: Optional[float] = None

    height_avg_ratio: Optional[float] = None
    height_min_ratio: Optional[float] = None
    height_max_ratio: Optional[float] = None

    width_avg_ratio: Optional[float] = None
    width_min_ratio: Optional[float] = None
    width_max_ratio: Optional[float] = None

    area_avg_px: Optional[float] = None
    area_min_px: Optional[int] = None
    area_max_px: Optional[int] = None

    height_avg_px: Optional[float] = None
    height_min_px: Optional[int] = None
    height_max_px: Optional[int] = None

    width_avg_px: Optional[float] = None
    width_min_px: Optional[int] = None
    width_max_px: Optional[int] = None

    annotation_type: Optional[List[DbAnnotationType]] = None


class DbDistributionValue(BaseModel, validate_assignment=True):  # type: ignore[call-arg]
    distribution_category: DbDistributionCategory
    percentage: Optional[float] = None

    @staticmethod
    def from_names(type_name: str, category_name: str, percentage: Optional[float]) -> "DbDistributionValue":
        dist_type = DbDistributionType(name=type_name)
        dist_category = DbDistributionCategory(distribution_type=dist_type, name=category_name)
        return DbDistributionValue(distribution_category=dist_category, percentage=percentage)


class DbSplitAnnotationsReport(BaseModel, validate_assignment=True):  # type: ignore[call-arg]
    model_config = ConfigDict(use_enum_values=True)  # To parse Enum values as strings
    variant_type: VariantTypeChoices  # Required
    split: str  # Required
    sample_count: Optional[int] = None
    annotated_object_reports: Optional[List[DbAnnotatedObjectReport]] = None
    distribution_values: Optional[List[DbDistributionValue]] = None


class DbDistributionCategory(BaseModel, validate_assignment=True):  # type: ignore[call-arg]
    distribution_type: DbDistributionType
    name: str


class DbAnnotationType(BaseModel, validate_assignment=True):  # type: ignore[call-arg]
    name: str


class NamedModel(BaseModel, validate_assignment=True):  # type: ignore[call-arg]
    name: str

    @classmethod
    def from_name(cls, name: str) -> "NamedModel":
        return cls(name=name)

    @classmethod
    def from_names(cls, names: List[str]) -> List["NamedModel"]:
        return [cls.from_name(name) for name in names]


class DbResolution(BaseModel, validate_assignment=True):  # type: ignore[call-arg]
    height: int
    width: int


class DataTypeChoices(str, Enum):  # Should match `DataTypeChoices` in `dipdatalib::src/apps/datasets/models.py`
    images = "images"
    video_frames = "video_frames"
    video_clips = "video_clips"


class VariantTypeChoices(str, Enum):  # Should match `VariantType` in `dipdatalib::src/apps/datasets/models.py`
    ORIGINAL = "original"
    HIDDEN = "hidden"
    SAMPLE = "sample"


class SplitChoices(str, Enum):  # Should match `SplitChoices` in `dipdatalib::src/apps/datasets/models.py`
    FULL = "full"
    TRAIN = "train"
    TEST = "test"
    VALIDATION = "validation"


class EntityTypeChoices(str, Enum):  # Should match `EntityTypeChoices` in `dipdatalib::src/apps/datasets/models.py`
    OBJECT = "OBJECT"
    EVENT = "EVENT"


class Annotations(BaseModel):
    """
    Used in 'DatasetImageMetadata' for visualizing image annotations
    in gallery images on the dataset detail page.
    """

    bboxes: Optional[List[Bbox]] = None
    classifications: Optional[List[Classification]] = None
    polygons: Optional[List[Polygon]] = None
    bitmasks: Optional[List[Bitmask]] = None


class DatasetImageMetadata(BaseModel):
    """
    Metadata for gallery images on the dataset detail page on portal.
    """

    annotations: Optional[Annotations] = None
    meta: Optional[Dict[str, Any]] = None

    @classmethod
    def from_sample(cls, sample: Sample) -> "DatasetImageMetadata":
        sample = sample.model_copy(deep=True)
        if sample.file_path is None:
            raise ValueError("Sample has no file_path defined.")
        sample.file_path = "/".join(PurePosixPath(sample.file_path.replace("\\", "/")).parts[-3:])
        metadata = {}
        metadata_field_names = [
            SampleField.FILE_PATH,
            SampleField.HEIGHT,
            SampleField.WIDTH,
            SampleField.SPLIT,
        ]
        for field_name in metadata_field_names:
            if hasattr(sample, field_name) and getattr(sample, field_name) is not None:
                metadata[field_name] = getattr(sample, field_name)

        obj = DatasetImageMetadata(
            annotations=Annotations(
                bboxes=sample.bboxes,
                classifications=sample.classifications,
                polygons=sample.polygons,
                bitmasks=sample.bitmasks,
            ),
            meta=metadata,
        )

        return obj


class DatasetImage(Attribution, validate_assignment=True):  # type: ignore[call-arg]
    img: str  # Base64-encoded image string
    order: Optional[int] = None
    metadata: Optional[DatasetImageMetadata] = None

    @field_validator("img", mode="before")
    def validate_image_path(cls, v: Union[str, Path]) -> str:
        if isinstance(v, Path):
            v = path_image_to_base64_str(path_image=v)

        if not isinstance(v, str):
            raise ValueError("Image must be a string or Path object representing the image path.")

        if not v.startswith("data:image/"):
            raise ValueError("Image must be a base64-encoded data URL.")

        return v


def path_image_to_base64_str(path_image: Path) -> str:
    image = Image.open(path_image)
    mime_format = Image.MIME[image.format]
    as_b64 = base64.b64encode(path_image.read_bytes()).decode("ascii")
    return f"data:{mime_format};base64,{as_b64}"


class DbDistributionType(BaseModel, validate_assignment=True):  # type: ignore[call-arg]
    name: str


VARIANT_TYPE_MAPPING: Dict[
    DatasetVariant, VariantTypeChoices
] = {  # Conider making DatasetVariant & VariantTypeChoices into one
    DatasetVariant.DUMP: VariantTypeChoices.ORIGINAL,
    DatasetVariant.HIDDEN: VariantTypeChoices.HIDDEN,
    DatasetVariant.SAMPLE: VariantTypeChoices.SAMPLE,
}

SPLIT_CHOICE_MAPPING: Dict[SplitChoices, List[str]] = {
    SplitChoices.FULL: SplitName.valid_splits(),
    SplitChoices.TRAIN: [SplitName.TRAIN],
    SplitChoices.TEST: [SplitName.TEST],
    SplitChoices.VALIDATION: [SplitName.VAL],
}


def get_folder_size(path: Path) -> int:
    if not path.exists():
        raise FileNotFoundError(f"The path {path} does not exist.")
    return sum([path.stat().st_size for path in path.rglob("*")])


def upload_dataset_details_to_platform(
    dataset: HafniaDataset,
    dataset_sample: Optional[HafniaDataset] = None,
    path_gallery_images: Optional[Path] = None,
    gallery_samples: Optional[Union[pl.DataFrame, HafniaDataset]] = None,
    distribution_task_names: Optional[List[str]] = None,
    update_platform: bool = True,
    extra_dataset_details: Optional[DatasetDetails] = None,
    cfg: Optional[Config] = None,
) -> dict:
    cfg = cfg or Config()
    dataset_details = dataset_details_from_hafnia_dataset(
        dataset=dataset,
        dataset_sample=dataset_sample,
        path_gallery_images=path_gallery_images,
        gallery_samples=gallery_samples,
        distribution_task_names=distribution_task_names,
        extra_dataset_details=extra_dataset_details,
    )

    if update_platform:
        dataset_details_exclude_none = dataset_details.model_dump(exclude_none=True, mode="json")
        upload_dataset_details(
            data=dataset_details_exclude_none,
            dataset_name=dataset_details.name,
            cfg=cfg,
        )

    dataset_details_dict = dataset_details.model_dump(exclude_none=False, mode="json")
    return dataset_details_dict


def get_resolutions(dataset: HafniaDataset, max_resolutions_selected: int = 8) -> List[DbResolution]:
    unique_resolutions = (
        dataset.samples.select([pl.col("height"), pl.col("width")]).unique().sort(by=["height", "width"])
    )
    if len(unique_resolutions) > max_resolutions_selected:
        skip_size = len(unique_resolutions) // max_resolutions_selected
        unique_resolutions = unique_resolutions.gather_every(skip_size)
    resolutions = [DbResolution(height=res["height"], width=res["width"]) for res in unique_resolutions.to_dicts()]
    return resolutions


def calculate_distribution_values(
    dataset_split: pl.DataFrame, distribution_tasks: Optional[List[TaskInfo]]
) -> List[DbDistributionValue]:
    distribution_tasks = distribution_tasks or []

    if len(distribution_tasks) == 0:
        return []
    classification_column = Classification.column_name()
    classifications = dataset_split.select(pl.col(classification_column).explode())
    classifications = classifications.filter(pl.col(classification_column).is_not_null()).unnest(classification_column)
    classifications = classifications.filter(
        pl.col(PrimitiveField.TASK_NAME).is_in([task.name for task in distribution_tasks])
    )
    dist_values = []
    for (task_name,), task_group in classifications.group_by(PrimitiveField.TASK_NAME):
        distribution_type = DbDistributionType(name=task_name)
        n_annotated_total = len(task_group)
        for (class_name,), class_group in task_group.group_by(PrimitiveField.CLASS_NAME):
            class_count = len(class_group)

            dist_values.append(
                DbDistributionValue(
                    distribution_category=DbDistributionCategory(distribution_type=distribution_type, name=class_name),
                    percentage=class_count / n_annotated_total * 100,
                )
            )
    dist_values = sorted(
        dist_values,
        key=lambda x: (
            x.distribution_category.distribution_type.name,
            x.distribution_category.name,
        ),
    )
    return dist_values


def s3_based_fields(bucket_name: str, variant_type: DatasetVariant, session: boto3.Session) -> tuple[datetime, int]:
    client = session.client("s3")
    file_objects = client.list_objects_v2(Bucket=bucket_name, Prefix=variant_type.value)["Contents"]
    last_modified = sorted([file_obj["LastModified"] for file_obj in file_objects])[-1]
    size = sum([file_obj["Size"] for file_obj in file_objects])
    return last_modified, size


def dataset_details_from_hafnia_dataset(
    dataset: HafniaDataset,
    dataset_sample: Optional[HafniaDataset] = None,
    path_gallery_images: Optional[Path] = None,
    gallery_samples: Optional[Union[pl.DataFrame, HafniaDataset]] = None,
    distribution_task_names: Optional[List[str]] = None,
    extra_dataset_details: Optional[DatasetDetails] = None,
) -> DatasetDetails:
    dataset_variants = []
    dataset_reports = []

    dataset_sample = dataset_sample or dataset.create_sample_dataset()
    path_and_variant = [DatasetVariant.SAMPLE, DatasetVariant.HIDDEN]

    path_gallery_images = path_gallery_images or get_path_dataset_gallery_images(dataset.info.dataset_name)
    gallery_images = create_gallery_images(gallery_samples=gallery_samples, path_gallery_images=path_gallery_images)

    for variant_type in path_and_variant:
        if variant_type == DatasetVariant.SAMPLE:
            samples_variant = dataset_sample
        else:
            samples_variant = dataset
        video_stats = samples_variant.calculate_video_stats()
        files_paths = set(samples_variant.samples[SampleField.FILE_PATH])
        size_bytes = sum([Path(file_path).stat().st_size for file_path in files_paths])
        dataset_variants.append(
            DbDatasetVariant(
                variant_type=VARIANT_TYPE_MAPPING[variant_type],  # type: ignore[index]
                size_bytes=size_bytes,
                data_type=DataTypeChoices.images,
                number_of_data_items=len(samples_variant),
                resolutions=get_resolutions(samples_variant, max_resolutions_selected=8),
                duration=video_stats.get("duration", None),
                duration_average=video_stats.get("duration_average", None),
                frame_rate=video_stats.get("frame_rate", None),
                bit_rate=video_stats.get("bit_rate", None),
                n_cameras=video_stats.get("n_cameras", None),
            )
        )

        distribution_task_names = distribution_task_names or []
        distribution_tasks = [t for t in dataset.info.tasks if t.name in distribution_task_names]
        for split_name in SplitChoices:
            split_names = SPLIT_CHOICE_MAPPING[split_name]
            dataset_split = samples_variant.samples.filter(pl.col(SampleField.SPLIT).is_in(split_names))

            distribution_values = calculate_distribution_values(
                dataset_split=dataset_split,
                distribution_tasks=distribution_tasks,
            )
            report = DbSplitAnnotationsReport(
                variant_type=VARIANT_TYPE_MAPPING[variant_type],  # type: ignore[index]
                split=split_name,
                sample_count=len(dataset_split),
                distribution_values=distribution_values,
            )

            object_reports: List[DbAnnotatedObjectReport] = []
            for PrimitiveType in [Classification, Bbox, Bitmask, Polygon, Segmentation]:
                object_reports.extend(create_reports_from_primitive(dataset_split, PrimitiveType=PrimitiveType))  # type: ignore[type-abstract]

            # Sort object reports by name to more easily compare between versions
            object_reports = sorted(object_reports, key=lambda x: x.obj.name)  # Sort object reports by name
            report.annotated_object_reports = object_reports

            if report.distribution_values is None:
                report.distribution_values = []

            dataset_reports.append(report)

    video_stats = dataset.calculate_video_stats()
    dataset_name = dataset.info.dataset_name
    annotation_types = [NamedModel.from_name(task.primitive.__name__) for task in dataset.info.tasks]
    annotation_date = recent_annotation_date(dataset, PrimitiveField.UPDATED_AT)

    dataset_details = DatasetDetails(
        name=dataset_name,
        title=dataset.info.dataset_title,
        overview=dataset.info.description,
        version=dataset.info.version,
        dataset_variants=dataset_variants,
        split_annotations_reports=dataset_reports,
        dataset_updated_at=dataset.info.updated_at,
        dataset_format_version=dataset.info.format_version,
        license_citation=dataset.info.reference_bibtex,
        annotation_date=annotation_date,
        data_captured_start=video_stats.get("data_captured_start", None),
        data_captured_end=video_stats.get("data_captured_end", None),
        data_received_start=video_stats.get("data_received_start", None),
        data_received_end=video_stats.get("data_received_end", None),
        annotation_types=annotation_types,
        imgs=gallery_images,
    )

    if extra_dataset_details is not None:
        # Update dataset_details with any additional fields provided in extra_dataset_details
        # This will overwrite all explicitly provided fields in 'extra_dataset_details' including
        # 'None' values. This allows users to explicitly set fields to 'None' if they want to
        # remove certain information from the dataset details.
        dataset_details = dataset_details.model_copy(
            update={field: getattr(extra_dataset_details, field) for field in extra_dataset_details.model_fields_set}
        )

    return dataset_details


def recent_annotation_date(dataset: HafniaDataset, primitive_field: str) -> Optional[datetime]:
    def has_primitive_field(df: pl.DataFrame, col_name: str, field_name: str) -> bool:
        if col_name not in df.columns:
            return False
        dtype = df[col_name].dtype
        inner = dtype.inner if isinstance(dtype, pl.List) else dtype
        return isinstance(inner, pl.Struct) and any(f.name == field_name for f in inner.fields)

    columns_with_field = [
        col for col in PRIMITIVE_COLUMN_NAMES if has_primitive_field(dataset.samples, col, primitive_field)
    ]

    candidates: List[datetime] = []
    for col in columns_with_field:
        exploded = dataset.samples[col].explode().drop_nulls()
        if exploded.is_empty():
            continue
        values = exploded.struct.field(primitive_field).drop_nulls()
        if values.is_empty():
            continue
        candidates.append(values.max())
    if len(candidates) == 0:
        return None
    return max(candidates)


def create_reports_from_primitive(
    dataset_split: pl.DataFrame, PrimitiveType: Type[Primitive]
) -> List[DbAnnotatedObjectReport]:
    if not table_transformations.has_primitive(dataset_split, PrimitiveType=PrimitiveType):
        return []

    if PrimitiveType == Segmentation:
        raise NotImplementedError("Not Implemented yet")

    df_per_instance = table_transformations.create_primitive_table(
        dataset_split, PrimitiveType=PrimitiveType, keep_sample_data=True
    )
    if df_per_instance is None:
        raise ValueError(f"Expected {PrimitiveType.__name__} primitive column to be present in the dataset split.")

    entity_type = EntityTypeChoices.OBJECT.value
    if PrimitiveType == Classification:
        entity_type = EntityTypeChoices.EVENT.value

    if PrimitiveType == Bbox:
        df_per_instance = df_per_instance.with_columns(area=pl.col("height") * pl.col("width"))

    if PrimitiveType == Bitmask:
        # width and height are in pixel format for Bitmask convert to ratio
        df_per_instance = df_per_instance.with_columns(
            width=pl.col("width") / pl.col("image.width"),
            height=pl.col("height") / pl.col("image.height"),
        )

    has_height_field = "height" in df_per_instance.columns and df_per_instance["height"].dtype != pl.Null
    if has_height_field:
        df_per_instance = df_per_instance.with_columns(
            height_px=pl.col("height") * pl.col("image.height"),
        )

    has_width_field = "width" in df_per_instance.columns and df_per_instance["width"].dtype != pl.Null
    if has_width_field:
        df_per_instance = df_per_instance.with_columns(
            width_px=pl.col("width") * pl.col("image.width"),
        )

    has_area_field = "area" in df_per_instance.columns and df_per_instance["area"].dtype != pl.Null
    if has_area_field:
        df_per_instance = df_per_instance.with_columns(
            area_px=pl.col("image.height") * pl.col("image.width") * pl.col("area")
        )
    object_reports: List[DbAnnotatedObjectReport] = []
    annotation_type = DbAnnotationType(name=PrimitiveType.__name__)
    for (class_name, task_name), class_group in df_per_instance.group_by(
        PrimitiveField.CLASS_NAME, PrimitiveField.TASK_NAME
    ):
        if class_name is None:
            continue

        object_report = DbAnnotatedObjectReport(
            obj=DbAnnotatedObject(
                name=class_name,
                entity_type=entity_type,
                annotation_type=annotation_type,
                task_name=task_name,
            ),
            unique_obj_ids=class_group[PrimitiveField.OBJECT_ID].n_unique(),
            obj_instances=len(class_group),
            annotation_type=[annotation_type],
            average_count_per_image=len(class_group) / class_group[SampleField.SAMPLE_INDEX].n_unique(),
            images_with_obj=class_group[SampleField.SAMPLE_INDEX].n_unique(),
        )
        if has_height_field:
            object_report.height_avg_ratio = class_group["height"].mean()
            object_report.height_min_ratio = class_group["height"].min()
            object_report.height_max_ratio = class_group["height"].max()
            object_report.height_avg_px = class_group["height_px"].mean()
            object_report.height_min_px = int(class_group["height_px"].min())
            object_report.height_max_px = int(class_group["height_px"].max())

        if has_width_field:
            object_report.width_avg_ratio = class_group["width"].mean()
            object_report.width_min_ratio = class_group["width"].min()
            object_report.width_max_ratio = class_group["width"].max()
            object_report.width_avg_px = class_group["width_px"].mean()
            object_report.width_min_px = int(class_group["width_px"].min())
            object_report.width_max_px = int(class_group["width_px"].max())

        if has_area_field:
            object_report.area_avg_ratio = class_group["area"].mean()
            object_report.area_min_ratio = class_group["area"].min()
            object_report.area_max_ratio = class_group["area"].max()
            object_report.area_avg_px = class_group["area_px"].mean()
            object_report.area_min_px = int(class_group["area_px"].min())
            object_report.area_max_px = int(class_group["area_px"].max())

        object_reports.append(object_report)
    return object_reports


def create_gallery_images(
    gallery_samples: Optional[Union[pl.DataFrame, HafniaDataset]],
    path_gallery_images: Path,
) -> Optional[List[DatasetImage]]:
    if gallery_samples is None or not isinstance(gallery_samples, (pl.DataFrame, HafniaDataset)):
        return None

    if isinstance(gallery_samples, HafniaDataset):
        gallery_samples = gallery_samples.samples

    if len(gallery_samples) == 0:
        return None

    path_gallery_images.mkdir(parents=True, exist_ok=True)
    max_samples = 20
    gallery_samples = gallery_samples.head(max_samples)  # type: ignore[union-attr]
    gallery_images = []
    for gallery_sample in gallery_samples.iter_rows(named=True):  # type: ignore[union-attr]
        sample = Sample(**gallery_sample)

        metadata = DatasetImageMetadata.from_sample(sample=sample)
        sample.classifications = None  # To not draw classifications in gallery images
        image = sample.draw_annotations()
        if sample.file_path is None:
            raise ValueError(f"Sample {sample} does not have a file path.")
        sample_name = PurePosixPath(sample.file_path.replace("\\", "/")).name
        path_gallery_image = path_gallery_images / sample_name
        Image.fromarray(image).save(path_gallery_image)

        dataset_image_dict = {
            "img": path_gallery_image,
            "metadata": metadata,
        }
        if sample.attribution is not None:
            sample.attribution.changes = "Annotations have been visualized"
            dataset_image_dict.update(sample.attribution.model_dump(exclude_none=True))
        gallery_img = DatasetImage(**dataset_image_dict)
        gallery_img.licenses = gallery_img.licenses or []
        gallery_images.append(gallery_img)
    return gallery_images

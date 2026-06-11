from enum import Enum
from typing import List

FILENAME_RECIPE_JSON = "recipe.json"
FILENAME_DATASET_INFO = "dataset_info.json"
FILENAME_ANNOTATIONS_JSONL = "annotations.jsonl"
FILENAME_ANNOTATIONS_PARQUET = "annotations.parquet"

DATASET_FILENAMES_REQUIRED = [
    FILENAME_DATASET_INFO,
    FILENAME_ANNOTATIONS_JSONL,
    FILENAME_ANNOTATIONS_PARQUET,
]


class DeploymentStage(Enum):
    STAGING = "staging"
    PRODUCTION = "production"


TAG_IS_SAMPLE = "sample"

OPS_REMOVE_CLASS = "__REMOVE__"

TASK_NAME_PREDICTIONS_POSTFIX = "/predictions"  # The default postfix for prediction task names


class PrimitiveField:
    CLASS_NAME: str = "class_name"  # Name of the class this primitive is associated with, e.g. "car" for Bbox
    CLASS_IDX: str = "class_idx"  # Index of the class this primitive is associated with, e.g. 0 for "car" if it is the first class  # noqa: E501
    OBJECT_ID: str = "object_id"  # Unique identifier for the object, e.g. "12345123"
    CONFIDENCE: str = "confidence"  # Confidence score (0-1.0) for the primitive, e.g. 0.95 for Bbox

    META: str = "meta"  # Contains metadata about each primitive, e.g. attributes color, occluded, iscrowd, etc.
    CREATED_AT: str = "created_at"  # Date when the annotation/prediction was created
    UPDATED_AT: str = "updated_at"  # Date when the annotation/prediction was last updated
    TASK_NAME: str = "task_name"  # Name of the task this primitive is associated with, e.g. "bboxes" for Bbox

    @staticmethod
    def fields() -> List[str]:
        """
        Returns a list of expected field names for primitives.
        """
        return [
            PrimitiveField.CLASS_NAME,
            PrimitiveField.CLASS_IDX,
            PrimitiveField.OBJECT_ID,
            PrimitiveField.CONFIDENCE,
            PrimitiveField.META,
            PrimitiveField.CREATED_AT,
            PrimitiveField.UPDATED_AT,
            PrimitiveField.TASK_NAME,
        ]


class SampleField:
    FILE_PATH: str = "file_path"
    HEIGHT: str = "height"
    WIDTH: str = "width"
    SPLIT: str = "split"
    TAGS: str = "tags"

    CLASSIFICATIONS: str = "classifications"
    BBOXES: str = "bboxes"
    BITMASKS: str = "bitmasks"
    POLYGONS: str = "polygons"

    STORAGE_FORMAT: str = "storage_format"  # E.g. "image", "video", "zip"
    COLLECTION_INDEX: str = "collection_index"
    COLLECTION_ID: str = "collection_id"
    REMOTE_PATH: str = "remote_path"  # Path to the file in remote storage, e.g. S3
    SAMPLE_INDEX: str = "sample_index"

    ATTRIBUTION: str = "attribution"  # Attribution for the sample (image/video), e.g. creator, license, source, etc.
    META: str = "meta"
    DATASET_NAME: str = "dataset_name"
    VIDEO_INFO: str = "video_info"  # Contains video-specific information such as frame rate, total frames, etc.
    CAMERA_INFO: str = "camera_info"  # Contains camera-specific information such as focal length, sensor size, etc.
    POSITION: str = "position"  # Contains geospatial information such as GPS coordinates, altitude, etc.
    ORIENTATION: str = "orientation"  # Contains orientation information such as yaw, pitch, roll, etc.


class VideoInfoField:
    NAME: str = "name"
    DESCRIPTION: str = "description"
    CAPTURED_AT: str = "captured_at"
    DOWNLOADED_AT: str = "downloaded_at"
    DURATION_SECONDS: str = "duration_seconds"
    FRAME_RATE: str = "frame_rate"
    TOTAL_FRAMES: str = "total_frames"
    RESOLUTION_WIDTH: str = "resolution_width"
    RESOLUTION_HEIGHT: str = "resolution_height"
    ASPECT_RATIO: str = "aspect_ratio"
    CODEC: str = "codec"
    CONTAINER_FORMAT: str = "container_format"
    BIT_RATE_KBPS: str = "bit_rate_kbps"
    COLOR_SPACE: str = "color_space"
    ROTATION_DEGREES: str = "rotation_degrees"


class CameraInfoField:
    ID: str = "id"
    NAME: str = "name"
    MAKE: str = "make"
    MODEL: str = "model"
    LENS: str = "lens"
    FOCAL_LENGTH_MM: str = "focal_length_mm"
    APERTURE_F_NUMBER: str = "aperture_f_number"
    ISO: str = "iso"
    EXPOSURE_TIME_SECONDS: str = "exposure_time_seconds"
    WHITE_BALANCE: str = "white_balance"
    COLOR_SPACE: str = "color_space"
    METERING_MODE: str = "metering_mode"
    FLASH: str = "flash"


class PositionField:
    LATITUDE: str = "latitude"
    LONGITUDE: str = "longitude"
    ALTITUDE_METERS: str = "altitude_meters"
    HEIGHT_ABOVE_GROUND_METERS: str = "height_above_ground_meters"


class OrientationField:
    HEADING_DEGREES: str = "heading_degrees"
    PITCH_DEGREES: str = "pitch_degrees"
    ROLL_DEGREES: str = "roll_degrees"
    YAW_DEGREES: str = "yaw_degrees"


class StorageFormat:
    IMAGE: str = "image"
    VIDEO: str = "video"
    ZIP: str = "zip"


class SplitName:
    TRAIN: str = "train"
    VAL: str = "validation"
    TEST: str = "test"
    UNDEFINED: str = "UNDEFINED"

    @staticmethod
    def valid_splits() -> List[str]:
        return [SplitName.TRAIN, SplitName.VAL, SplitName.TEST]

    @staticmethod
    def all_split_names() -> List[str]:
        return [*SplitName.valid_splits(), SplitName.UNDEFINED]

    @staticmethod
    def map_split_name(potential_split_name: str, strict: bool = True) -> str:
        normalized = potential_split_name.strip().lower()

        if normalized in SPLIT_NAME_MAPPINGS:
            return SPLIT_NAME_MAPPINGS[normalized]

        if strict:
            raise ValueError(f"Unrecognized split name: {potential_split_name}")
        else:
            return SplitName.UNDEFINED


SPLIT_NAME_MAPPINGS = {
    # Train variations
    "train": SplitName.TRAIN,
    "training": SplitName.TRAIN,
    # Validation variations
    "validation": SplitName.VAL,
    "val": SplitName.VAL,
    "valid": SplitName.VAL,
    # Test variations
    "test": SplitName.TEST,
    "testing": SplitName.TEST,
}


class DatasetVariant(Enum):
    DUMP = "dump"
    SAMPLE = "sample"
    HIDDEN = "hidden"

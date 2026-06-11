from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from pydantic import Field

from hafnia.dataset.primitives.primitive import Primitive
from hafnia.dataset.primitives.utils import (
    anonymize_by_resizing,
    class_color_by_name,
    clip,
    get_class_name,
    round_int_clip_value,
)

if TYPE_CHECKING:
    from hafnia.dataset.primitives import Bitmask, Classification, Polygon


class Bbox(Primitive):
    # Names should match names in FieldName
    height: float = Field(
        description="Normalized height of the bounding box (0.0=no height, 1.0=full image height) as a fraction of image height"
    )
    width: float = Field(
        description="Normalized width of the bounding box (0.0=no width, 1.0=full image width) as a fraction of image width"
    )
    top_left_x: float = Field(
        description="Normalized x-coordinate of top-left corner (0.0=left edge, 1.0=right edge) as a fraction of image width"
    )
    top_left_y: float = Field(
        description="Normalized y-coordinate of top-left corner (0.0=top edge, 1.0=bottom edge) as a fraction of image height"
    )
    area: Optional[float] = Field(
        default=None, description="Area of the bounding box as a fraction of the image area (0.0 to 1.0)"
    )
    class_name: Optional[str] = Field(default=None, description="Class name, e.g. 'car'")
    class_idx: Optional[int] = Field(default=None, description="Class index, e.g. 0 for 'car' if it is the first class")
    object_id: Optional[str] = Field(default=None, description="Unique identifier for the object, e.g. '12345123'")
    confidence: float = Field(default=1.0, description="Confidence score (0-1.0) for the primitive, e.g. 0.95 for Bbox")
    ground_truth: bool = Field(default=True, description="Whether this is ground truth or a prediction")
    task_name: str = Field(
        default="", description="Task name to support multiple Bbox tasks in the same dataset. '' defaults to 'bboxes'"
    )
    created_at: Optional[datetime] = Field(default=None, description="Date when the primitive was created")
    updated_at: Optional[datetime] = Field(default=None, description="Date when the primitive was last updated")
    meta: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata for the annotation")

    # Attributes - allow nesting of any primitive type including itself
    bboxes: Optional[List["Bbox"]] = None
    classifications: Optional[List["Classification"]] = None
    polygons: Optional[List["Polygon"]] = None
    bitmasks: Optional[List["Bitmask"]] = None

    @staticmethod
    def default_task_name() -> str:
        return "object_detection"

    @staticmethod
    def column_name() -> str:
        return "bboxes"

    def calculate_area(self, image_height: int, image_width: int) -> float:
        """Calculates the area of the bounding box as a fraction of the image area."""
        return self.height * self.width

    @staticmethod
    def from_coco(bbox: List, height: int, width: int) -> Bbox:
        """
        Converts a COCO-style bounding box to a Bbox object.
        The bbox is in the format [x_min, y_min, width, height].
        """
        x_min, y_min, bbox_width, bbox_height = bbox
        return Bbox(
            top_left_x=x_min / width,
            top_left_y=y_min / height,
            width=bbox_width / width,
            height=bbox_height / height,
        )

    def to_bbox(self) -> Tuple[float, float, float, float]:
        """
        Converts Bbox to a tuple of (x_min, y_min, width, height) with normalized coordinates.
        Values are floats in the range [0, 1].
        """
        return (self.top_left_x, self.top_left_y, self.width, self.height)

    def to_coco_ints(self, image_height: int, image_width: int) -> Tuple[int, int, int, int]:
        xmin = round_int_clip_value(self.top_left_x * image_width, max_value=image_width)
        bbox_width = round_int_clip_value(self.width * image_width, max_value=image_width)

        ymin = round_int_clip_value(self.top_left_y * image_height, max_value=image_height)
        bbox_height = round_int_clip_value(self.height * image_height, max_value=image_height)

        return xmin, ymin, bbox_width, bbox_height

    def to_pixel_coordinates(
        self, image_shape: Tuple[int, int], as_int: bool = True, clip_values: bool = True
    ) -> Union[Tuple[float, float, float, float], Tuple[int, int, int, int]]:
        bb_height = self.height * image_shape[0]
        bb_width = self.width * image_shape[1]
        bb_top_left_x = self.top_left_x * image_shape[1]
        bb_top_left_y = self.top_left_y * image_shape[0]
        xmin, ymin, xmax, ymax = bb_top_left_x, bb_top_left_y, bb_top_left_x + bb_width, bb_top_left_y + bb_height

        if as_int:
            xmin, ymin, xmax, ymax = int(round(xmin)), int(round(ymin)), int(round(xmax)), int(round(ymax))  # noqa: RUF046

        if clip_values:
            xmin = clip(value=xmin, v_min=0, v_max=image_shape[1])
            xmax = clip(value=xmax, v_min=0, v_max=image_shape[1])
            ymin = clip(value=ymin, v_min=0, v_max=image_shape[0])
            ymax = clip(value=ymax, v_min=0, v_max=image_shape[0])

        return xmin, ymin, xmax, ymax

    def draw(self, image: np.ndarray, inplace: bool = False, draw_label: bool = True) -> np.ndarray:
        if not inplace:
            image = image.copy()
        xmin, ymin, xmax, ymax = self.to_pixel_coordinates(image_shape=image.shape[:2])

        class_name = self.get_class_name()
        color = class_color_by_name(class_name)
        font = cv2.FONT_HERSHEY_SIMPLEX
        margin = 5
        bottom_left = (xmin + margin, ymax - margin)
        if draw_label:
            cv2.putText(
                img=image, text=class_name, org=bottom_left, fontFace=font, fontScale=0.75, color=color, thickness=2
            )
        cv2.rectangle(image, pt1=(xmin, ymin), pt2=(xmax, ymax), color=color, thickness=2)

        return image

    def mask(
        self, image: np.ndarray, inplace: bool = False, color: Optional[Tuple[np.uint8, np.uint8, np.uint8]] = None
    ) -> np.ndarray:
        if not inplace:
            image = image.copy()
        xmin, ymin, xmax, ymax = self.to_pixel_coordinates(image_shape=image.shape[:2])
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)

        if color is None:
            color = np.mean(image[ymin:ymax, xmin:xmax], axis=(0, 1)).astype(np.uint8)

        image[ymin:ymax, xmin:xmax] = color
        return image

    def anonymize_by_blurring(self, image: np.ndarray, inplace: bool = False, max_resolution: int = 20) -> np.ndarray:
        if not inplace:
            image = image.copy()
        xmin, ymin, xmax, ymax = self.to_pixel_coordinates(image_shape=image.shape[:2])
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
        blur_region = image[ymin:ymax, xmin:xmax]
        blur_region_upsized = anonymize_by_resizing(blur_region, max_resolution=max_resolution)
        image[ymin:ymax, xmin:xmax] = blur_region_upsized
        return image

    def get_class_name(self) -> str:
        return get_class_name(self.class_name, self.class_idx)

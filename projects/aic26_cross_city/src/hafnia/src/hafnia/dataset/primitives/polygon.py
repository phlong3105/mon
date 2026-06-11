from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from more_itertools import collapse
from pycocotools import mask as coco_utils
from pydantic import Field

from hafnia.dataset.primitives.point import Point
from hafnia.dataset.primitives.primitive import Primitive
from hafnia.dataset.primitives.utils import anonymize_by_resizing, class_color_by_name, get_class_name

if TYPE_CHECKING:
    from hafnia.dataset.primitives import Bbox, Bitmask, Classification


class Polygon(Primitive):
    # Names should match names in FieldName
    points: List[Point] = Field(description="List of points defining the polygon")
    area: Optional[float] = Field(default=None, description="Area of the polygon in pixels")
    class_name: Optional[str] = Field(default=None, description="Class name of the polygon")
    class_idx: Optional[int] = Field(default=None, description="Class index of the polygon")
    object_id: Optional[str] = Field(default=None, description="Object ID of the polygon")
    confidence: float = Field(default=1.0, description="Confidence score (0-1.0) for the primitive, e.g. 0.95 for Bbox")
    ground_truth: bool = Field(default=True, description="Whether this is ground truth or a prediction")

    task_name: str = Field(
        default="", description="Task name to support multiple Polygon tasks in the same dataset. Defaults to 'polygon'"
    )
    created_at: Optional[datetime] = Field(default=None, description="Date when the primitive was created")
    updated_at: Optional[datetime] = Field(default=None, description="Date when the primitive was last updated")
    meta: Optional[Dict[str, Any]] = Field(
        default=None, description="This can be used to store additional information about the polygon"
    )

    # Attributes - allow nesting of any primitive type including itself
    bboxes: Optional[List["Bbox"]] = None
    classifications: Optional[List["Classification"]] = None
    polygons: Optional[List["Polygon"]] = None
    bitmasks: Optional[List["Bitmask"]] = None

    @staticmethod
    def from_list_of_points(
        points: Sequence[Sequence[float]],
        class_name: Optional[str] = None,
        class_idx: Optional[int] = None,
        object_id: Optional[str] = None,
    ) -> "Polygon":
        list_points = [Point(x=point[0], y=point[1]) for point in points]
        return Polygon(points=list_points, class_name=class_name, class_idx=class_idx, object_id=object_id)

    @staticmethod
    def from_list_of_pixel_points(
        points: Sequence[Sequence[int]],
        image_height: int,
        image_width: int,
        class_name: Optional[str] = None,
        class_idx: Optional[int] = None,
        object_id: Optional[str] = None,
    ) -> "Polygon":
        normalized_points = [Point(x=point[0] / image_width, y=point[1] / image_height) for point in points]
        return Polygon(points=normalized_points, class_name=class_name, class_idx=class_idx, object_id=object_id)

    @staticmethod
    def default_task_name() -> str:
        return "polygon_detection"

    @staticmethod
    def column_name() -> str:
        return "polygons"

    def calculate_area(self, image_height: int, image_width: int) -> float:
        raise NotImplementedError()

    def to_pixel_coordinates(
        self, image_shape: Tuple[int, int], as_int: bool = True, clip_values: bool = True
    ) -> List[Tuple]:
        points = [
            point.to_pixel_coordinates(image_shape=image_shape, as_int=as_int, clip_values=clip_values)
            for point in self.points
        ]
        return points

    def draw(self, image: np.ndarray, inplace: bool = False) -> np.ndarray:
        if not inplace:
            image = image.copy()
        points = np.array(self.to_pixel_coordinates(image_shape=image.shape[:2]))

        bottom_left_idx = np.lexsort((-points[:, 1], points[:, 0]))[0]
        bottom_left_np = points[bottom_left_idx, :]
        margin = 5
        bottom_left = (bottom_left_np[0] + margin, bottom_left_np[1] - margin)

        class_name = self.get_class_name()
        color = class_color_by_name(class_name)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.polylines(image, [points], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.putText(
            img=image, text=class_name, org=bottom_left, fontFace=font, fontScale=0.75, color=color, thickness=2
        )
        return image

    def anonymize_by_blurring(self, image: np.ndarray, inplace: bool = False, max_resolution: int = 20) -> np.ndarray:
        if not inplace:
            image = image.copy()
        points = np.array(self.to_pixel_coordinates(image_shape=image.shape[:2]))
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        mask = cv2.fillPoly(mask, [points], color=255).astype(bool)
        xy_min = points.min(axis=0)
        xy_max = points.max(axis=0)

        region_image = image[xy_min[1] : xy_max[1], xy_min[0] : xy_max[0]].copy()
        region_mask = mask[xy_min[1] : xy_max[1], xy_min[0] : xy_max[0]]

        region_image_blurred = anonymize_by_resizing(blur_region=region_image, max_resolution=max_resolution)
        region_image[region_mask] = region_image_blurred[region_mask]

        image_after = np.array(image)
        image_after[xy_min[1] : xy_max[1], xy_min[0] : xy_max[0]] = region_image
        return image_after

    def to_mask(self, img_height: int, img_width: int, use_coco_utils=False) -> np.ndarray:
        if use_coco_utils:
            points = list(collapse(self.to_pixel_coordinates(image_shape=(img_height, img_width))))
            rles = coco_utils.frPyObjects([points], img_height, img_width)
            rle = coco_utils.merge(rles)
            mask = coco_utils.decode(rle).astype(bool)
            return mask

        mask = np.zeros((img_height, img_width), dtype=np.uint8)
        points = np.array(self.to_pixel_coordinates(image_shape=(img_height, img_width)))
        mask = cv2.fillPoly(mask, [points], color=255).astype(bool)
        return mask

    def to_bitmask(self, img_height: int, img_width: int) -> "Bitmask":
        from hafnia.dataset.primitives import Bitmask

        points = list(collapse(self.to_pixel_coordinates(image_shape=(img_height, img_width))))
        rles = coco_utils.frPyObjects([points], img_height, img_width)
        rle = coco_utils.merge(rles)
        top, left, height, width = coco_utils.toBbox(rle)

        rle_string = rle["counts"]
        if isinstance(rle_string, bytes):
            rle_string = rle_string.decode("utf-8")

        return Bitmask(
            rle_string=rle_string,
            top=int(top),
            left=int(left),
            width=int(width),
            height=int(height),
            class_name=self.class_name,
            class_idx=self.class_idx,
            object_id=self.object_id,
            confidence=self.confidence,
            ground_truth=self.ground_truth,
            task_name=self.task_name,
            meta=self.meta,
        )

    def mask(
        self,
        image: np.ndarray,
        inplace: bool = False,
        color: Optional[Tuple[np.uint8, np.uint8, np.uint8]] = None,
    ) -> np.ndarray:
        if not inplace:
            image = image.copy()
        points = self.to_pixel_coordinates(image_shape=image.shape[:2])

        if color is None:
            mask = np.zeros_like(image[:, :, 0])
            bitmask = cv2.fillPoly(mask, pts=[np.array(points)], color=255).astype(bool)  # type: ignore[assignment]
            color = tuple(int(value) for value in np.mean(image[bitmask], axis=0))  # type: ignore[assignment]

        cv2.fillPoly(image, [np.array(points)], color=color)
        return image

    def get_class_name(self) -> str:
        return get_class_name(self.class_name, self.class_idx)

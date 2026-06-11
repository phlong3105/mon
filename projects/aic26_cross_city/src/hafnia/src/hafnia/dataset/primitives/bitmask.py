from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pycocotools.mask as coco_mask
from pydantic import Field

from hafnia.dataset.primitives.primitive import Primitive
from hafnia.dataset.primitives.utils import (
    anonymize_by_resizing,
    class_color_by_name,
    get_class_name,
    text_org_from_left_bottom_to_centered,
)

if TYPE_CHECKING:
    from hafnia.dataset.primitives import Bbox, Classification, Polygon


class Bitmask(Primitive):
    # Names should match names in FieldName
    top: int = Field(description="Bitmask top coordinate in pixels ")
    left: int = Field(description="Bitmask left coordinate in pixels")
    height: int = Field(description="Bitmask height of the bounding box in pixels")
    width: int = Field(description="Bitmask width of the bounding box in pixels")
    rle_string: str = Field(
        description="Run-length encoding (RLE) string for the bitmask region of size (height, width) at (top, left)."
    )
    area: Optional[float] = Field(
        default=None, description="Area of the bitmask as a fraction of the image area (0.0 to 1.0)"
    )
    class_name: Optional[str] = Field(default=None, description="Class name of the object represented by the bitmask")
    class_idx: Optional[int] = Field(default=None, description="Class index of the object represented by the bitmask")
    object_id: Optional[str] = Field(default=None, description="Object ID of the instance represented by the bitmask")
    confidence: float = Field(default=1.0, description="Confidence score (0-1.0) for the primitive, e.g. 0.95 for Bbox")
    ground_truth: bool = Field(default=True, description="Whether this is ground truth or a prediction")

    task_name: str = Field(
        default="", description="Task name to support multiple Bitmask tasks in the same dataset. Defaults to 'bitmask'"
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
        return "mask_detection"

    @staticmethod
    def column_name() -> str:
        return "bitmasks"

    def calculate_area(self, image_height: int, image_width: int) -> float:
        area_px = coco_mask.area(self.to_coco_rle(img_height=image_height, img_width=image_width))
        return area_px / (image_height * image_width)

    @staticmethod
    def from_mask(
        mask: np.ndarray,
        top: int,  # Bounding box top coordinate in pixels
        left: int,  # Bounding box left coordinate in pixels
        class_name: Optional[str] = None,  # This should match the string in 'FieldName.CLASS_NAME'
        class_idx: Optional[int] = None,  # This should match the string in 'FieldName.CLASS_IDX'
        object_id: Optional[str] = None,  # This should match the string in 'FieldName.OBJECT_ID') -> "Bitmask":
    ):
        if len(mask.shape) != 2:
            raise ValueError("Bitmask should be a 2-dimensional array.")

        if mask.dtype != "|b1":
            raise TypeError("Bitmask should be an array of boolean values. For numpy array call .astype(bool).")

        h, w = mask.shape[:2]
        area_pixels = np.sum(mask != 0)
        area = area_pixels / (h * w)

        mask_fortran = np.asfortranarray(mask, np.prod(h * w))  # Convert to Fortran order for COCO encoding
        rle_coding = coco_mask.encode(mask_fortran.astype(bool))  # Encode the mask using COCO RLE
        rle_string = rle_coding["counts"].decode("utf-8")  # Convert the counts to string

        return Bitmask(
            top=top,
            left=left,
            height=h,
            width=w,
            area=area,
            rle_string=rle_string,
            class_name=class_name,
            class_idx=class_idx,
            object_id=object_id,
        )

    def anonymize_by_blurring(self, image: np.ndarray, inplace: bool = False, max_resolution: int = 20) -> np.ndarray:
        mask = self.to_mask(img_height=image.shape[0], img_width=image.shape[1])
        region_mask = mask[self.top : self.top + self.height, self.left : self.left + self.width]
        region_image = image[self.top : self.top + self.height, self.left : self.left + self.width]
        region_image_blurred = anonymize_by_resizing(blur_region=region_image, max_resolution=max_resolution)
        image_mixed = np.where(region_mask[:, :, None], region_image_blurred, region_image)
        image[self.top : self.top + self.height, self.left : self.left + self.width] = image_mixed
        return image

    def to_coco_rle(self, img_height: int, img_width: int, as_bytes: bool = True) -> Dict[str, Any]:
        """Returns the COCO RLE dictionary from the RLE string."""

        rle_string = self.rle_string
        if as_bytes:
            rle_string = rle_string.encode()  # type: ignore[assignment]
        rle = {"counts": rle_string, "size": [img_height, img_width]}
        return rle

    def to_mask(self, img_height: int, img_width: int) -> np.ndarray:
        """Creates a full image mask from the RLE string."""
        mask = coco_mask.decode(self.to_coco_rle(img_height=img_height, img_width=img_width)) > 0
        return mask

    def draw(self, image: np.ndarray, inplace: bool = False, draw_label: bool = True) -> np.ndarray:
        if not inplace:
            image = image.copy()
        if image.ndim == 2:  # for grayscale/monochromatic images
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        img_height, img_width = image.shape[:2]
        bitmask_np = self.to_mask(img_height=img_height, img_width=img_width)

        class_name = self.get_class_name()
        color = class_color_by_name(class_name)

        # Creates transparent masking with the specified color
        image_masked = image.copy()
        image_masked[bitmask_np] = color
        cv2.addWeighted(src1=image, alpha=0.3, src2=image_masked, beta=0.7, gamma=0, dst=image)

        if draw_label:
            # Determines the center of mask
            xy = np.stack(np.nonzero(bitmask_np))
            xy_org = tuple(np.median(xy, axis=1).astype(int))[::-1]

            xy_org = np.median(xy, axis=1).astype(int)[::-1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.75
            thickness = 2
            xy_centered = text_org_from_left_bottom_to_centered(xy_org, class_name, font, font_scale, thickness)

            cv2.putText(
                img=image,
                text=class_name,
                org=xy_centered,
                fontFace=font,
                fontScale=font_scale,
                color=(255, 255, 255),
                thickness=thickness,
            )
        return image

    def mask(
        self, image: np.ndarray, inplace: bool = False, color: Optional[Tuple[np.uint8, np.uint8, np.uint8]] = None
    ) -> np.ndarray:
        if not inplace:
            image = image.copy()

        bitmask_np = self.to_mask(img_height=image.shape[0], img_width=image.shape[1])

        if color is None:
            color = tuple(int(value) for value in np.mean(image[bitmask_np], axis=0))  # type: ignore[assignment]
        image[bitmask_np] = color
        return image

    def get_class_name(self) -> str:
        return get_class_name(self.class_name, self.class_idx)

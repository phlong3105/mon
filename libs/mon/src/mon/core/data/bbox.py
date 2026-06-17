#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Bounding Box Data Structures.

This module provides data structures and utilities for handling bounding boxes.
"""

from __future__ import annotations

__all__ = [
    "BBox",
    "BBoxes",
]

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from numpy import ndarray

from mon.core.dtype import BBoxFormat
from mon.core.path import Path
from mon.core.typing import Int2, Int3
from mon.core.utils import is_list_of, is_valid_str
from .data import Data
from .size import Size


# ==============================================================================
# region BASE CLASSES
# ==============================================================================

@dataclass
class BBox(Data):
    """Data structure for handling a single bounding box.

    Attributes:
        class_id (int): Class ID of the object.
        bbox (ndarray): Bounding box array of shape (4,) and in CXCYWHN format
            (i.e., YOLO). The data format is: [cx, cy, w, h].
        angle (float): Rotation angle in degrees.
        score (float): Confidence score of the detection.
        track_id (int): Tracking ID for multi-object tracking.
        imgsz (Size): Size of the corresponding image as (H, W).
        index (int, optional): Index of the bounding box in the image.
            Defaults to -1.
        path (Path | None, optional): Path to the label file. Defaults to None.
        base_dir (Path | None, optional): Base directory for relative paths.
            This is useful to resolve other files related to the data.
            Defaults to None.
    """

    class_id: int
    bbox: ndarray
    angle: float
    score: float
    track_id: int
    imgsz: Size
    index: int = -1
    path: Path | None = None
    base_dir: Path | None = None

    # --- Lifecycle & Initialization ---
    def __post_init__(self):
        """Perform post-initialization tasks.

        Raises:
            TypeError: If ``bbox`` is not a ndarray.
        """
        # Validate inputs
        if (
            not isinstance(self.bbox, ndarray)
            or len(self.bbox) < 4
            or any(self.bbox < 0)
        ):
            raise TypeError(
                f"expected bbox to be a non-negative 1D array of shape (4,), "
                f"got {type(self.bbox).__name__} and {self.bbox.shape}."
            )
        if not isinstance(self.imgsz, Size):
            self.imgsz = Size.from_any(self.imgsz)
        if is_valid_str(self.path):
            self.path = Path(self.path).normalize()
        if is_valid_str(self.base_dir):
            self.base_dir = Path(self.base_dir).normalize()

    # --- Container / Sequence Methods ---
    def __len__(self) -> int:
        """Return the length of the container."""
        return 1

    def __getitem__(self, index: int) -> BBox:
        """Return the element at the given ``index``."""
        return self

    # --- Properties ---
    @property
    def data(self) -> ndarray:
        """Return the underlying data as a single array of shape (8,).
        The data format is: [class_id, cx, cy, w, h, angle, score, track_id].
        """
        return np.array([
            self.class_id,
            self.bbox[0], self.bbox[1], self.bbox[2], self.bbox[3],
            self.angle, self.score, self.track_id
        ], dtype=np.float32)

    @property
    def shape(self) -> int:
        """Return the data shape."""
        return self.bbox.shape

    @property
    def meta(self) -> dict[str, Any]:
        """Return metadata describing the data."""
        return {
            "path": self.path,
            "index": self.index,
            "shape": self.shape,
            "imgsz": self.imgsz,
        }

    @property
    def is_empty(self) -> bool:
        """Return True if the bounding box is empty."""
        return len(self.bbox) == 0

    # --- Creation ---
    @classmethod
    def from_xyxy(
        cls,
        bbox: ndarray,
        imgsz: Size,
        index: int = -1,
        path: Path | None = None,
        base_dir: Path | None = None,
    ) -> "BBox":
        """Create a bounding box from XYXY format.

        Args:
            bbox (ndarray): Bounding box array of shape (8+,) in XYXY format.
                The data format is: [class_id, x1, y1, x2, y2, angle, score, track_id].
            imgsz (Size): Size of the corresponding image as (H, W).
            index (int, optional): Index of the bounding box in the image.
                Defaults to -1.
            path (Path | None, optional): Path to the label file. Defaults to None.
            base_dir (Path | None, optional): Base directory for relative paths.
                Defaults to None.
        """
        imgsz = Size.from_any(imgsz)
        eps = 1e-7  # Avoid division by zero
        cx = ((bbox[1] + bbox[3]) / 2.0) / (imgsz.w + eps)
        cy = ((bbox[2] + bbox[4]) / 2.0) / (imgsz.h + eps)
        w = (bbox[3] - bbox[1]) / (imgsz.w + eps)
        h = (bbox[4] - bbox[2]) / (imgsz.h + eps)
        return cls(
            class_id=int(bbox[0]),
            bbox=np.array([cx, cy, w, h], dtype=np.float32),
            angle=float(bbox[5]),
            score=float(bbox[6]),
            track_id=int(bbox[7]),
            imgsz=imgsz,
            index=index,
            path=path,
            base_dir=base_dir,
        )

    @classmethod
    def from_xywh(
        cls,
        bbox: ndarray,
        imgsz: Size,
        index: int = -1,
        path: Path | None = None,
        base_dir: Path | None = None
    ) -> "BBox":
        """Create a bounding box from XYWH format.

        Args:
            bbox (ndarray): Bounding box array of shape (8+,) in XYWH format.
                The data format is: [class_id, x, y, w, h, angle, score, track_id].
            imgsz (Size): Size of the corresponding image as (H, W).
            index (int, optional): Index of the bounding box in the image.
                Defaults to -1.
            path (Path | None, optional): Path to the label file. Defaults to None.
            base_dir (Path | None, optional): Base directory for relative paths.
                Defaults to None.
        """
        imgsz = Size.from_any(imgsz)
        eps = 1e-7  # Avoid division by zero
        cx = (bbox[1] + bbox[3] / 2.0) / (imgsz.w + eps)
        cy = (bbox[2] + bbox[4] / 2.0) / (imgsz.h + eps)
        w = bbox[3] / (imgsz.w + eps)
        h = bbox[4] / (imgsz.h + eps)
        return cls(
            class_id=int(bbox[0]),
            bbox=np.array([cx, cy, w, h], dtype=np.float32),
            angle=float(bbox[5]),
            score=float(bbox[6]),
            track_id=int(bbox[7]),
            imgsz=imgsz,
            index=index,
            path=path,
            base_dir=base_dir,
        )

    # --- Computation ---
    def area(self, imgsz: Size | None = None) -> float:
        """Return the area of the bounding box.

        Args:
            imgsz (Size | None, optional): Image size as (H, W). If None, use
                the stored ``self.imgsz``. Defaults to None.
        """
        _, _, w, h = self.bbox
        imgsz: Size = imgsz or self.imgsz
        h = h * imgsz.h
        w = w * imgsz.w
        return h * w

    def center(self, imgsz: Size | None = None) -> ndarray:
        """Return the bounding box center of shape (2,).

         Args:
            imgsz (Size | None, optional): Image size as (H, W). If None, use
                the stored ``self.imgsz``. Defaults to None.
        """
        imgsz: Size = imgsz or self.imgsz
        cx, cy = self.bbox[0:2]
        cx = cx * imgsz.w
        cy = cy * imgsz.h
        return np.array([cx, cy], dtype=np.float32)

    def corners(self, imgsz: Size | None = None) -> ndarray:
        """Return the bounding box corners in [x1, y1, x2, y1, x2, y2, x1, y2]
        format.

        Args:
            imgsz (Size | None, optional): Image size as (H, W). If None, use
                the stored ``self.imgsz``. Defaults to None.
        """
        xyxy = self.xyxy(imgsz)
        x1, y1, x2, y2 = xyxy[0:4]
        # Standard order: top-left, top-right, bottom-right, bottom-left
        return np.array([x1, y1, x2, y1, x2, y2, x1, y2])

    def corners_pts(self, imgsz: Size | None = None) -> ndarray:
        """Return the bounding box corners as 4 points of shape (4, 2).

        Args:
            imgsz (Size | None, optional): Image size as (H, W). If None, use
                the stored ``self.imgsz``. Defaults to None.
        """
        corners = self.corners(imgsz)
        return corners.reshape(4, 2)

    # --- Transformation ---
    def cxcywhn(self, imgsz: Size | None = None) -> ndarray:
        """Return the bounding box array of shape (8,) in CXCYWHN format.

        Args:
            imgsz (Size | None, optional): Image size as (H, W). If None, use
                the stored ``self.imgsz``. Defaults to None.
        """
        return self.bbox

    def cxcywh(self, imgsz: Size | None = None) -> ndarray:
        """Return the bounding box array of shape (8,) in CXCYWH format.

        Args:
            imgsz (Size | None, optional): Image size as (H, W). If None, use
                the stored ``self.imgsz``. Defaults to None.
        """
        cx, cy, w, h = self.bbox
        imgsz: Size = imgsz or self.imgsz
        cx = cx * imgsz.w
        cy = cy * imgsz.h
        w = w * imgsz.w
        h = h * imgsz.h
        return np.array([cx, cy, w, h], dtype=np.float32)

    def xyxy(self, imgsz: Size | None = None) -> ndarray:
        """Return the bounding box array of shape (8,) in XYXY format.

        Args:
            imgsz (Size | None, optional): Image size as (H, W). If None, use
                the stored ``self.imgsz``. Defaults to None.
        """
        cx, cy, w, h = self.bbox
        imgsz: Size = imgsz or self.imgsz
        x1 = (cx - w / 2.0) * imgsz.w
        y1 = (cy - h / 2.0) * imgsz.h
        x2 = (cx + w / 2.0) * imgsz.w
        y2 = (cy + h / 2.0) * imgsz.h
        return np.array([x1, y1, x2, y2], dtype=np.float32)

    def xywh(self, imgsz: Size | None = None) -> ndarray:
        """Return the bounding box array of shape (8,) in XYWH format.

        Args:
            imgsz (Size | None, optional): Image size as (H, W). If None, use
                the stored ``self.imgsz``. Defaults to None.
        """
        cx, cy, w, h = self.bbox
        imgsz: Size = imgsz or self.imgsz
        x = (cx - w / 2.0) * imgsz.w
        y = (cy - h / 2.0) * imgsz.h
        w = w * imgsz.w
        h = h * imgsz.h
        return np.array([x, y, w, h], dtype=np.float32)

    # --- Visualization ---
    def draw(
        self,
        image: ndarray,
        label: str | int | None = None,
        imgsz: Size | None = None,
        color: Int3 = (255, 255, 255),
        alpha: float = 0.3,
        scale: float = 0.6,
        thickness: int = 2,
    ) -> ndarray:
        """Draw the bounding box on the given image.

        Args:
            image (ndarray): The image on which to draw the bounding box.
            label (str | int | None, optional): The label to display on the
                bounding box. Can be either a string or a tracking ID.
                Defaults to None.
            imgsz (Size | None, optional): Image size as (H, W). If None, use
                the stored ``self.imgsz``. Defaults to None.
            color (Int3, optional): The color of the bounding box in BGR format.
                Defaults to (255, 255, 255) for white.
            alpha (float, optional): The transparency of the bounding box.
                Defaults to 0.3.
            scale (float, optional): Font scale for the text. Defaults to 0.6.
            thickness (int, optional): The thickness of the bounding box lines.
                Defaults to 2.

        Returns:
            ndarray: The image with the bounding box drawn on it.
        """
        # 1. Prepare bbox coordinates
        xyxy = self.xyxy(imgsz=imgsz or self.imgsz)
        x1, y1, x2, y2 = xyxy[0:4].astype(int)

        # 2. Create a semi-transparent fill for the bounding box
        overlay = image.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)  # -1 means filled
        cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0.0, image)

        # 3. Draw the bounding box border (optional, but makes edges crisp)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

        # 4. Setup the text properties
        label = str(label or self.class_id)
        font = cv2.FONT_HERSHEY_SIMPLEX
        # Get the width and height of the text to size the background box properly
        (text_w, text_h), baseline = cv2.getTextSize(label, font, scale, thickness)

        # 5. Draw the solid background rectangle for the text
        # We adjust y_min so the label doesn't get cut off if the box is at the
        # very top of the image
        label_y = max(y1, text_h + 10)
        cv2.rectangle(
            image,
            (x1, label_y - text_h - 10),
            (x1 + text_w + 10, label_y),
            color,
            -1,
        )  # Solid fill

        # 6. Draw the text over the solid background
        cv2.putText(
            image, label,
            (x1 + 5, label_y - 5),
            font,
            scale,
            (0, 0, 0),
            thickness
        )

        return image


@dataclass
class BBoxes(Data):
    """Data structure for handling multiple bounding boxes.

    Attributes:
        bboxes (list[BBox]): List of BBox instances representing the bounding boxes.
        imgsz (Size): Size of the corresponding image as (H, W).
        path (Path | None, optional): Path to the label file. Defaults to None.
        base_dir (Path | None, optional): Base directory for relative paths.
            This is useful to resolve other files related to the data.
            Defaults to None.
    """

    bboxes: list[BBox]
    imgsz: Size
    path: Path | None = None
    base_dir: Path | None = None

    # --- Lifecycle & Initialization ---
    def __post_init__(self):
        """Perform post-initialization tasks.

        Raises:
            TypeError: If ``bbox`` is not a list of BBox instances.
        """
        # Validate inputs
        if not is_list_of(self.bboxes, BBox):
            raise TypeError(
                f"expected bboxes to be a list of BBox instances, "
                f"got {type(self.bboxes).__name__} with elements of type "
                f"{type(self.bboxes[0]).__name__ if self.bboxes else 'N/A'}."
            )
        if not isinstance(self.imgsz, Size):
            self.imgsz = Size.from_any(self.imgsz)
        if is_valid_str(self.path):
            self.path = Path(self.path).normalize()
        if is_valid_str(self.base_dir):
            self.base_dir = Path(self.base_dir).normalize()

    # --- Container / Sequence Methods ---
    def __len__(self) -> int:
        """Return the length of the container."""
        return len(self.bboxes)

    def __getitem__(self, index: int) -> BBox:
        """Return the element at the given ``index``."""
        return self.bboxes[index]

    # --- Retrieval ---
    @property
    def data(self) -> ndarray:
        """Return the underlying data."""
        return np.array([b.data for b in self.bboxes], dtype=np.float32)

    @property
    def shape(self) -> Int2:
        """Return the data shape."""
        n = self.__len__()
        m = self.bboxes[0].shape if self.is_empty else 0
        return n, m

    @property
    def class_id(self) -> ndarray:
        """Return the class ID of shape (N,)."""
        return np.array([b.class_id for b in self.bboxes], dtype=np.int32)

    @property
    def coords(self) -> ndarray:
        """Return the bounding box coordinates of shape (N, 4) in CXCYWHN format."""
        return np.array([b.bbox for b in self.bboxes], dtype=np.float32)

    @property
    def angles(self) -> ndarray:
        """Return the rotation angle of shape (N,)."""
        return np.array([b.angle for b in self.bboxes], dtype=np.float32)

    @property
    def scores(self) -> ndarray:
        """Return the confidence score of shape (N,)."""
        return np.array([b.score for b in self.bboxes], dtype=np.float32)

    @property
    def track_ids(self) -> ndarray:
        """Return the tracking ID of shape (N,)."""
        return np.array([b.track_id for b in self.bboxes], dtype=np.int32)

    @property
    def hash(self) -> int | None:
        """Return the hash of the image file if ``path`` is available."""
        return self.path.stat().st_size if isinstance(self.path, Path) else None

    @property
    def meta(self) -> dict:
        """Return metadata describing the data."""
        return {
            "path": self.path,
            "shape": self.shape,
            "imgsz": self.imgsz,
            "hash": self.hash,
        }

    @property
    def is_empty(self) -> bool:
        """Return True if the bounding boxes are empty."""
        return len(self.bboxes) == 0

    # --- Creation ---
    @classmethod
    def from_cxcywhn(
        cls,
        bbox: ndarray,
        imgsz: Size,
        path: Path | None = None,
        base_dir: Path | None = None
    ) -> "BBoxes":
        """Create bounding boxes from XYXY format.

        Args:
            bbox (ndarray): Bounding box array of shape (N, 8+) in XYXY format.
                The data format is:
                [class_id, cx, cy, w, h, angle, score, track_id].
            imgsz (Size): Size of the corresponding image as (H, W).
            path (Path | None, optional): Path to the label file. Defaults to None.
            base_dir (Path | None, optional): Base directory for relative paths.
                Defaults to None.
        """
        imgsz = Size.from_any(imgsz)
        bboxes = [
            BBox(
                class_id=int(bbox[i, 0]),
                bbox=bbox[i, 1:5],
                angle=float(bbox[i, 5]),
                score=float(bbox[i, 6]),
                track_id=int(bbox[i, 7]),
                imgsz=imgsz,
                index=i,
                path=path,
                base_dir=base_dir,
            )
            for i in range(bbox.shape[0])
        ]
        return cls(bboxes=bboxes, imgsz=imgsz, path=path, base_dir=base_dir)

    @classmethod
    def from_xyxy(
        cls,
        bbox: ndarray,
        imgsz: Size,
        path: Path | None = None,
        base_dir: Path | None = None
    ) -> "BBoxes":
        """Create bounding boxes from XYXY format.

        Args:
            bbox (ndarray): Bounding box array of shape (N, 8+) in XYXY format.
                The data format is:
                [class_id, x1, y1, x2, y2, angle, score, track_id].
            imgsz (Size): Size of the corresponding image as (H, W).
            path (Path | None, optional): Path to the label file. Defaults to None.
            base_dir (Path | None, optional): Base directory for relative paths.
                Defaults to None.
        """
        imgsz = Size.from_any(imgsz)
        eps = 1e-7  # Avoid division by zero
        cx = ((bbox[:, 1] + bbox[:, 3]) / 2.0) / (imgsz.w + eps)
        cy = ((bbox[:, 2] + bbox[:, 4]) / 2.0) / (imgsz.h + eps)
        w  = (bbox[:, 3] - bbox[:, 1]) / (imgsz.w + eps)
        h  = (bbox[:, 4] - bbox[:, 2]) / (imgsz.h + eps)
        bboxes = [
            BBox(
                class_id=int(bbox[i, 0]),
                bbox=np.array([cx[i], cy[i], w[i], h[i]], dtype=np.float32),
                angle=float(bbox[i, 5]),
                score=float(bbox[i, 6]),
                track_id=int(bbox[i, 7]),
                imgsz=imgsz,
                index=i,
                path=path,
                base_dir=base_dir,
            )
            for i in range(bbox.shape[0])
        ]
        return cls(bboxes=bboxes, imgsz=imgsz, path=path, base_dir=base_dir)

    @classmethod
    def from_xywh(
        cls,
        bbox: ndarray,
        imgsz: Size,
        path: Path | None = None,
        base_dir: Path | None = None
    ) -> "BBoxes":
        """Create bounding boxes from XYWH format.

        Args:
            bbox (ndarray): Bounding box array of shape (N, 8+) in XYWH format.
                The data format is:
                [class_id, x, y, w, h, angle, score, track_id].
            imgsz (Size): Size of the corresponding image as (H, W).
            path (Path | None, optional): Path to the label file. Defaults to None.
            base_dir (Path | None, optional): Base directory for relative paths.
                Defaults to None.
        """
        imgsz = Size.from_any(imgsz)
        eps = 1e-7  # Avoid division by zero
        cx = (bbox[:, 1] + bbox[:, 3] / 2.0) / (imgsz.w + eps)
        cy = (bbox[:, 2] + bbox[:, 4] / 2.0) / (imgsz.h + eps)
        w  = bbox[:, 3] / (imgsz.w + eps)
        h  = bbox[:, 4] / (imgsz.h + eps)
        bboxes = [
            BBox(
                class_id=int(bbox[i, 0]),
                bbox=np.array([cx[i], cy[i], w[i], h[i]], dtype=np.float32),
                angle=float(bbox[i, 5]),
                score=float(bbox[i, 6]),
                track_id=int(bbox[i, 7]),
                imgsz=imgsz,
                index=i,
                path=path,
                base_dir=base_dir,
            )
            for i in range(bbox.shape[0])
        ]
        return cls(bboxes=bboxes, imgsz=imgsz, path=path, base_dir=base_dir)

    @classmethod
    def from_any(
        cls,
        bbox: list[BBox] | ndarray,
        imgsz: Size,
        fmt: BBoxFormat = BBoxFormat.CXCYWHN,
        path: Path | None = None,
        base_dir: Path | None = None,
    ) -> "BBoxes":
        """Create bounding boxes from an arbitrary format.

        Args:
            bbox (list[BBox] | ndarray): Bounding box data in
                various formats. Can be a single array of shape (N, 8+),
                a list of BBox instances.
            imgsz (Size): Size of the corresponding image as (H, W).
            fmt (BBoxFormat, optional): Format of the input bounding box data.
                Defaults to BBoxFormat.CXCYWHN.
            path (Path | None, optional): Path to the label file. Defaults to None.
            base_dir (Path | None, optional): Base directory for relative paths.
        """
        if is_list_of(bbox, BBox):
            return cls(bboxes=bbox, imgsz=imgsz, path=path, base_dir=base_dir)
        elif fmt == BBoxFormat.CXCYWHN:
            return cls.from_cxcywhn(bbox=bbox, imgsz=imgsz, path=path, base_dir=base_dir)
        elif fmt == BBoxFormat.XYXY:
            return cls.from_xyxy(bbox=bbox, imgsz=imgsz, path=path, base_dir=base_dir)
        elif fmt == BBoxFormat.XYWH:
            return cls.from_xywh(bbox=bbox, imgsz=imgsz, path=path, base_dir=base_dir)
        else:
            raise ValueError(f"unsupported bbox format {fmt}.")

    # --- Mutation ---
    def filter(
        self,
        class_id: int | list[int] = None,
        conf_thres: float = None,
        area_thres: float = None,
    ) -> "BBoxes":
        """Filter bounding boxes by class ID, confidence score, and/or area.

        Args:
            class_id (int | list[int], optional): Class ID(s) to filter by.
                If None, do not filter by class ID. Defaults to None.
            conf_thres (float, optional): Confidence score threshold. If None,
                do not filter by confidence score. Defaults to None.
            area_thres (float, optional): Area threshold. If None, do not filter
                by area. Defaults to None.

        Returns:
            BBoxes: A new BBoxes instance containing only the filtered bounding boxes.
        """
        filtered = self.bboxes

        # Filter by class ID
        if class_id is not None:
            class_id = [class_id] if isinstance(class_id, int) else class_id
            filtered = [b for b in filtered if b.class_id in class_id]

        # Filter by confidence score
        if conf_thres is not None:
            filtered = [b for b in filtered if b.score >= conf_thres]

        # Filter by area
        if area_thres is not None:
            filtered = [b for b in filtered if b.area() >= area_thres]

        return BBoxes(bboxes=filtered, imgsz=self.imgsz, path=self.path, base_dir=self.base_dir)

    # --- Computation ---
    def areas(self, imgsz: Size | None = None) -> ndarray:
        """Return the area of the bounding box.

        Args:
            imgsz (Size | None, optional): Image size as (H, W). If None, use
                the stored ``self.imgsz``. Defaults to None.
        """
        return np.array([b.area(imgsz) for b in self.bboxes], dtype=np.float32)

    def centers(self, imgsz: Size | None = None) -> ndarray:
        """Return the bounding box center of shape (N, 2).

        Args:
            imgsz (Size | None, optional): Image size as (H, W). If None, use
                the stored ``self.imgsz``. Defaults to None.
        """
        return np.array([b.center(imgsz) for b in self.bboxes], dtype=np.float32)

    def corners(self, imgsz: Size | None = None) -> ndarray:
        """Return the bounding box corners in [x1, y1, x2, y1, x2, y2, x1, y2]
        format.

        Args:
            imgsz (Size | None, optional): Image size as (H, W). If None, use
                the stored ``self.imgsz``. Defaults to None.
        """
        return np.array([b.corners(imgsz) for b in self.bboxes], dtype=np.float32)

    def corners_pts(self, imgsz: Size | None = None) -> ndarray:
        """Return the bounding box corners as 4 points of shape (N, 4, 2).

        Args:
            imgsz (Size | None, optional): Image size as (H, W). If None, use
                the stored ``self.imgsz``. Defaults to None.
        """
        corners = self.corners(imgsz)
        return corners.reshape(-1, 4, 2)

    # --- Transformation ---
    def cxcywhn(self, imgsz: Size | None = None) -> ndarray:
        """Return the bounding box in CXCYWHN format.

        Args:
            imgsz (Size | None, optional): Image size as (H, W). If None, use
                the stored ``self.imgsz``. Defaults to None.
        """
        return np.array([b.cxcywhn(imgsz) for b in self.bboxes], dtype=np.float32)

    def cxcywh(self, imgsz: Size | None = None) -> ndarray:
        """Return the bounding box array of shape (N, 8) in CXCYWH format.

        Args:
            imgsz (Size | None, optional): Image size as (H, W). If None, use
                the stored ``self.imgsz``. Defaults to None.
        """
        return np.array([b.cxcywh(imgsz) for b in self.bboxes], dtype=np.float32)

    def xyxy(self, imgsz: Size | None = None) -> ndarray:
        """Return the bounding box array of shape (N, 8) in XYXY format.

        Args:
            imgsz (Size | None, optional): Image size as (H, W). If None, use
                the stored ``self.imgsz``. Defaults to None.
        """
        return np.array([b.xyxy(imgsz) for b in self.bboxes], dtype=np.float32)

    def xywh(self, imgsz: Size | None = None) -> ndarray:
        """Return the bounding box array of shape (N, 8) in XYWH format.

        Args:
            imgsz (Size | None, optional): Image size as (H, W). If None, use
                the stored ``self.imgsz``. Defaults to None.
        """
        return np.array([b.xywh(imgsz) for b in self.bboxes], dtype=np.float32)

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion

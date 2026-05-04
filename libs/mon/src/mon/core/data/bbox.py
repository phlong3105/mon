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

import numpy as np
from numpy import ndarray

from mon.core.path import Path
from mon.core.typing import Int2, IntOrTuple2
from mon.core.utils import is_valid_str
from .data import Data
from .size import Size


# ==============================================================================
# region BASE CLASSES
# ==============================================================================

@dataclass
class BBox(Data):
    """Data structure for handling a single bounding box.

    Attributes:
        bbox (ndarray): Bounding box array of shape (8+) and in CXCYWHN format
            (i.e., YOLO). The data format is: [cx, cy, w, h, angle, class_id,
            score, track_id].
        imgsz (Size | IntOrTuple2): Size of the corresponding image as (H, W).
        index (int, optional): Index of the bounding box in the image.
            Defaults to -1.
        path (Path | None, optional): Path to the label file. Defaults to None.
        base_dir (Path | None, optional): Base directory for relative paths.
            This is useful to resolve other files related to the data.
            Defaults to None.
    """

    bbox: ndarray
    imgsz: Size | IntOrTuple2
    index: int = -1
    path: Path | None = None
    base_dir: Path | None = None

    # --- Lifecycle & Initialization ---
    def __post_init__(self):
        """Perform post-initialization tasks.

        Raises:
            TypeError: If ``bbox`` is not a ndarray.
            ValueError: If ``bbox`` does not have at least 8 elements.
            ValueError: If any element in ``bbox[4:]`` is negative.
            ValueError: If ``imgsz`` is not a tuple of length 2.
        """
        # Validate inputs
        if not isinstance(self.bbox, ndarray):
            raise TypeError(
                f"Expected 'image' to be a array, "
                f"but got '{type(self.bbox).__name__}'."
            )
        if len(self.bbox) < 8:
            raise ValueError(
                f"Expected 'bbox' to be a 1D array of shape (8+), "
                f"but got 1D array of {len(self.bbox)} elements."
            )
        if any(self.bbox[4:] < 0):
            raise ValueError(
                f"Expected all elements of 'bbox' to be non-negative, "
                f"but got {self.bbox[4:]}.",
            )
        if not isinstance(self.imgsz, Size):
            self.imgsz = Size.from_value(self.imgsz)
        if is_valid_str(self.path):
            self.path = Path(self.path).normalize()
        if is_valid_str(self.base_dir):
            self.base_dir = Path(self.base_dir).normalize()

    # --- Container / Sequence Methods ---
    def __len__(self) -> int:
        """Return the length of the container."""
        return 1

    def __getitem__(self, index: int) -> ndarray:
        """Return the element at the given ``index``."""
        return self.bbox

    # --- Properties ---
    @property
    def data(self) -> ndarray:
        """Return the underlying data."""
        return self.bbox

    @property
    def shape(self) -> int:
        """Return the data shape."""
        return len(self.bbox)

    @property
    def coords(self) -> ndarray:
        """Return the bounding box coordinates of shape (4,) in CXCYWHN format."""
        return self.bbox[0:4]

    @property
    def angle(self) -> float:
        """Return the rotation angle."""
        return self.bbox[4]

    @property
    def class_id(self) -> int:
        """Return the class ID."""
        return int(self.bbox[5])

    @property
    def conf(self) -> float:
        """Return the confidence score."""
        return self.bbox[6]

    @property
    def track_id(self) -> int:
        """Return the tracking ID."""
        return int(self.bbox[7])

    @property
    def meta(self) -> dict[str, Any]:
        """Return metadata describing the data."""
        return {
            "path": self.path,
            "index": self.index,
            "shape": self.shape,
            "imgsz": self.imgsz,
        }

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
            bbox (ndarray): Bounding box array of shape (4+,) in XYXY format.
            imgsz (Size): Size of the corresponding image as (H, W).
            index (int, optional): Index of the bounding box in the image.
                Defaults to -1.
            path (Path | None, optional): Path to the label file. Defaults to None.
            base_dir (Path | None, optional): Base directory for relative paths.
                Defaults to None.

        Returns:
            BBox: Created bounding box instance.
        """
        imgsz = Size.from_value(imgsz)
        eps = 1e-7  # Avoid division by zero
        cx = ((bbox[0] + bbox[2]) / 2.0) / (imgsz.w + eps)
        cy = ((bbox[1] + bbox[3]) / 2.0) / (imgsz.h + eps)
        w = (bbox[2] - bbox[0]) / (imgsz.w + eps)
        h = (bbox[3] - bbox[1]) / (imgsz.h + eps)
        return cls(
            bbox=np.array([cx, cy, w, h, *bbox[4:]], dtype=np.float32),
            imgsz=imgsz, index=index, path=path, base_dir=base_dir
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
            bbox (ndarray): Bounding box array of shape (4+,) in XYWH format.
            imgsz (Size): Size of the corresponding image as (H, W).
            index (int, optional): Index of the bounding box in the image.
                Defaults to -1.
            path (Path | None, optional): Path to the label file. Defaults to None.
            base_dir (Path | None, optional): Base directory for relative paths.
                Defaults to None.

        Returns:
            BBox: Created bounding box instance.
        """
        imgsz = Size.from_value(imgsz)
        eps = 1e-7  # Avoid division by zero
        cx = (bbox[0] + bbox[2] / 2.0) / (imgsz.w + eps)
        cy = (bbox[1] + bbox[3] / 2.0) / (imgsz.h + eps)
        w = bbox[2] / (imgsz.w + eps)
        h = bbox[3] / (imgsz.h + eps)
        return cls(
            bbox=np.array([cx, cy, w, h, *bbox[4:]], dtype=np.float32),
            imgsz=imgsz, index=index, path=path, base_dir=base_dir
        )

    # --- Computation ---
    def area(self, imgsz: Size | None = None) -> float:
        """Return the area of the bounding box.

        Args:
            imgsz (Size | None, optional): Image size as (H, W). If None, use
                the stored ``self.imgsz``. Defaults to None.
        """
        _, _, w, h = self.coords
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
        cx, cy = self.coords[0:2]
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
        cx, cy, w, h = self.coords
        imgsz: Size = imgsz or self.imgsz
        cx = cx * imgsz.w
        cy = cy * imgsz.h
        w = w * imgsz.w
        h = h * imgsz.h
        return np.array([cx, cy, w, h, *self.bbox[4:]], dtype=np.float32)

    def xyxy(self, imgsz: Size | None = None) -> ndarray:
        """Return the bounding box array of shape (8,) in XYXY format.

        Args:
            imgsz (Size | None, optional): Image size as (H, W). If None, use
                the stored ``self.imgsz``. Defaults to None.
        """
        cx, cy, w, h = self.coords
        imgsz: Size = imgsz or self.imgsz
        x1 = (cx - w / 2.0) * imgsz.w
        y1 = (cy - h / 2.0) * imgsz.h
        x2 = (cx + w / 2.0) * imgsz.w
        y2 = (cy + h / 2.0) * imgsz.h
        return np.array([x1, y1, x2, y2, *self.bbox[4:]], dtype=np.float32)

    def xywh(self, imgsz: Size | None = None) -> ndarray:
        """Return the bounding box array of shape (8,) in XYWH format.

        Args:
            imgsz (Size | None, optional): Image size as (H, W). If None, use
                the stored ``self.imgsz``. Defaults to None.
        """
        cx, cy, w, h = self.coords
        imgsz: Size = imgsz or self.imgsz
        x = (cx - w / 2.0) * imgsz.w
        y = (cy - h / 2.0) * imgsz.h
        w = w * imgsz.w
        h = h * imgsz.h
        return np.array([x, y, w, h, *self.bbox[4:]], dtype=np.float32)


@dataclass
class BBoxes(Data):
    """Data structure for handling multiple bounding boxes.

    Attributes:
        bbox (ndarray): Bounding box array of shape (N, 8+) and in CXCYWHN format.
           (i.e., YOLO). The data format is: [cx, cy, w, h, angle, class_id,
            score, track_id].
        imgsz (Size | IntOrTuple2): Size of the corresponding image as (H, W).
        path (Path | None, optional): Path to the label file. Defaults to None.
        base_dir (Path | None, optional): Base directory for relative paths.
            This is useful to resolve other files related to the data.
            Defaults to None.
    """

    bbox: ndarray
    imgsz: Size | IntOrTuple2
    path: Path | None = None
    base_dir: Path | None = None

    # --- Lifecycle & Initialization ---
    def __post_init__(self):
        """Perform post-initialization tasks.

        Raises:
            TypeError: If ``bbox`` is not a ndarray.
            ValueError: If ``bbox`` does not have at least 8 elements.
            ValueError: If any element in ``bbox[4:]`` is negative.
            ValueError: If ``imgsz`` is not a tuple of length 2.
        """
        # Validate inputs
        if not isinstance(self.bbox, ndarray):
            raise TypeError(
                f"Expected 'image' to be a array, "
                f"but got '{type(self.bbox).__name__}'."
            )
        if self.bbox.ndim != 2 or self.bbox.shape[1] < 8:
            raise ValueError(
                f"Expected 'bbox' to be a 2D array of shape (N, 8+), "
                f"but got {self.bbox.shape}D array."
            )
        if (self.bbox[:, 4:] < 0).any():
            raise ValueError(
                f"Expected all elements of 'bbox' to be non-negative."
            )
        if not isinstance(self.imgsz, Size):
            self.imgsz = Size.from_value(self.imgsz)
        if is_valid_str(self.path):
            self.path = Path(self.path).normalize()
        if is_valid_str(self.base_dir):
            self.base_dir = Path(self.base_dir).normalize()

    # --- Container / Sequence Methods ---
    def __len__(self) -> int:
        """Return the length of the container."""
        return len(self.bbox)

    def __getitem__(self, index: int) -> BBox:
        """Return the element at the given ``index``."""
        return BBox(
            bbox=self.bbox[index],
            imgsz=self.imgsz,
            index=index,
            path=self.path,
            base_dir=self.base_dir
        )

    # --- Retrieval ---
    @property
    def data(self) -> ndarray:
        """Return the underlying data."""
        return self.bbox

    @property
    def shape(self) -> Int2:
        """Return the data shape."""
        return self.bbox.shape

    @property
    def coords(self) -> ndarray:
        """Return the bounding box coordinates of shape (N, 4) in CXCYWHN format."""
        return self.bbox[:, 0:4]

    @property
    def angle(self) -> ndarray:
        """Return the rotation angle of shape (N,)."""
        return self.bbox[:, 4]

    @property
    def class_id(self) -> ndarray:
        """Return the class ID of shape (N,)."""
        return self.bbox[:, 5]

    @property
    def conf(self) -> ndarray:
        """Return the confidence score of shape (N,)."""
        return self.bbox[:, 6]

    @property
    def track_id(self) -> ndarray:
        """Return the tracking ID of shape (N,)."""
        return self.bbox[:, 7]

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

    # --- Creation ---
    @classmethod
    def from_bbox_list(
        cls,
        bbox_list: list[BBox],
        imgsz: Size,
        path: Path | None = None,
        base_dir: Path | None = None,
    ) -> "BBoxes":
        """Create bounding boxes from a list of bounding boxes.

        Args:
            bbox_list (list[BBox]): List of BBox instances.
            imgsz (Size): Size of the corresponding image as (H, W).
            path (Path | None, optional): Path to the label file. Defaults to None.
            base_dir (Path | None, optional): Base directory for relative paths.
                Defaults to None.

        Returns:
            BBoxes: Created bounding boxes instance.
        """
        return cls(
            bbox=np.array([b.bbox for b in bbox_list], dtype=np.float32),
            imgsz=imgsz, path=path, base_dir=base_dir
        )

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
            bbox (ndarray): Bounding box array of shape (N, 4+) in XYXY format.
            imgsz (Size): Size of the corresponding image as (H, W).
            path (Path | None, optional): Path to the label file. Defaults to None.
            base_dir (Path | None, optional): Base directory for relative paths.
                Defaults to None.

        Returns:
            BBoxes: Created bounding boxes instance.
        """
        imgsz = Size.from_value(imgsz)
        eps = 1e-7  # Avoid division by zero
        cx = ((bbox[:, 0] + bbox[:, 2]) / 2.0) / (imgsz[1] + eps)
        cy = ((bbox[:, 1] + bbox[:, 3]) / 2.0) / (imgsz[0] + eps)
        w = (bbox[:, 2] - bbox[:, 0]) / (imgsz[1] + eps)
        h = (bbox[:, 3] - bbox[:, 1]) / (imgsz[0] + eps)
        return cls(
            bbox=np.array([cx, cy, w, h, *bbox[:, 4:].T], dtype=np.float32).T,
            imgsz=imgsz, path=path, base_dir=base_dir
        )

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
            bbox (ndarray): Bounding box array of shape (N, 4+) in XYWH format.
            imgsz (Size): Size of the corresponding image as (H, W).
            path (Path | None, optional): Path to the label file. Defaults to None.
            base_dir (Path | None, optional): Base directory for relative paths.
                Defaults to None.

        Returns:
            BBoxes: Created bounding boxes instance.
        """
        imgsz = Size.from_value(imgsz)
        eps = 1e-7  # Avoid division by zero
        cx = (bbox[:, 0] + bbox[:, 2] / 2.0) / (imgsz[1] + eps)
        cy = (bbox[:, 1] + bbox[:, 3] / 2.0) / (imgsz[0] + eps)
        w = bbox[:, 2] / (imgsz[1] + eps)
        h = bbox[:, 3] / (imgsz[0] + eps)
        return cls(
            bbox=np.array([cx, cy, w, h, *bbox[:, 4:].T], dtype=np.float32).T,
            imgsz=imgsz, path=path, base_dir=base_dir
        )

    # --- Computation ---
    def area(self, imgsz: Size | None = None) -> ndarray:
        """Return the area of the bounding box.

        Args:
            imgsz (Size | None, optional): Image size as (H, W). If None, use
                the stored ``self.imgsz``. Defaults to None.
        """
        _, _, w, h = self.coords.T
        imgsz: Size = imgsz or self.imgsz
        h = h * imgsz.h
        w = w * imgsz.w
        return h * w

    def center(self, imgsz: Size | None = None) -> ndarray:
        """Return the bounding box center of shape (N, 2).

        Args:
            imgsz (Size | None, optional): Image size as (H, W). If None, use
                the stored ``self.imgsz``. Defaults to None.
        """
        imgsz: Size = imgsz or self.imgsz
        cx, cy = self.coords[:, 0:2].T
        cx = cx * imgsz.w
        cy = cy * imgsz.h
        return np.array([cx, cy], dtype=np.float32).T

    def corners(self, imgsz: Size | None = None) -> ndarray:
        """Return the bounding box corners in [x1, y1, x2, y1, x2, y2, x1, y2]
        format.

        Args:
            imgsz (Size | None, optional): Image size as (H, W). If None, use
                the stored ``self.imgsz``. Defaults to None.
        """
        xyxy = self.xyxy(imgsz)
        x1, y1, x2, y2 = xyxy[:, 0:4].T
        # Standard order: top-left, top-right, bottom-right, bottom-left
        return np.array([x1, y1, x2, y1, x2, y2, x1, y2], dtype=np.float32).T

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
        return self.bbox

    def cxcywh(self, imgsz: Size | None = None) -> ndarray:
        """Return the bounding box array of shape (N, 8) in CXCYWH format.

        Args:
            imgsz (Size | None, optional): Image size as (H, W). If None, use
                the stored ``self.imgsz``. Defaults to None.
        """
        imgsz: Size = imgsz or self.imgsz
        cx, cy, w, h = self.coords.T
        rest = self.bbox[:, 4:]
        cx = cx * imgsz.w
        cy = cy * imgsz.h
        w = w * imgsz.w
        h = h * imgsz.h
        return np.column_stack((cx, cy, w, h, rest)).astype(np.float32)

    def xyxy(self, imgsz: Size | None = None) -> ndarray:
        """Return the bounding box array of shape (N, 8) in XYXY format.

        Args:
            imgsz (Size | None, optional): Image size as (H, W). If None, use
                the stored ``self.imgsz``. Defaults to None.
        """
        imgsz: Size = imgsz or self.imgsz
        cx, cy, w, h = self.coords.T
        rest = self.bbox[:, 4:]
        x1 = (cx - w / 2) * imgsz.w
        y1 = (cy - h / 2) * imgsz.h
        x2 = (cx + w / 2) * imgsz.w
        y2 = (cy + h / 2) * imgsz.h
        return np.column_stack((x1, y1, x2, y2, rest)).astype(np.float32)

    def xywh(self, imgsz: Size | None = None) -> ndarray:
        """Return the bounding box array of shape (N, 8) in XYWH format.

        Args:
            imgsz (Size | None, optional): Image size as (H, W). If None, use
                the stored ``self.imgsz``. Defaults to None.
        """
        imgsz: Size = imgsz or self.imgsz
        cx, cy, w, h = self.coords.T
        rest = self.bbox[:, 4:]
        x = (cx - w / 2) * imgsz.w
        y = (cy - h / 2) * imgsz.h
        h = h * imgsz.h
        w = w * imgsz.w
        return np.column_stack((x, y, w, h, rest)).astype(np.float32)

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion

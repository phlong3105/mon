#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Bounding Box Data Structures.

This module provides data structures and utilities for handling bounding boxes.
"""

from __future__ import annotations

__all__ = [
    "BBox",
    "BBoxes",
    "to_2d_bbox",
]

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy import ndarray

from mon.core.path import Path
from mon.core.typing import Int2, PathLike
from mon.core.utils import is_valid_str
from .data import Data


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
        imgsz (tuple[int, int]): Size of the corresponding image as (H, W).
        index (int, optional): Index of the bounding box in the image.
            Defaults to -1.
        path (Path, optional): Path to the label file. Defaults to None.
        base_dir (Path, optional): Base directory for relative paths. This is
            useful to resolve other files related to the data. Defaults to None.
    """

    bbox: ndarray
    imgsz: tuple[int, int]
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
        if len(self.imgsz) != 2:
            raise ValueError(
                f"Expected 'imgsz' to be a tuple of length 2, "
                f"but got {self.imgsz}."
            )
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
        imgsz: tuple[int, int],
        index: int = -1,
        path: PathLike | None = None,
        base_dir: PathLike | None = None,
    ) -> "BBox":
        """Create a bounding box from XYXY format.

        Args:
            bbox (ndarray): Bounding box array of shape (4+,) in XYXY format.
            imgsz (tuple[int, int]): Size of the corresponding image as (H, W).
            index (int, optional): Index of the bounding box in the image.
                Defaults to -1.
            path (PathLike, optional): Path to the label file. Defaults to None.
            base_dir (PathLike, optional): Base directory for relative paths.
                Defaults to None.

        Returns:
            BBox: Created bounding box instance.
        """
        eps = 1e-7  # to avoid division by zero
        cx = ((bbox[0] + bbox[2]) / 2.0) / (imgsz[1] + eps)
        cy = ((bbox[1] + bbox[3]) / 2.0) / (imgsz[0] + eps)
        w = (bbox[2] - bbox[0]) / (imgsz[1] + eps)
        h = (bbox[3] - bbox[1]) / (imgsz[0] + eps)
        return cls(
            bbox=np.array([cx, cy, w, h, *bbox[4:]], dtype=np.float32),
            imgsz=imgsz,
            index=index,
            path=path,
            base_dir=base_dir
        )

    @classmethod
    def from_xywh(
        cls,
        bbox: ndarray,
        imgsz: tuple[int, int],
        index: int = -1,
        path: PathLike | None = None,
        base_dir: PathLike | None = None
    ) -> "BBox":
        """Create a bounding box from XYWH format.

        Args:
            bbox (ndarray): Bounding box array of shape (4+,) in XYWH format.
            imgsz (tuple[int, int]): Size of the corresponding image as (H, W).
            index (int, optional): Index of the bounding box in the image.
                Defaults to -1.
            path (PathLike, optional): Path to the label file. Defaults to None.
            base_dir (PathLike, optional): Base directory for relative paths.
                Defaults to None.

        Returns:
            BBox: Created bounding box instance.
        """
        eps = 1e-7  # to avoid division by zero
        cx = (bbox[0] + bbox[2] / 2.0) / (imgsz[1] + eps)
        cy = (bbox[1] + bbox[3] / 2.0) / (imgsz[0] + eps)
        w = bbox[2] / (imgsz[1] + eps)
        h = bbox[3] / (imgsz[0] + eps)
        return cls(
            bbox=np.array([cx, cy, w, h, *bbox[4:]], dtype=np.float32),
            imgsz=imgsz,
            index=index,
            path=path,
            base_dir=base_dir
        )

    # --- Computation ---
    def area(self, imgsz: tuple[int, int] = None) -> float:
        """Return the area of the bounding box.

        Args:
            imgsz (tuple[int, int], optional): Image size as (H, W). If None,
                use the stored ``imgsz``. Defaults to None.
        """
        _, _, w, h = self.coords
        imgsz = imgsz or self.imgsz
        h = h * imgsz[0]
        w = w * imgsz[1]
        return h * w

    def center(self, imgsz: tuple[int, int] = None) -> ndarray:
        """Return the bounding box center of shape (2,).

         Args:
            imgsz (tuple[int, int], optional): Image size as (H, W). If None,
                use the stored ``imgsz``. Defaults to None.
        """
        imgsz = imgsz or self.imgsz
        cx, cy = self.coords[0:2]
        cx = cx * imgsz[1]
        cy = cy * imgsz[0]
        return np.array([cx, cy], dtype=np.float32)

    def corners(self, imgsz: tuple[int, int] = None) -> ndarray:
        """Return the bounding box corners in [x1, y1, x2, y1, x2, y2, x1, y2]
        format.

        Args:
            imgsz (tuple[int, int], optional): Image size as (H, W). If None,
                use the stored ``imgsz``. Defaults to None.
        """
        xyxy = self.xyxy(imgsz)
        x1, y1, x2, y2 = xyxy[0:4]
        # Standard order: top-left, top-right, bottom-right, bottom-left
        return np.array([x1, y1, x2, y1, x2, y2, x1, y2])

    def corners_pts(self, imgsz: tuple[int, int] = None) -> ndarray:
        """Return the bounding box corners as 4 points of shape (4, 2).

        Args:
            imgsz (tuple[int, int], optional): Image size as (H, W). If None,
                use the stored ``imgsz``. Defaults to None.
        """
        corners = self.corners(imgsz)
        return corners.reshape(4, 2)

    # --- Transformation ---
    def cxcywhn(self) -> ndarray:
        """Return the bounding box array of shape (8,) in CXCYWHN format."""
        return self.bbox

    def cxcywh(self, imgsz: tuple[int, int] = None) -> ndarray:
        """Return the bounding box array of shape (8,) in CXCYWH format.

        Args:
            imgsz (tuple[int, int], optional): Image size as (H, W). If None,
                use the stored ``imgsz``. Defaults to None.
        """
        cx, cy, w, h = self.coords
        imgsz = imgsz or self.imgsz
        img_h, img_w = imgsz
        cx = cx * img_w
        cy = cy * img_h
        w = w * img_w
        h = h * img_h
        return np.array([cx, cy, w, h, *self.bbox[4:]], dtype=np.float32)

    def xyxy(self, imgsz: tuple[int, int] = None) -> ndarray:
        """Return the bounding box array of shape (8,) in XYXY format.

        Args:
            imgsz (tuple[int, int], optional): Image size as (H, W). If None,
                use the stored ``imgsz``. Defaults to None.
        """
        cx, cy, w, h = self.coords
        imgsz = imgsz or self.imgsz
        img_h, img_w = imgsz
        x1 = (cx - w / 2.0) * img_w
        y1 = (cy - h / 2.0) * img_h
        x2 = (cx + w / 2.0) * img_w
        y2 = (cy + h / 2.0) * img_h
        return np.array([x1, y1, x2, y2, *self.bbox[4:]], dtype=np.float32)

    def xywh(self, imgsz: tuple[int, int] = None) -> ndarray:
        """Return the bounding box array of shape (8,) in XYWH format.

        Args:
            imgsz (tuple[int, int], optional): Image size as (H, W). If None,
                use the stored ``imgsz``. Defaults to None.
        """
        cx, cy, w, h = self.coords
        imgsz = imgsz or self.imgsz
        img_h, img_w = imgsz
        x = (cx - w / 2.0) * img_w
        y = (cy - h / 2.0) * img_h
        w = w * img_w
        h = h * img_h
        return np.array([x, y, w, h, *self.bbox[4:]], dtype=np.float32)


@dataclass
class BBoxes(Data):
    """Data structure for handling multiple bounding boxes.

    Attributes:
        bbox (ndarray): Bounding box array of shape (N, 8+) and in CXCYWHN format.
           (i.e., YOLO). The data format is: [cx, cy, w, h, angle, class_id,
            score, track_id].
        imgsz (tuple[int, int]): Size of the corresponding image as (H, W).
        path (Path, optional): Path to the label file. Defaults to None.
        base_dir (Path, optional): Base directory for relative paths. This is
            useful to resolve other files related to the data. Defaults to None.
    """

    bbox: ndarray
    imgsz: tuple[int, int]
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
        if len(self.imgsz) != 2:
            raise ValueError(
                f"Expected 'imgsz' to be a tuple of length 2, "
                f"but got {self.imgsz}."
            )
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
    def from_xyxy(
        cls,
        bbox: ndarray,
        imgsz: tuple[int, int],
        path: PathLike | None = None,
        base_dir: PathLike | None = None
    ) -> "BBoxes":
        """Create bounding boxes from XYXY format.

        Args:
            bbox (ndarray): Bounding box array of shape (N, 4+) in XYXY format.
            imgsz (tuple[int, int]): Size of the corresponding image as (H, W).
            path (PathLike, optional): Path to the label file. Defaults to None.
            base_dir (PathLike, optional): Base directory for relative paths.
                Defaults to None.

        Returns:
            BBoxes: Created bounding boxes instance.
        """
        eps = 1e-7  # to avoid division by zero
        cx = ((bbox[:, 0] + bbox[:, 2]) / 2.0) / (imgsz[1] + eps)
        cy = ((bbox[:, 1] + bbox[:, 3]) / 2.0) / (imgsz[0] + eps)
        w = (bbox[:, 2] - bbox[:, 0]) / (imgsz[1] + eps)
        h = (bbox[:, 3] - bbox[:, 1]) / (imgsz[0] + eps)
        return cls(
            bbox=np.array([cx, cy, w, h, *bbox[:, 4:].T], dtype=np.float32).T,
            imgsz=imgsz,
            path=path,
            base_dir=base_dir
        )

    @classmethod
    def from_xywh(
        cls,
        bbox: ndarray,
        imgsz: tuple[int, int],
        path: PathLike | None = None,
        base_dir: PathLike | None = None
    ) -> "BBoxes":
        """Create bounding boxes from XYWH format.

        Args:
            bbox (ndarray): Bounding box array of shape (N, 4+) in XYWH format.
            imgsz (tuple[int, int]): Size of the corresponding image as (H, W).
            path (PathLike, optional): Path to the label file. Defaults to None.
            base_dir (PathLike, optional): Base directory for relative paths.
                Defaults to None.

        Returns:
            BBoxes: Created bounding boxes instance.
        """
        eps = 1e-7  # to avoid division by zero
        cx = (bbox[:, 0] + bbox[:, 2] / 2.0) / (imgsz[1] + eps)
        cy = (bbox[:, 1] + bbox[:, 3] / 2.0) / (imgsz[0] + eps)
        w = bbox[:, 2] / (imgsz[1] + eps)
        h = bbox[:, 3] / (imgsz[0] + eps)
        return cls(
            bbox=np.array([cx, cy, w, h, *bbox[:, 4:].T], dtype=np.float32).T,
            imgsz=imgsz,
            path=path,
            base_dir=base_dir
        )

    @classmethod
    def from_bbox_list(
        cls,
        bbox_list: list[BBox],
        imgsz: tuple[int, int],
        path: PathLike | None = None,
        base_dir: PathLike | None = None,
    ) -> "BBoxes":
        """Create bounding boxes from a list of bounding boxes.

        Args:
            bbox_list (list[BBox]): List of BBox instances.
            imgsz (tuple[int, int]): Size of the corresponding image as (H, W).
            path (PathLike, optional): Path to the label file. Defaults to None.
            base_dir (PathLike, optional): Base directory for relative paths.
                Defaults to None.

        Returns:
            BBoxes: Created bounding boxes instance.
        """
        return cls(
            bbox=np.array([b.bbox for b in bbox_list], dtype=np.float32),
            imgsz=imgsz,
            path=path,
            base_dir=base_dir
        )

    # --- Computation ---
    def area(self, imgsz: tuple[int, int] = None) -> ndarray:
        """Return the area of the bounding box.

        Args:
            imgsz (tuple[int, int], optional): Image size as (H, W). If None,
                use the stored ``imgsz``. Defaults to None.
        """
        _, _, w, h = self.coords.T
        imgsz = imgsz or self.imgsz
        h = h * imgsz[0]
        w = w * imgsz[1]
        return h * w

    def center(self, imgsz: tuple[int, int] = None) -> ndarray:
        """Return the bounding box center of shape (N, 2).

        Args:
            imgsz (tuple[int, int], optional): Image size as (H, W). If None,
                use the stored ``imgsz``. Defaults to None.
        """
        imgsz = imgsz or self.imgsz
        cx, cy = self.coords[:, 0:2].T
        cx = cx * imgsz[1]
        cy = cy * imgsz[0]
        return np.array([cx, cy], dtype=np.float32).T

    def corners(self, imgsz: tuple[int, int] = None) -> ndarray:
        """Return the bounding box corners in [x1, y1, x2, y1, x2, y2, x1, y2]
        format.

        Args:
            imgsz (tuple[int, int], optional): Image size as (H, W). If None,
                use the stored ``imgsz``. Defaults to None.
        """
        xyxy = self.xyxy(imgsz)
        x1, y1, x2, y2 = xyxy[:, 0:4].T
        # Standard order: top-left, top-right, bottom-right, bottom-left
        return np.array([x1, y1, x2, y1, x2, y2, x1, y2], dtype=np.float32).T

    def corners_pts(self, imgsz: tuple[int, int] = None) -> ndarray:
        """Return the bounding box corners as 4 points of shape (N, 4, 2).

        Args:
            imgsz (tuple[int, int], optional): Image size as (H, W). If None,
                use the stored ``imgsz``. Defaults to None.
        """
        corners = self.corners(imgsz)
        return corners.reshape(-1, 4, 2)

    # --- Transformation ---
    def cxcywhn(self) -> ndarray:
        """Return the bounding box in CXCYWHN format."""
        return self.bbox

    def cxcywh(self, imgsz: tuple[int, int] = None) -> ndarray:
        """Return the bounding box array of shape (N, 8) in CXCYWH format.

        Args:
            imgsz (tuple[int, int], optional): Image size as (H, W). If None,
                use the stored ``imgsz``. Defaults to None.
        """
        imgsz = imgsz or self.imgsz
        img_h, img_w = imgsz
        cx, cy, w, h = self.coords.T
        rest = self.bbox[:, 4:]
        cx = cx * img_w
        cy = cy * img_h
        w = w * img_w
        h = h * img_h
        return np.column_stack((cx, cy, w, h, rest)).astype(np.float32)

    def xyxy(self, imgsz: tuple[int, int] = None) -> ndarray:
        """Return the bounding box array of shape (N, 8) in XYXY format.

        Args:
            imgsz (tuple[int, int], optional): Image size as (H, W). If None,
                use the stored ``imgsz``. Defaults to None.
        """
        imgsz = imgsz or self.imgsz
        img_h, img_w = imgsz
        cx, cy, w, h = self.coords.T
        rest = self.bbox[:, 4:]
        x1 = (cx - w / 2) * img_w
        y1 = (cy - h / 2) * img_h
        x2 = (cx + w / 2) * img_w
        y2 = (cy + h / 2) * img_h
        return np.column_stack((x1, y1, x2, y2, rest)).astype(np.float32)

    def xywh(self, imgsz: tuple[int, int] = None) -> ndarray:
        """Return the bounding box array of shape (N, 8) in XYWH format.

        Args:
            imgsz (tuple[int, int], optional): Image size as (H, W). If None,
                use the stored ``imgsz``. Defaults to None.
        """
        imgsz = imgsz or self.imgsz
        img_h, img_w = imgsz
        cx, cy, w, h = self.coords.T
        rest = self.bbox[:, 4:]
        x = (cx - w / 2) * img_w
        y = (cy - h / 2) * img_h
        h = h * img_h
        w = w * img_w
        return np.column_stack((x, y, w, h, rest)).astype(np.float32)

# endregion


# ==============================================================================
# region TRANSFORMATION
# ==============================================================================

# --- Casting ---


# --- Encoding ---


# --- Standardization ---


# --- Structural ---

def to_2d_bbox(bbox: ndarray | list | tuple) -> ndarray:
    """Standardize bounding boxes as a 2D array.

    Args:
        bbox (ndarray | list | tuple): Bounding boxes to standardize. Can be a
            single or a batch of bounding boxes.

    Returns:
        ndarray: Standardized bounding boxes of shape (N, M).

    Raises:
        ValueError: If ``bbox``'s type is unsupported.
        TypeError: If list/tuple elements have inconsistent shapes.
    """
    # Convert a list or tuple into an array
    if isinstance(bbox, (list, tuple)):
        try:
            bbox = np.array(bbox, dtype=np.float32)
        except ValueError:
            # Handle jagged arrays (e.g., one box has 7 elements, another has 8)
            raise ValueError(
                "Expected all elements in 'bbox' to have the same shape."
            )

    # Validate inputs
    if not isinstance(bbox, ndarray):
        raise TypeError(
            f"Expected 'bbox' to be an array, but got {type(bbox).__name__}."
        )

    # Handle various shapes
    if bbox.ndim == 1:
        return bbox[np.newaxis, :]   # [5+] -> [1, 5+]
    elif bbox.ndim == 3:
        return np.squeeze(bbox)      # [1, N, 5+] -> [N, 5+]

    return bbox


# --- Statistical ---


# --- Geometric ---

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion

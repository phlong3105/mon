#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Keypoint Data Structures.

This module provides data structures and utilities for handling keypoints.
"""

from __future__ import annotations

__all__ = [
    "Skeleton",
    "Skeletons",
]

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from numpy import ndarray

from mon.core.path import Path
from mon.core.typing import Float4
from mon.core.utils import is_list_of, is_valid_str
from .data import Data
from .size import Size


# ==============================================================================
# region CONSTANTS
# ==============================================================================

COCOSkeleton = [
    (16, 14),
    (14, 12),
    (17, 15),
    (15, 13),
    (12, 13),
    (6, 12),
    (7, 13),
    (6, 7),
    (6, 8),
    (7, 9),
    (8, 10),
    (9, 11),
    (2, 3),
    (1, 2),
    (1, 3),
    (2, 4),
    (3, 5),
    (4, 6),
    (5, 7),
]
COCOSkeleton = [(a - 1, b - 1) for a, b in COCOSkeleton]

# endregion


# ==============================================================================
# region BASE CLASSES
# ==============================================================================

@dataclass
class Skeleton(Data):
    """Data structure for handling a single skeleton keypoint.

    Attributes:
        class_id (int): Class ID of the object.
        keypoints (ndarray): Keypoint coordinates of shape (N, 2) where N is
            the number of keypoints.
        score (float): Confidence score of the detection.
        imgsz (Size): Size of the corresponding image as (H, W).
        index (int, optional): Index of the bounding box in the image.
            Defaults to -1.
        path (Path | None, optional): Path to the label file. Defaults to None.
        base_dir (Path | None, optional): Base directory for relative paths.
            This is useful to resolve other files related to the data.
            Defaults to None.
    """

    class_id: int
    keypoints: ndarray
    score: float
    imgsz: Size
    index: int = -1
    path: Path | None = None
    base_dir: Path | None = None

    # --- Lifecycle & Initialization ---
    def __post_init__(self):
        """Perform post-initialization tasks.

        Raises:
            TypeError: If ``keypoints`` is not a 2D ndarray.
        """
        # Validate inputs
        if (
            not isinstance(self.keypoints, ndarray)
            or self.keypoints.ndim < 2
            or self.keypoints.shape[1] < 2
            or any(self.keypoints < 0)
        ):
            raise TypeError(
                f"expected bbox to be a non-negative 2D array of shape (N, 2), "
                f"got {type(self.keypoints).__name__} and {self.keypoints.shape}."
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

    def __getitem__(self, index: int) -> ndarray:
        """Return the element at the given ``index``."""
        return self

    # --- Properties ---
    @property
    def data(self) -> ndarray:
        """Return the underlying data as a single array of shape (...).
        The data format is: [class_id, keypoints (1D), score].
        """
        t = [self.class_id, *list(self.keypoints.ravel()), self.score]
        return  np.ndarray(t, dtype=np.float32)

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the data shape."""
        return self.keypoints.shape

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
        return len(self.keypoints) == 0

    # --- Creation ---

    # --- Computation ---

    # --- Transformation ---
    def decode(self) -> tuple[ndarray, ndarray]:
        """Decode keypoints into xy and visibility arrays."""
        kpts = np.asarray(self.keypoints)

        if kpts.ndim == 2:
            if kpts.shape[1] < 2:
                return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32)
            xy = kpts[:, :2].astype(np.float32)
            if kpts.shape[1] >= 3:
                vis = kpts[:, 2].astype(np.float32)
            else:
                vis = np.ones((kpts.shape[0],), dtype=np.float32)
            return xy, vis

        if kpts.ndim == 1:
            if kpts.size % 3 == 0:
                kpts_ = kpts.reshape(-1, 3)
                return kpts_[:, :2].astype(np.float32), kpts_[:, 2].astype(np.float32)
            if kpts.size % 2 == 0:
                kpts_ = kpts.reshape(-1, 2)
                return kpts_.astype(np.float32), np.ones((kpts_.shape[0],), dtype=np.float32)

        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    # --- Visualization ---
    def _get_draw_params(self, imgsz: Size, input_sz: int = 640) -> Float4:
        """Calculate drawing parameters based on the image size and a reference
        input size.

        Args:
            imgsz (Size): Size of the image as (H, W).
            input_sz (int, optional): Reference input size for scaling.
                Defaults to 640.
        """
        h, w = imgsz.hw
        min_side = max(1, min(h, w))
        base = float(min_side / input_sz)
        font_scale = max(0.9, 0.95 * base)
        text_thickness = max(2, int(round(1.6 * base)))
        point_radius = max(2, int(round(3.0 * base)))
        skeleton_thickness = max(2, int(round(2.0 * base)))
        return font_scale, text_thickness, point_radius, skeleton_thickness

    def draw(
        self,
        image: ndarray,
        label: str | int | None = None,
        imgsz: Size | None = None,
        input_sz: int = 640,
        draw_skeleton: bool = True,
    ) -> ndarray:
        """Draw the bounding box on the given image.

        Args:
            image (ndarray): The image on which to draw the bounding box.
            label (str | int | None, optional): The label to display on the
                bounding box. Can be either a string or a tracking ID.
                Defaults to None.
            imgsz (Size | None, optional): Image size as (H, W). If None, use
                the stored ``self.imgsz``. Defaults to None.
            input_sz (int, optional): Reference input size for scaling.
                Defaults to 640.
            draw_skeleton (bool, optional): Whether to draw the skeleton.
                Defaults to True.

        Returns:
            ndarray: The image with the bounding box drawn on it.
        """
        # 1. Prepare drawing parameters
        imgsz = imgsz or self.imgsz
        font_scale, text_thickness, point_radius, line_thickness \
            = self._get_draw_params(imgsz, input_sz)

        # 2. Prepare keypoints
        kpts, vis = self.decode()
        kpts = kpts.astype(np.int32)
        vis = vis.astype(np.float32)

        # 3. Draw the keypoints
        for idx, (x, y) in enumerate(kpts):
            if vis[idx]:
                cv2.circle(image, (int(x), int(y)), point_radius, (0, 255, 0), -1)

        # 4. Draw the skeleton
        if draw_skeleton:
            for a, b in COCOSkeleton:
                if a < len(kpts) and b < len(kpts) and vis[a] and vis[b]:
                    xa, ya = kpts[a]
                    xb, yb = kpts[b]
                    cv2.line(
                        image,
                        (int(xa), int(ya)),
                        (int(xb), int(yb)),
                        (255, 128, 0),
                        line_thickness
                    )

        min_xy = np.min(kpts[vis], axis=0) if np.any(vis) else np.min(kpts, axis=0)
        min_xy = np.maximum(min_xy, 0)
        label = str(label or self.class_id)
        text = f"{label} {self.score:.2f}"
        cv2.putText(
            image,
            text,
            (int(min_xy[0]), int(max(min_xy[1] - 5, 10))),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            text_thickness,
            cv2.LINE_AA,
        )

        return image


@dataclass
class Skeletons(Data):
    """Data structure for handling multiple skeleton keypoints in an image.

    Attributes:
        bbox (ndarray): Bounding box array of shape (N, 8+) and in CXCYWHN format.
           (i.e., YOLO). The data format is: [cx, cy, w, h, angle, class_id,
            score, track_id].
        imgsz (Size): Size of the corresponding image as (H, W).
        path (Path | None, optional): Path to the label file. Defaults to None.
        base_dir (Path | None, optional): Base directory for relative paths.
            This is useful to resolve other files related to the data.
            Defaults to None.
    """

    skeletons: list[Skeleton]
    imgsz: Size
    path: Path | None = None
    base_dir: Path | None = None

    # --- Lifecycle & Initialization ---
    def __post_init__(self):
        """Perform post-initialization tasks.

        Raises:
            TypeError: If ``skeletons`` is not a list of Skeleton instances.
        """
        # Validate inputs
        if not is_list_of(self.skeletons, Skeleton):
            raise TypeError(
                f"expected skeletons to be a list of Skeleton instances, "
                f"got {type(self.skeletons).__name__} and "
                f"{[type(s).__name__ for s in self.skeletons]}."
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
        return len(self.skeletons)

    def __getitem__(self, index: int) -> Skeleton:
        """Return the element at the given ``index``."""
        return self.skeletons[index]

    # --- Retrieval ---
    @property
    def data(self) -> ndarray:
        """Return the underlying data."""
        return np.array([s.data for s in self.skeletons], dtype=np.float32)

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the data shape."""
        n = self.__len__()
        m = self.skeletons[0].shape if self.is_empty else 0
        return n, *m

    @property
    def class_id(self) -> ndarray:
        """Return the class ID of shape (N,)."""
        return np.array([b.class_id for b in self.skeletons], dtype=np.int32)

    @property
    def coords(self) -> ndarray:
        """Return the bounding box coordinates of shape (N, 4) in CXCYWHN format."""
        return np.array([s.keypoints for s in self.skeletons], dtype=np.float32)

    @property
    def scores(self) -> ndarray:
        """Return the confidence score of shape (N,)."""
        return np.array([b.score for b in self.skeletons], dtype=np.float32)

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
        return len(self.skeletons) == 0

    # --- Creation ---

    # --- Computation ---

    # --- Transformation ---

    # --- Visualization ---


# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Object instance data structures.

This module provides base classes and mixins for object instances.
"""

from __future__ import annotations

__all__ = [
    "Instance",
]

import numpy as np

from mon.core.dtypes import bbox as B, image as I
from mon.core.pathlib import Path
from ..array import TensorOrArray


# ==============================================================================
# region CONSTANTS
# ==============================================================================


# endregion


# ==============================================================================
# region TYPE DEFINITIONS & PROTOCOLS
# ==============================================================================

# --- Type Aliases ---


# --- Protocols ---


# endregion


# ==============================================================================
# region BASE CLASSES & MIXINS
# ==============================================================================

# --- Base Classes ---


# --- Mixins ---


# endregion


# ==============================================================================
# region CONCRETE IMPLEMENTATIONS
# ==============================================================================

class Instance(TensorOrArray):
    """Instance annotation base class.

    Extend ``TensorOrArray`` to encapsulate per-object annotations and provide
    convenient accessors. Encapsulate these kinds of annotations:
        - ``bbox``: Bounding box, support both OBB and HBB (*primary).
        - ``mask``: Instance mask (i.e., pixels that belong to the object).
        - ``polygon``: Points to outline the object's shape.
        - ``keypoints``: Points on key parts, like eyes or joints.
        - ``cuboid``: 3D bounding box with depth.
        - ``cls``: Type of object, like "car".

    Expect bounding boxes in the following format:
        <cx, cy, w, h, a, conf, cls, id, ...>
    where:
        - <cx, cy, w, h> are the bounding box coordinates in CXCYWHN format.
        - <a> is the angle.
        - <conf> is the confidence score (optional).
        - <cls> is the class ID (optional).
        - <id> is the tracking ID (optional).

    Attributes:
        _data (numpy.ndarray): Bounding box, formatted as a numpy.ndarray of
            shape (7+) and in CXCYWHN format.
        _mask (numpy.ndarray | None): Instance mask, formatted as a
            numpy.ndarray of shape (H, W, C) and pixel values ranging from 0 to
            255.
        _imgsz (tuple[int, int]): Image size as (H, W).
        _image_path (Path | None): Associated image file path.
        _root (Path | None): Root directory for the label file.
    """

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        data      : np.ndarray,
        imgsz     : tuple[int, int] | None = None,
        mask      : np.ndarray | None      = None,
        image_path: Path | str | None      = None,
        root      : Path | str | None      = None,
    ):
        """Initialize a new instance.

        Args:
            data: Bounding box, formatted as a numpy.ndarray of shape (7+) and
                in CXCYWHN format.
            imgsz: Image size as (H, W). Defaults to None.
            mask: Instance mask, formatted as a numpy.ndarray of shape (H, W, C)
                and pixel values ranging from 0 to 255. Defaults to None.
            image_path: Associated image file path. Defaults to None.
            root: Root directory for the label file. Defaults to None.

        Raises:
            ValueError: If ``imgsz`` is not provided and ``image_path`` is not
                valid.
        """
        # Validate paths
        image_path = Path(image_path).normalize(exist=True) if image_path else None
        root       = Path(root).normalize(exist=True)       if root       else None
        
        # Infer image size if not provided
        if imgsz is None:
            if image_path and image_path.is_image_file(exist=True):
                imgsz = I.read_size(image_path)
            else:
                raise ValueError(f"Expected either 'imgsz' or a valid 'image_path', "
                                 f"but got both None.")
        else:
            imgsz = I.imgsz(imgsz)
        
        # Call the setter to ensure type validation on init
        self._imgsz      = imgsz
        self._mask       = mask
        self._image_path = image_path
        self._root       = root
        self.data        = data
        
        # Continue the initialization chain
        super().__init__(data=self.data)
    
    # --- Properties ---
    @property
    def data(self) -> np.ndarray:
        """Return the bounding box, formatted as a numpy.ndarray of shape (7+)
        and in CXCYWHN format.
        """
        return self._data
    
    @data.setter
    def data(self, value: np.ndarray):
        """Set the bounding box data.

        Args:
            value: Bounding box, formatted as a numpy.ndarray of shape (7+) and
                in CXCYWHN format.

        Raises:
            TypeError: If ``value`` is not a numpy.ndarray.
            ValueError: If ``value`` has incorrect shape.
        """
        # Ensure we are working with a float ndarray for normalization precision
        if not isinstance(value, np.ndarray):
            value = np.array(value, dtype=np.float32)
        
        if value.ndim != 1 or value.shape[0] < 7:
            raise ValueError(f"Expected 'data' to be a numpy.ndarray of shape (7+), "
                             f"but got {value.shape}.")
        
        # Internal conversion logic
        if B.is_xywh(value, self._imgsz):
            value = B.xywh_to_cxcywhn(value, self._imgsz)[0]
        elif B.is_xyxy(value):
            value = B.xyxy_to_cxcywhn(value, self._imgsz)[0]
        
        self._data = value
    
    @property
    def mask(self) -> np.ndarray:
        """Return the instance segmentation mask."""
        return self._mask
    
    @mask.setter
    def mask(self, value: np.ndarray):
        """Set or update the instance segmentation mask.

        Args:
            value: Mask array of shape (H, W, C) or compatible shape.
        """
        self._mask = value
    
    @property
    def imgsz(self) -> tuple[int, int]:
        """Return the image size as (H, W)."""
        return self._imgsz
    
    @property
    def image_path(self) -> Path:
        """Return the associated image path."""
        return self._image_path
    
    @property
    def root(self) -> Path:
        """Return the root directory for the label file."""
        return self._root
    
    @property
    def conf(self) -> float:
        """Return the confidence score."""
        return float(self.data[5])
    
    @property
    def cls(self) -> int:
        """Return the class identifier."""
        return int(self.data[6])
    
    @property
    def id(self) -> int:
        """Return the tracking identifier."""
        return int(self.data[7])
    
    @property
    def cxcywhn(self) -> np.ndarray:
        """Return the bounding box, formatted as a numpy.ndarray of shape (7+)
        and in CXCYWHN format.
        """
        return self.data
    
    def xyxy(self, imgsz: tuple[int, int] = None) -> np.ndarray:
        """Convert the bounding box from CXCYWHN to XYXY format.

        Args:
            imgsz: Image size as (H, W). If omitted, uses the stored ``_imgsz``.
                Defaults to None.

        Returns:
            Bounding box in XYXY format.
        """
        imgsz = I.imgsz(imgsz) if imgsz is not None else self._imgsz
        return B.cxcywhn_to_xyxy(self.data, imgsz)[0]
    
    def xywh(self, imgsz: tuple[int, int] = None) -> np.ndarray:
        """Convert the bounding box from CXCYWHN to XYWH format.

        Args:
            imgsz: Image size as (H, W). If omitted, uses the stored ``_imgsz``.
                Defaults to None.

        Returns:
            Bounding box in XYWH format.
        """
        imgsz = I.imgsz(imgsz) if imgsz is not None else self._imgsz
        return B.cxcywhn_to_xywh(self.data, imgsz)[0]
    
    @property
    def area(self) -> float:
        """Compute the area of the bounding box in pixels."""
        # Using normalization factors: (W_norm * W_img) * (H_norm * H_img)
        h0, w0 = self._imgsz
        return float((self.data[2] * w0) * (self.data[3] * h0))

# endregion

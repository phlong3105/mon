#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Instance annotation base classes and mixins.

This module provides the base classes and mixins for instance annotations.
"""

__all__ = [
    "Instance",
]

import numpy as np

from mon.core.pathlib import Path
from .. import bbox as B, image as I
from ..array import TensorOrArray


# ==============================================================================
# TYPE DEFINITIONS & PROTOCOLS (Interfaces)
# ==============================================================================

# --- Type Aliases ---


# --- Structural Protocols ---


# ==============================================================================
# BASE CLASSES & MIXINS (Behaviors)
# ==============================================================================

# --- Structural Bases ---


# --- Lifecycle Mixins ---


# --- Compute Mixins ---


# ==============================================================================
# CONCRETE IMPLEMENTATIONS (The Concrete Classes)
# ==============================================================================

# --- Primary Data Types ---
class Instance(TensorOrArray):
    """A base class for instance annotations.
    
    Extend TensorOrArray to encapsulate per-object annotations (i.e., bbox,
    mask, polygon, keypoints, cuboid, class, confidence, tracking id) and
    provide convenient accessors.

    One instance can have these kinds of annotations (i.e., attributes):
        - bbox     : bounding box, support both OBB and HBB (*primary).
        - mask     : instance mask (i.e., pixels that belong to the object).
        - polygon  : points to outline the object's shape.
        - keypoints: points on key parts, like eyes or joints.
        - cuboid   : 3D bounding box with depth.
        - cls      : the type of object, like "car".
    
    The bounding boxes are expected to in the following format:
            <cx, cy, w, h, a, cls, conf, id, ...>
        where:
            - <cx, cy, w, h> are the bounding box coordinates in CXCYWHN format.
            - <a> is the angle.
            - <cls> is the class ID (optional).
            - <conf> is the confidence score (optional).
            - <id> is the tracking ID (optional).
    
    Notes:
        I am in the process of adding more annotations to this class, so it
        may subject to changes in the future.
    
    Attributes:
        _data (np.ndarray): A bounding box, formatted as a numpy.ndarray of
            dimensions (7+) and in CXCYWHN format.
        _mask (np.ndarray): Instance mask, formatted as a numpy.ndarray of
            dimensions (H, W, C) and pixel values ranging from 0 to 255.
        _imgsz (tuple[int, int]): Image size as (H, W) used for conversions.
        _image_path (Path): Associated image file path.
        _root (Path): Root directory for the label file.
    """

    def __init__(
        self,
        data      : np.ndarray,
        imgsz     : tuple[int, int],
        mask      : np.ndarray = None,
        image_path: Path       = None,
        root      : Path       = None,
    ):
        """Initialize a new instance.
        
        Args:
            data: A bounding box, formatted as a numpy.ndarray of dimensions
                (7+) and in CXCYWHN format.
            imgsz: Image size as (H, W).
            mask: Instance mask, formatted as a numpy.ndarray of dimensions
                (H, W, C) and pixel values ranging from 0 to 255.
            image_path: Associated image file path.
            root: Root directory for the label file.

        Raises:
            ValueError: If neither ``imgsz`` nor a valid ``image_path`` is provided.
            TypeError: If ``data`` is not a numpy.ndarray.
        """
        # Validate and set imgsz
        if imgsz is None and Path(image_path).is_image_file(exist=True):
            imgsz = I.read_size(image_path)
        if imgsz is None:
            raise ValueError("Either ``imgsz`` or a valid ``image_path`` must be provided to determine the original image size.")
        imgsz = I.imgsz(imgsz)
        
        # Initialize parent classes and assign attributes
        self._imgsz      = imgsz
        self._mask       = mask
        self._image_path = Path(image_path) if image_path is not None else None
        self._root       = Path(root)       if root       is not None else None
        super().__init__(data=data)  # This will call the data setter
    
    # --- Properties ---
    @property
    def data(self) -> np.ndarray:
        """Return the bounding box in CXCYWHN format."""
        return self._data
    
    @data.setter
    def data(self, data: np.ndarray):
        """Set the bounding box data.
        
        Args:
            data: A bounding box, formatted as a numpy.ndarray of dimensions
                (7+) and in CXCYWHN format.
                
        Raises:
            TypeError: If ``data`` is not a numpy.ndarray.
            ValueError: If ``data`` has incorrect shape.
        """
        if not isinstance(data, np.ndarray):
            raise TypeError(f"``data`` must be a numpy.ndarray, got {type(data)}.")
        if data.ndim != 1 or data.shape[0] < 7:
            raise ValueError(f"``data`` must be of shape (7+), got {data.shape}.")
        if B.is_xywh(data, self._imgsz):
            data = B.xywh_to_cxcywhn(data[None, :], self._imgsz)[0]
        elif B.is_xyxy(data, self._imgsz):
            data = B.xyxy_to_cxcywhn(data[None, :], self._imgsz)[0]
        
        self._data = data
    
    @property
    def mask(self) -> np.ndarray:
        """Return the instance segmentation mask."""
        return self._mask
    
    @mask.setter
    def mask(self, mask: np.ndarray):
        """Set or update the instance segmentation mask.

        Args:
            mask: Mask array of shape (H, W, C) or compatible shape.
        """
        self._mask = mask
    
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
        """Return the bounding box in CXCYWHN format."""
        return self.data
    
    def xyxy(self, imgsz: tuple[int, int] = None) -> np.ndarray:
        """Convert the bounding box from CXCYWHN to XYXY format.

        Args:
            imgsz: Image size as (H, W). If omitted, uses the stored ``_imgsz``.

        Returns:
            Bounding box in XYXY format.
        """
        imgsz = I.imgsz(imgsz) if imgsz is not None else self.imgsz
        return B.cxcywhn_to_xyxy(self.data, imgsz)[0]
    
    def xywh(self, imgsz: tuple[int, int] = None) -> np.ndarray:
        """Convert the bounding box from CXCYWHN to XYWH format.

        Args:
            imgsz: Image size as (H, W). If omitted, uses the stored ``_imgsz``.

        Returns:
            Bounding box in XYWH format.
        """
        imgsz = I.imgsz(imgsz) if imgsz is not None else self.imgsz
        return B.cxcywhn_to_xywh(self.data, imgsz)[0]

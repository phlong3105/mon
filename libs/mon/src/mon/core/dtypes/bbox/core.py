#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Bounding box base classes and mixins.

This module provides the base classes and mixins for bounding box data.
"""

__all__ = [
    "BBox",
    "BBoxList",
]

import numpy as np

from mon.core.enum import BBoxFormat
from mon.core.pathlib import Path
from .. import image as I
from ..array import TensorOrArray
from ..base import DataLoadMixin


# ==============================================================================
# CONCRETE IMPLEMENTATIONS (The Concrete Classes)
# ==============================================================================

# --- Primary Data Types ---
class BBox(TensorOrArray):
    """A basic class for managing a bounding box.

    Extend TensorOrArray to handle a single bounding box and provide properties
    and methods related to bounding box conversions and accessors.

    Attributes:
        _data (np.ndarray): A bounding box, formatted as a numpy.ndarray of
            dimensions (7+) and in CXCYWHN format.
        _imgsz (tuple[int, int]): Image size as (H, W) used for conversions.
    """
    
    # --- Lifecycle & Initialization ---
    def __init__(self, data: np.ndarray, imgsz: tuple[int, int]):
        """Initialize a new instance.

        Args:
            data: A bounding box, formatted as a numpy.ndarray of dimensions
                (7+) and in CXCYWHN format.
            imgsz: Image size as (H, W).

        Raises:
            ValueError: If ``imgsz`` is not provided.
        """
        # Validate and set imgsz
        if imgsz is None:
            raise ValueError(f"``imgsz`` must be specified.")
        imgsz = I.imgsz(imgsz)
        
        # Initialize parent classes and assign attributes
        self._imgsz = imgsz
        super().__init__(data=data)  # This will call the data setter
    
    # ---- Properties ---
    @property
    def data(self) -> np.ndarray:
        """Return the bounding box, formatted as a numpy.ndarray of dimensions
        (7+) and in CXCYWHN format.
        """
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
        from .ops import is_xywh, is_xyxy, xywh_to_cxcywhn, xyxy_to_cxcywhn
        
        if not isinstance(data, np.ndarray):
            raise TypeError(f"``data`` must be a numpy.ndarray, got {type(data)}.")
        if data.ndim != 1 or data.shape[0] < 7:
            raise ValueError(f"``data`` must be of shape (7+), got {data.shape}.")
        if is_xywh(data, self._imgsz):
            data = xywh_to_cxcywhn(data[None, :], self._imgsz)[0]
        elif is_xyxy(data, self._imgsz):
            data = xyxy_to_cxcywhn(data[None, :], self._imgsz)[0]
        
        self._data = data
    
    @property
    def imgsz(self) -> tuple[int, int]:
        """Return the image size as (H, W)."""
        return self._imgsz
    
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
            imgsz: image size as (H, W). If omitted, uses the stored ``_imgsz``.

        Returns:
            A bounding box in XYXY format.
        """
        from .ops import cxcywhn_to_xyxy
        imgsz = I.imgsz(imgsz) if imgsz is not None else self.imgsz
        return cxcywhn_to_xyxy(self.data, imgsz)[0]
    
    def xywh(self, imgsz: tuple[int, int] = None) -> np.ndarray:
        """Convert the bounding box from CXCYWHN to XYWH format.

        Args:
            imgsz: Image size as (H, W). If omitted, uses the stored ``_imgsz``.

        Returns:
            A bounding box in XYWH format.
        """
        from .ops import cxcywhn_to_xywh
        imgsz = I.imgsz(imgsz) if imgsz is not None else self.imgsz
        return cxcywhn_to_xywh(self.data, imgsz)[0]
    
    @property
    def area(self) -> float:
        """Compute the area of the bounding box in pixels."""
        h0, w0  = self.imgsz
        w, h    = self.data[2:4]
        w      *= w0
        h      *= h0
        return float(w * h)


class BBoxList(TensorOrArray, DataLoadMixin):
    """A basic class for managing a list of bounding boxes.

    Extend TensorOrArray and DataLoadMixin to handle a batch of bounding boxes
    and provide properties and methods related to bounding box conversions,
    accessors, and loading from label files.

    Attributes:
        _data (np.ndarray): A batch of bounding boxes, formatted as a
            numpy.ndarray of dimensions (N, 7+) and in CXCYWHN.
        _imgsz (tuple[int, int]): Image size as (H, W).
        _path (Path): Label file path.
        _root (Path): Root directory of the label file.
        _fmt (BBoxFormat): Bounding box format in the label file.
        _cvt_fmt (BBoxFormat): Conversion code used when loading from file.
    """
    
    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        data   : np.ndarray | Path,
        imgsz  : tuple[int, int],
        path   : Path       = None,
        root   : Path       = None,
        fmt    : BBoxFormat = BBoxFormat.CXCYWHN,
        persist: bool       = True,
    ):
        """Initialize a new instance.

        Args:
            data: Either A batch of bounding boxes, formatted as a numpy.ndarray
                of dimensions (N, 7+) and in CXCYWHN; or a path to a label file.
            imgsz: Image size as (H, W), used for conversions.
            path: Label file path.
            root: Root directory of the label file.
            fmt: Bounding box format in the label file or conversion code.
            persist: If True, persist loaded data in memory. Defaults to True.
        """
        # Validate and set imgsz
        if imgsz is None:
            raise ValueError(f"``imgsz`` must be specified.")
        imgsz = I.imgsz(imgsz)
        
        # Validate data
        if isinstance(data, Path | str) and Path(data).is_txt_file(exist=True):
            path = data
            data = None
        elif not isinstance(data, np.ndarray):
            raise TypeError(f"``data`` must be a ``numpy.ndarray`` or a valid label file path, got {type(data)}.")
        
        # Validate and set path
        path = Path(path) if path is not None else None
        root = Path(root) if root is not None else None
        if data is None and path is not None:  # Load from path if data not provided
            if not path.is_txt_file(exist=True):
                raise FileNotFoundError(f"Label file not found: {path}.")
        
        # Validate
        if all(v is None for v in [data, path]):
            raise ValueError("Either ``data`` or a valid label ``path`` must be provided.")
        
        # Initialize parent classes and assign attributes
        self._imgsz = imgsz
        self._set_fmt(fmt)  # Set fmt and cvt_fmt in case of loading bounding boxes from file
        super().__init__(data=data, path=path, root=root, persist=persist)  # This will call the data setter
    
    # --- Properties ---
    @property
    def data(self) -> np.ndarray:
        """Return stored batch of bounding boxes, formatted as a numpy.ndarray
        of dimensions (N, 7+) and in CXCYWHN format.
        """
        return self._data if self._data is not None else self.load()
    
    @data.setter
    def data(self, data: np.ndarray):
        """Set the bounding boxes.

        Args:
            data: A batch of bounding boxes, formatted as a numpy.ndarray of
                dimensions (N, 7+) and in CXCYWHN format.

        Raises:
            TypeError: If ``data`` is not a numpy.ndarray.
            ValueError: If ``data`` is invalid.
        """
        from .ops import is_xywh, is_xyxy, xywh_to_cxcywhn, xyxy_to_cxcywhn
        
        if not isinstance(data, np.ndarray):
            raise TypeError(f"``data`` must be a numpy.ndarray, got {type(data)}.")
        if data.ndim != 2 or data.shape[0] < 7:
            raise ValueError(f"``data`` must be of shape (N, 7+), got {data.shape}.")
        if is_xywh(data, self._imgsz):
            data = xywh_to_cxcywhn(data[None, :], self._imgsz)
        elif is_xyxy(data, self._imgsz):
            data = xyxy_to_cxcywhn(data[None, :], self._imgsz)
        
        self._data = data
        
    @property
    def imgsz(self) -> tuple[int, int]:
        """Return the image size as (H, W)."""
        return self._imgsz
    
    @property
    def fmt(self) -> BBoxFormat:
        """Return label file format."""
        return self._fmt
    
    def _set_fmt(self, fmt: BBoxFormat):
        """Set the bounding box format in the label file and conversion code.

        Args:
            fmt: Bounding box format in the label file or conversion code.
        """
        fmt = BBoxFormat(fmt)
        if fmt in BBoxFormat.conversion_codes():
            from_fmt = BBoxFormat(fmt.value.split("_to_")[0])
        else:
            from_fmt = fmt

        # We default to CXCYWHN format for bboxes
        to_fmt = BBoxFormat.CXCYWHN
        if from_fmt != to_fmt:
            cvt_fmt = BBoxFormat(f"{from_fmt.value}_to_{to_fmt.value}")
        else:
            cvt_fmt = to_fmt

        self._fmt     = fmt
        self._cvt_fmt = cvt_fmt
    
    @property
    def conf(self) -> np.ndarray:
        """Return confidence scores for all bounding boxes."""
        return self.data[:, 5:6]

    @property
    def cls(self) -> np.ndarray:
        """Return class ids for all bounding boxes."""
        return self.data[:, 6:7]

    @property
    def id(self) -> np.ndarray:
        """Return tracking ids for all bounding boxes."""
        return self.data[:, 6:7]
    
    @property
    def cxcywhn(self) -> np.ndarray:
        """Return all bounding boxes in CXCYWHN format."""
        return self.data
    
    def xyxy(self, imgsz: tuple[int, int] = None) -> np.ndarray:
        """Convert all bounding boxes from CXCYWHN to XYXY format.

        Args:
            imgsz: Optional image size as (H, W). If omitted, uses the stored
                ``_imgsz``.

        Returns:
            Bounding boxes in XYXY format.
        """
        from .ops import cxcywhn_to_xyxy
        imgsz = I.imgsz(imgsz) if imgsz is not None else self.imgsz
        return cxcywhn_to_xyxy(self.data, imgsz)
    
    def xywh(self, imgsz: tuple[int, int] = None) -> np.ndarray:
        """Convert all bounding boxes from CXCYWHN to XYWH format.

        Args:
            imgsz: Optional image size as (H, W). If omitted, uses the stored
                ``_imgsz``.

        Returns:
            Bounding boxes in XYWH format.
        """
        from .ops import cxcywhn_to_xywh
        imgsz = I.imgsz(imgsz) if imgsz is not None else self.imgsz
        return cxcywhn_to_xywh(self.data, imgsz)
    
    # --- Data Loading ---
    def load(self, reload: bool = False) -> np.ndarray:
        """Load all bounding boxes from a label file.

        Args:
            reload: If True, force reloading even if data is already in memory.

        Returns:
            The loaded bounding boxes.
        """
        # Return the bbox if it is already loaded and not reloading
        if not reload and self._data is not None:
            return self._data

        # Load all bounding boxes from the label file
        from .io import load
        bbox = load(path=self._path, fmt=self._cvt_fmt, imgsz=self._imgsz)
        
        # Cache the bounding boxes if needed
        self._data = bbox
        return self._data
    
    def clear(self):
        """Clear the loaded bounding boxes from memory."""
        if (
            not self._persist
            and self._path is not None
            and self._path.is_txt_file(exist=True)
        ):
            self._data = None

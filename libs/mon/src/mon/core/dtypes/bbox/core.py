#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Bounding box data structures.

This module provides base classes and mixins for bounding boxes.
"""

from __future__ import annotations

__all__ = [
    "BBox",
    "BBoxList",
]

import numpy as np

from mon.core.enum import BBoxFormat
from mon.core.pathlib import Path
from .. import image as I
from ..array import TensorOrArray
from ..base import PersistentData


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

class BBox(TensorOrArray):
    """Single bounding box management class.

    Extend ``TensorOrArray`` to handle a single bounding box and provide
    properties and methods related to bounding box conversions and accessors.

    Attributes:
        _data (numpy.ndarray): Bounding box, formatted as a numpy.ndarray of
            shape (7+) and in CXCYWHN format.
        _imgsz (tuple[int, int]): Image size as (H, W).
    """
    
    # --- Lifecycle & Initialization ---
    def __init__(self, data: np.ndarray, imgsz: tuple[int, int]):
        """Initialize a new instance.

        Args:
            data: Bounding box, formatted as a numpy.ndarray of shape (7+) and
                in CXCYWHN format.
            imgsz: Image size as (H, W).
        """
        # Validate and set image size
        self._imgsz = I.imgsz(imgsz)
        
        # Call the setter to ensure type validation on init
        self.data = data
        
        # Continue the initialization chain
        super().__init__(data=self.data)
    
    # ---- Properties ---
    @property
    def data(self) -> np.ndarray:
        """Return the bounding box.

        Returns:
            Bounding box, formatted as a numpy.ndarray of shape (7+) and in
            CXCYWHN format.
        """
        return self._data
    
    @data.setter
    def data(self, value: np.ndarray):
        """Set the bounding box data.

        Args:
            value: Bounding box, formatted as a numpy.ndarray of shape (7+) and
                in CXCYWHN format.

        Raises:
            ValueError: If ``value`` is invalid.
        """
        from .ops import is_xywh, is_xyxy, xywh_to_cxcywhn, xyxy_to_cxcywhn
        
        # Ensure we are working with a float ndarray for normalization precision
        if not isinstance(value, np.ndarray):
            value = np.array(value, dtype=np.float32)
        
        if value.ndim != 1 or value.shape[0] < 7:
            raise ValueError(
                f"Expected 'data' to be a numpy.ndarray of shape (7+), "
                f"but got {value.shape}."
            )
        
        # Internal conversion logic
        if is_xywh(value, self._imgsz):
            value = xywh_to_cxcywhn(value, self._imgsz)[0]
        elif is_xyxy(value):
            value = xyxy_to_cxcywhn(value, self._imgsz)[0]
        
        self._data = value
        
    @property
    def imgsz(self) -> tuple[int, int]:
        """Return the image size."""
        return self._imgsz
    
    @property
    def conf(self) -> float:
        """Return the confidence score."""
        return float(self._data[5])
    
    @property
    def cls(self) -> int:
        """Return the class identifier."""
        return int(self._data[6])
    
    @property
    def id(self) -> int:
        """Return the tracking identifier."""
        return int(self._data[7]) if len(self._data) > 7 else -1

    @property
    def cxcywhn(self) -> np.ndarray:
        """Return the bounding box in CXCYWHN format."""
        return self._data
    
    def xyxy(self, imgsz: tuple[int, int] = None) -> np.ndarray:
        """Convert the bounding box from CXCYWHN to XYXY format.

        Args:
            imgsz: Image size as (H, W). If omitted, uses the stored
                ``_imgsz``. Defaults to None.

        Returns:
            Bounding box in XYXY format.
        """
        from .ops import cxcywhn_to_xyxy
        
        imgsz = I.imgsz(imgsz) if imgsz is not None else self._imgsz
        return cxcywhn_to_xyxy(self._data, imgsz)[0]
    
    def xywh(self, imgsz: tuple[int, int] = None) -> np.ndarray:
        """Convert the bounding box from CXCYWHN to XYWH format.

        Args:
            imgsz: Image size as (H, W). If omitted, uses the stored
                ``_imgsz``. Defaults to None.

        Returns:
            Bounding box in XYWH format.
        """
        from .ops import cxcywhn_to_xywh
        
        imgsz = I.imgsz(imgsz) if imgsz is not None else self._imgsz
        return cxcywhn_to_xywh(self._data, imgsz)[0]
    
    @property
    def area(self) -> float:
        """Compute the area of the bounding box in pixels."""
        # Using normalization factors: (W_norm * W_img) * (H_norm * H_img)
        h0, w0 = self._imgsz
        return float((self._data[2] * w0) * (self._data[3] * h0))


class BBoxList(PersistentData):
    """Bounding box list management class.

    Extend ``PersistentData`` to handle a batch of bounding boxes and provide
    properties and methods related to bounding box conversions, accessors, and
    loading from label files.

    Attributes:
        _data (numpy.ndarray): Batch of bounding boxes, formatted as a
            numpy.ndarray of shape (N, 7+) and in CXCYWHN.
        _imgsz (tuple[int, int]): Image size as (H, W).
        _path (Path): Label file path.
        _root (Path): Root directory of the label file.
        _fmt (BBoxFormat): Bounding box format in the label file.
        _cvt_fmt (BBoxFormat): Conversion code for bounding box format.
    """
    
    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        data   : np.ndarray | Path | str | None,
        imgsz  : tuple[int, int],
        path   : Path | str | None = None,
        root   : Path | str | None = None,
        fmt    : BBoxFormat        = BBoxFormat.CXCYWHN,
        persist: bool              = True,
    ):
        """Initialize a new instance.

        Args:
            data: Either a batch of bounding boxes, formatted as a
                numpy.ndarray of shape (N, 7+) and in CXCYWHN; or a path to a
                label file.
            imgsz: Image size as (H, W), used for conversions.
            path: Label file path. Defaults to None.
            root: Root directory of the label file. Defaults to None.
            fmt: Bounding box format in the label file or conversion code.
                Defaults to BBoxFormat.CXCYWHN.
            persist: If True, persist loaded data in memory. Defaults to True.
        """
        # Validate and set image size
        self._imgsz = I.imgsz(imgsz)
        
        # Validate data
        if data is None:
            pass
        elif isinstance(data, (Path, str)):
            if Path(data).normalize().is_txt_file(exist=True):
                path = data
                data = None
        elif isinstance(data, np.ndarray):
            pass
        else:
            raise TypeError(
                f"Expected 'data' to be a numpy.ndarray or a valid label file path, "
                f"but got {type(data).__name__}."
            )
        
        # Validate paths
        path = Path(path).normalize(exist=True) if path else None
        root = Path(root).normalize(exist=True) if root else None
        if data is None and path is not None:  # Load from path if data not provided
            if not path.is_txt_file(exist=True):
                raise FileNotFoundError(f"Label file not found at: {path}")
        
        # Ensure at least one of data or path is provided
        if all(v is None for v in [data, path]):
            raise ValueError(f"Expected 'data' or a valid label file path, but got both None.")
        
        # Call the setter to ensure type validation on init
        self._set_fmt(fmt)
        self.data = data
        
        # Continue the initialization chain
        super().__init__(data=self.data, path=path, root=root, persist=persist)
        
    # --- Container / Sequence Methods ---
    def __len__(self) -> int:
        """Return the length of the container.

        Returns:
            Length of the container.
        """
        return len(self.data)

    def __getitem__(self, index: int | slice | np.ndarray) -> BBox:
        """Return a bounding box at the given index.

        Args:
            index: Index or slice to access.

        Returns:
            Bounding box.
        """
        return BBox(data=self.data[index], imgsz=self._imgsz)
    
    # --- Properties ---
    @property
    def data(self) -> np.ndarray:
        """Return a batch of bounding boxes, formatted as a numpy.ndarray of
        shape (N, 7+) and in CXCYWHN format.
        """
        # Use super's behavior to load data from a file if needed.
        # This is implemented because we want to redefine the setter below.
        return super().data
    
    @data.setter
    def data(self, value: np.ndarray):
        """Set the bounding boxes.

        Args:
            value: Batch of bounding boxes, formatted as a numpy.ndarray of
                shape (N, 7+) and in CXCYWHN format.

        Raises:
            ValueError: If ``value`` is invalid.
        """
        if value is None:
            self._data = None
            return

        from .ops import is_xywh, is_xyxy, xywh_to_cxcywhn, xyxy_to_cxcywhn
        
        if not isinstance(value, np.ndarray):
            value = np.array(value, dtype=np.float32)
        
        if value.ndim != 2 or value.shape[1] < 7:
            raise ValueError(
                f"Expected 'data' to be a numpy.ndarray of shape (N, 7+), "
                f"but got {value.shape}."
            )
        
        # Internal conversion logic
        # Vectorized format detection and conversion
        if is_xywh(value, self._imgsz):
            value = xywh_to_cxcywhn(value, self._imgsz)
        elif is_xyxy(value):
            value = xyxy_to_cxcywhn(value, self._imgsz)
        
        self._data = value
    
    @property
    def shape(self) -> tuple[int, ...]:
        """Return the data shape."""
        return self.data.shape

    @property
    def meta(self) -> dict:
        """Return metadata describing the data."""
        return {
            "shape": self.shape,
            "dtype": self.data.dtype,
            "type" : type(self.data),
        }
    
    @property
    def imgsz(self) -> tuple[int, int]:
        """Return the image size."""
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
        """Return class identifiers for all bounding boxes."""
        return self.data[:, 6:7]

    @property
    def id(self) -> np.ndarray:
        """Return tracking identifiers for all bounding boxes."""
        return self.data[:, 7:8] if self.data.shape[1] > 7 else None
    
    @property
    def cxcywhn(self) -> np.ndarray:
        """Return all bounding boxes in CXCYWHN format."""
        return self.data
    
    def xyxy(self, imgsz: tuple[int, int] = None) -> np.ndarray:
        """Convert all bounding boxes from CXCYWHN to XYXY format.

        Args:
            imgsz: Optional image size as (H, W). If omitted, uses the stored
                ``_imgsz``. Defaults to None.

        Returns:
            Bounding boxes in XYXY format.
        """
        from .ops import cxcywhn_to_xyxy
        
        imgsz = I.imgsz(imgsz) if imgsz is not None else self._imgsz
        return cxcywhn_to_xyxy(self.data, imgsz)
    
    def xywh(self, imgsz: tuple[int, int] = None) -> np.ndarray:
        """Convert all bounding boxes from CXCYWHN to XYWH format.

        Args:
            imgsz: Optional image size as (H, W). If omitted, uses the stored
                ``_imgsz``. Defaults to None.

        Returns:
            Bounding boxes in XYWH format.
        """
        from .ops import cxcywhn_to_xywh
        
        imgsz = I.imgsz(imgsz) if imgsz is not None else self._imgsz
        return cxcywhn_to_xywh(self.data, imgsz)
    
    # --- Data Loading ---
    def load(self, reload: bool = False) -> np.ndarray:
        """Load all bounding boxes from a label file.

        Args:
            reload: If True, force reloading even if data is already in memory.
                Defaults to False.

        Returns:
            Loaded bounding boxes.
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

# endregion

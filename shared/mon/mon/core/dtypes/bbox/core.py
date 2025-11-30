#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module for bounding box data types.

This module provides classes for handling bounding boxes (BBoxes) in images,
including both single bounding boxes and multiple bounding boxes. It supports
various bounding box formats and provides methods for conversion and accessing
bounding box properties.

The default format of a bounding box is: <cx, cy, w, h, a, cls, ...>, where ...
can be any additional information such as confidence score or tracking ID.
For HBBs, the angle ``a`` is always ``0``.
"""

__all__ = [
    "BBox",
    "BBoxes",
]

import numpy as np

from mon.core.enum import BBoxFormat
from mon.core.pathlib import Path
from .utils import is_xywh, is_xyxy
from .. import image as I
from ..base import BaseTensorOrArray


class BBox(BaseTensorOrArray):
    """A base class for storing and manipulating a single bounding box in an
    image.
    
    This class extends BaseTensorOrArray to handle a single bounding box. It
    provides properties to access the bounding box in different formats, as well
    as class ID, confidence score, and tracking ID. It also supports conversion
    between different bounding box formats.
    
    Attributes:
        data (numpy.ndarray): A single bounding box as a numpy.ndarray of shape
            (7+) in CXCYWHN format.
        _imgsz (tuple[int, int]): Original image size as (H, W).
    
    Notes:
        The bounding boxes are expected to in the following format:
            <cx, cy, w, h, a, cls, conf, id, ...>
        where:
            - <cx, cy, w, h> are the bounding box coordinates in CXCYWHN format.
            - <a> is the angle.
            - <cls> is the class ID (optional).
            - <conf> is the confidence score (optional).
            - <id> is the tracking ID (optional).
    """
    
    def __init__(self, data: np.ndarray, imgsz: tuple[int, int]):
        """Initializes the BBox instance.
        
        Args:
            data (numpy.ndarray): A single bounding box as a numpy.ndarray of
                shape (7+), preferably in CXCYWHN format.
            imgsz (tuple[int, int]): Original image size as (H, W).
            
        Raises:
            ValueError: If ``imgsz`` is not provided.
        """
        # Validate and set imgsz
        if imgsz is None:
            raise ValueError(f"``imgsz`` must be specified.")
        self._imgsz = I.imgsz(imgsz)
        
        super().__init__(data=data)  # This will call the data setter
    
    # ---- Properties -----
    @property
    def data(self) -> np.ndarray:
        """Getter for the bounding box data.
        
        This property is overridden to ensure the returned data is a numpy.ndarray.
        Also, it is needed to override the setter to validate the shape.
        
        Returns:
            numpy.ndarray: The bounding box as a numpy.ndarray of shape (7+) in
                CXCYWHN format.
        """
        return self._data
    
    @data.setter
    def data(self, data: np.ndarray):
        """Setter for the bounding box data.
        
        This setter validates that the input data is a numpy.ndarray of shape
        (7+). If the input data is in XYWH or XYXY format, it will be converted
        to CXCYWHN format.
        
        Args:
            data (numpy.ndarray): A single bounding box as a numpy.ndarray of
                shape (7+) in any format.
        """
        if not isinstance(data, np.ndarray):
            raise TypeError(f"``data`` must be a numpy.ndarray, got {type(data)}.")
        if data.ndim != 1 or data.shape[0] < 7:
            raise ValueError(f"``data`` must be of shape (7+), got {data.shape}.")
        if is_xywh(data, self._imgsz):
            from .processing import xywh_to_cxcywhn
            data = xywh_to_cxcywhn(data[None, :], self._imgsz)[0]
        elif is_xyxy(data, self._imgsz):
            from .processing import xyxy_to_cxcywhn
            data = xyxy_to_cxcywhn(data[None, :], self._imgsz)[0]
        
        self._data = data
    
    @property
    def imgsz(self) -> tuple[int, int]:
        """Getter for the image size as (H, W).
        
        Returns:
            tuple[int, int]: The image size.
        """
        return self._imgsz
    
    @property
    def conf(self) -> float:
        """Getter for the confidence score of the bounding box.
        
        This property assumes that the confidence score is stored at index 5
        of the bounding box data.
        
        Returns:
            float: The confidence score.
        """
        return float(self.data[5])
    
    @property
    def cls(self) -> int:
        """Getter for the class ID of the bounding box.
        
        This property assumes that the class ID is stored at index 6 of the
        bounding box data.
        
        Returns:
            int: The class ID.
        """
        return int(self.data[6])
    
    @property
    def id(self) -> int:
        """Getter for the tracking ID of the bounding box.

        This property assumes that the tracking ID is stored at index 7 of the
        bounding box data.
        
        Returns:
            int: The tracking ID.
        """
        return int(self.data[7])

    @property
    def cxcywhn(self) -> np.ndarray:
        """An alias for the ``data`` property, returning the bounding box in
        CXCYWHN format.
        
        Returns:
            numpy.ndarray: The bounding box as a numpy.ndarray of shape (7+) in
                CXCYWHN format.
        """
        return self.data
    
    def xyxy(self, imgsz: tuple[int, int] = None) -> np.ndarray:
        """Converts the bounding box to XYXY format.
        
        Args:
            imgsz (tuple[int, int], optional): Image size as (H, W). If not
                provided, the original ``_imgsz`` will be used. Defaults to None.
        
        Returns:
            numpy.ndarray: The bounding box as a numpy.ndarray of shape (7) in
                XYXY format.
        """
        from .processing import cxcywhn_to_xyxy
        imgsz = I.imgsz(imgsz) if imgsz is not None else self.imgsz
        return cxcywhn_to_xyxy(self.data, imgsz)[0]
    
    def xywh(self, imgsz: tuple[int, int] = None) -> np.ndarray:
        """Converts the bounding box to XYWH format.
        
        Args:
            imgsz (tuple[int, int], optional): Image size as (H, W). If not
                provided, the original ``_imgsz`` will be used. Defaults to None.
        
        Returns:
            numpy.ndarray: The bounding box as a numpy.ndarray of shape (7) in
                XYWH format.
        """
        from .processing import cxcywhn_to_xywh
        imgsz = I.imgsz(imgsz) if imgsz is not None else self.imgsz
        return cxcywhn_to_xywh(self.data, imgsz)[0]
    
    @property
    def area(self) -> float:
        """Calculates the area of the bounding box.
        
        Returns:
            float: The area of the bounding box.
        """
        h0, w0  = self.imgsz
        w, h    = self.data[2:4]
        w      *= w0
        h      *= h0
        return float(w * h)


class BBoxes(BaseTensorOrArray):
    """A base class for storing and manipulating multiple bounding boxes in an
    image.
    
    This class extends BaseTensorOrArray to handle multiple bounding boxes. It
    provides properties to access bounding boxes in different formats, as well
    as class IDs, confidence scores, and tracking IDs. It also supports loading
    bounding boxes from a label file.
    
    Attributes:
        data (numpy.ndarray): A single bounding box as a numpy.ndarray of shape
            (7+) in CXCYWHN format.
        _imgsz (tuple[int, int]): Original image size as (H, W).
        _path (Path): Path to the label file.
        _root (Path): Root directory of the label file.
        _fmt (BBoxFormat): Bounding box format of the label file.
        _cvt_fmt (BBoxFormat): Conversion format from the label file format to
            CXCYWHN format.
    
    Notes:
        The bounding boxes are expected to in the following format:
            <cx, cy, w, h, a, cls, conf, id, ...>
        where:
            - <cx, cy, w, h> are the bounding box coordinates in CXCYWHN format.
            - <a> is the angle.
            - <cls> is the class ID (optional).
            - <conf> is the confidence score (optional).
            - <id> is the tracking ID (optional).
    """
    
    def __init__(
        self,
        data : np.ndarray | Path,
        imgsz: tuple[int, int],
        path : Path       = None,
        root : Path       = None,
        fmt  : BBoxFormat = BBoxFormat.CXCYWHN,
    ):
        # Validate and set imgsz
        if imgsz is None:
            raise ValueError(f"``imgsz`` must be specified.")
        self._imgsz = I.imgsz(imgsz)
        
        # Set fmt and cvt_fmt in case of loading bounding boxes from file
        self._set_fmt(fmt)
        
        # Validate data
        if isinstance(data, Path | str) and Path(data).is_txt_file(exist=True):
            path = data
            data = None
        elif not isinstance(data, np.ndarray):
            raise TypeError(f"``data`` must be a ``numpy.ndarray`` or a valid label file path, got {type(data)}.")
        
        # Validate and set path
        self._path = Path(path) if path is not None else None
        self._root = Path(root) if root is not None else None
        if data is None and self._path is not None:  # Load from path if data not provided
            if self._path.is_txt_file(exist=True):
                from .io import load
                data = load(path=self._path, fmt=self._cvt_fmt, imgsz=self._imgsz)
            else:
                raise FileNotFoundError(f"Label file not found: {self._path}.")
        
        super().__init__(data=data)  # This will call the data setter
        
    # ----- Properties -----
    @property
    def data(self) -> np.ndarray:
        """Getter for the bounding box data.
        
        This property is overridden to ensure the returned data is a numpy.ndarray.
        Also, it is needed to override the setter to validate the shape.
        
        Returns:
            numpy.ndarray: The bounding box as a numpy.ndarray of shape (N, 7+)
                in CXCYWHN format.
        """
        return self._data
    
    @data.setter
    def data(self, data: np.ndarray):
        """Setter for the bounding box data.
        
        This setter validates that the input data is a numpy.ndarray of shape
        (N, 7+). If the input data is in XYWH or XYXY format, it will be converted
        to CXCYWHN format.
        
        Args:
            data (numpy.ndarray): A single bounding box as a numpy.ndarray of
                shape (N, 7+) in any format.
        """
        if not isinstance(data, np.ndarray):
            raise TypeError(f"``data`` must be a numpy.ndarray, got {type(data)}.")
        if data.ndim != 2 or data.shape[0] < 7:
            raise ValueError(f"``data`` must be of shape (N, 7+), got {data.shape}.")
        if is_xywh(data, self._imgsz):
            from .processing import xywh_to_cxcywhn
            data = xywh_to_cxcywhn(data[None, :], self._imgsz)
        elif is_xyxy(data, self._imgsz):
            from .processing import xyxy_to_cxcywhn
            data = xyxy_to_cxcywhn(data[None, :], self._imgsz)
        
        self._data = data
        
    @property
    def imgsz(self) -> tuple[int, int]:
        """Getter for the image size as (H, W).
        
        Returns:
            tuple[int, int]: The image size.
        """
        return self._imgsz
    
    @property
    def path(self) -> Path:
        """Getter for the label file path.
        
        Returns:
            Path: The label file path.
        """
        return self._path

    @property
    def root(self) -> Path:
        """Getter for the root directory of the label file.
        
        Returns:
            Path: The root directory path.
        """
        return self._root

    @property
    def fmt(self) -> BBoxFormat:
        """Getter for the bounding box format of the label file.
        
        This property indicates the format in which the bounding boxes are
        stored in the label file. It is used when loading and converting the
        bounding boxes from the file to CXCYWHN format.
        
        Returns:
            BBoxFormat: The bounding box format of the label file.
        """
        return self._fmt
    
    def _set_fmt(self, fmt: BBoxFormat):
        """Sets the bounding box format and conversion format.
        
        Args:
            fmt (BBoxFormat): The bounding box format of the label file.
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
        """Getter for the confidence score vector representing confidence scores
        for each bounding box.
        
        This property assumes that the confidence scores are stored at index 5
        of the bounding box data.
        
        Returns:
            numpy.ndarray: The confidence scores as a numpy.ndarray of shape
                (N, 1).
        """
        return self.data[:, 5:6]

    @property
    def cls(self) -> np.ndarray:
        """Getter for the class ID vector representing class IDs for each
        bounding box.
        
        This property assumes that the class IDs are stored at index 6 of the
        bounding box data.
        
        Returns:
            numpy.ndarray: The class IDs as a numpy.ndarray of shape (N, 1).
        """
        return self.data[:, 6:7]

    @property
    def id(self) -> np.ndarray:
        """Getter for the tracking ID vector representing tracking IDs for each
        bounding box.
        
        This property assumes that the tracking IDs are stored at index 7 of the
        bounding box data.

        Returns:
            numpy.ndarray: The tracking IDs as a numpy.ndarray of shape (N, 1).
        """
        return self.data[:, 6:7]
    
    @property
    def cxcywhn(self) -> np.ndarray:
        """An alias for the ``data`` property, returning the bounding box in
        CXCYWHN format.
        
        Returns:
            numpy.ndarray: The bounding boxes as a numpy.ndarray of shape (N, 7+)
                in CXCYWHN format.
        """
        return self.data
    
    def xyxy(self, imgsz: tuple[int, int] = None) -> np.ndarray:
        """Converts the bounding boxes to XYXY format.
        
        Args:
            imgsz (tuple[int, int], optional): Image size as (H, W). If not
                provided, the original ``_imgsz`` will be used. Defaults to None.
        
        Returns:
            numpy.ndarray: The bounding boxes as a numpy.ndarray of shape (N, 7)
                in XYXY format.
        """
        from .processing import cxcywhn_to_xyxy
        imgsz = I.imgsz(imgsz) if imgsz is not None else self.imgsz
        return cxcywhn_to_xyxy(self.data, imgsz)
    
    def xywh(self, imgsz: tuple[int, int] = None) -> np.ndarray:
        """Converts the bounding boxes to XYWH format.
        
        Args:
            imgsz (tuple[int, int], optional): Image size as (H, W). If not
                provided, the original ``_imgsz`` will be used. Defaults to None.
                
        Returns:
            numpy.ndarray: The bounding boxes as a numpy.ndarray of shape (N, 7)
                in XYWH format.
        """
        from .processing import cxcywhn_to_xywh
        imgsz = I.imgsz(imgsz) if imgsz is not None else self.imgsz
        return cxcywhn_to_xywh(self.data, imgsz)
    
    # ----- Initialize -----
    def load(self, reload: bool = False) -> np.ndarray:
        """Loads the bounding boxes from the label file.

        This method can be called internally or externally to reload the data if
        needed.
        
        Args:
            reload (bool): If True, forces reloading the bounding boxes from the
                label file. Defaults to False.
        
        Returns:
            numpy.ndarray: The loaded bounding boxes as a numpy.ndarray of shape
                (N, 7+) in CXCYWHN format.
        """
        # Return the bbox if it is already loaded and not reloading
        if not reload and self._data is not None:
            return self._data

        # Load all bounding boxes from the label file
        from .io import load
        bbox = load(path=self._path, fmt=self._cvt_fmt, imgsz=self._imgsz)
        
        # Cache
        self._data = bbox
        return self._data

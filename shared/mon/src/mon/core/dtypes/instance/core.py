#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module for instance annotation data type.

This module provides a class for handling instance annotations, which combine
multiple types of annotations for a single object, such as bounding boxes, masks,
polygons, keypoints, and more. It supports lazy loading and access to various
properties of the instance.
"""

__all__ = [
    "Instance",
]

import numpy as np

from mon.core.pathlib import Path
from .. import bbox as B, image as I
from ..base import BaseTensorOrArray


class Instance(BaseTensorOrArray):
    """A base class for a single instance annotation.
    
    This class extends BaseTensorOrArray to handle a single instance annotation,
    which can include various types of annotations such as bounding boxes, masks,
    polygons, keypoints, and more. It supports lazy loading and provides
    properties to access instance metadata.
    
    One instance can have these kinds of annotations (i.e., attributes):
        - bbox     : bounding box, support both OBB and HBB (*primary).
        - mask     : instance mask (i.e., pixels that belong to the object).
        - polygon  : points to outline the object's shape.
        - keypoints: points on key parts, like eyes or joints.
        - cuboid   : 3D bounding box with depth.
        - cls      : the type of object, like "car".
        
    Attributes:
        data (numpy.ndarray): The bounding box as a numpy.ndarray of shape
            (7+) in CXCYWHN format.
        mask (numpy.ndarray): Instance mask as a numpy.ndarray of shape
            (H, W, C) with pixel values in [0, 255].
        _imgsz (tuple[int, int]): Original image size as (H, W).
        _image_path (Path): Associated image file path.
        _root (Path): Root directory for the label file.
        
    Notes:
        The bounding boxes are expected to in the following format:
            <cx, cy, w, h, a, cls, conf, id, ...>
        where:
            - <cx, cy, w, h> are the bounding box coordinates in CXCYWHN format.
            - <a> is the angle.
            - <cls> is the class ID (optional).
            - <conf> is the confidence score (optional).
            - <id> is the tracking ID (optional).
        
    **I am in the process of adding more annotations to this class, so it may
    subject to changes in the future.**
    """
    
    def __init__(
        self,
        data      : np.ndarray,
        imgsz     : tuple[int, int],
        mask      : np.ndarray = None,
        image_path: Path       = None,
        root      : Path       = None,
    ):
        """Initializes the Instance object.
        
        Args:
            data (numpy.ndarray): A single bounding box as a numpy.ndarray of
                shape (7+), preferably in CXCYWHN format.
            imgsz (tuple[int, int]): Original image size in (H, W) format.
            mask (numpy.ndarray, optional): Instance segmentation mask as a
                numpy.ndarray of shape (H, W, C) with pixel values in [0, 255].
                Defaults to None.
            image_path (Path, optional): Associated image file path. Defaults to
                None.
            root (Path, optional): Root directory for the label file. Defaults to
                None.
        """
        # Validate and set imgsz
        if imgsz is None and Path(image_path).is_image_file(exist=True):
            imgsz = I.read_size(image_path)
        if imgsz is None:
            raise ValueError("Either ``imgsz`` or a valid ``image_path`` must be provided to determine the original image size.")
        self._imgsz = I.imgsz(imgsz)
        
        super().__init__(data=data)  # This will call the data setter
        self._mask       = mask
        self._image_path = Path(image_path) if image_path is not None else None
        self._root       = Path(root)       if root       is not None else None
    
    # ----- Properties -----
    @property
    def data(self) -> np.ndarray:
        """Getter for the bounding box of the instance.
        
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
        if B.is_xywh(data, self._imgsz):
            data = B.xywh_to_cxcywhn(data[None, :], self._imgsz)[0]
        elif B.is_xyxy(data, self._imgsz):
            data = B.xyxy_to_cxcywhn(data[None, :], self._imgsz)[0]
        
        self._data = data
    
    @property
    def mask(self) -> np.ndarray:
        """Getter for the instance segmentation mask.
        
        Returns:
            numpy.ndarray: Instance mask as a numpy.ndarray of shape (H, W, C)
                with pixel values in [0, 255].
        """
        return self._mask
    
    @mask.setter
    def mask(self, mask: np.ndarray):
        """Getter for the instance segmentation mask.
        
        Sometimes you may want to set or update the mask after initialization.
        
        Args:
            mask (numpy.ndarray): Instance mask as a numpy.ndarray of shape
                (H, W, C) with pixel values in [0, 255].
        """
        self._mask = mask
    
    @property
    def imgsz(self) -> tuple[int, int]:
        """Getter for the image size as (H, W).
        
        Returns:
            tuple[int, int]: The image size.
        """
        return self._imgsz
    
    @property
    def image_path(self) -> Path:
        """Getter for the associated image file path.
        
        Returns:
            Path: The associated image file path.
        """
        return self._image_path
    
    @property
    def root(self) -> Path:
        """Getter for the root directory of the label file.
        
        Returns:
            Path: The root directory of the label file.
        """
        return self._root
    
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
        imgsz = I.imgsz(imgsz) if imgsz is not None else self.imgsz
        return B.cxcywhn_to_xyxy(self.data, imgsz)[0]
    
    def xywh(self, imgsz: tuple[int, int] = None) -> np.ndarray:
        """Converts the bounding box to XYWH format.
        
        Args:
            imgsz (tuple[int, int], optional): Image size as (H, W). If not
                provided, the original ``_imgsz`` will be used. Defaults to None.
        
        Returns:
            numpy.ndarray: The bounding box as a numpy.ndarray of shape (7) in
                XYWH format.
        """
        imgsz = I.imgsz(imgsz) if imgsz is not None else self.imgsz
        return B.cxcywhn_to_xywh(self.data, imgsz)[0]

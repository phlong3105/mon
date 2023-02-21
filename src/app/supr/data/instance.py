#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the instance dataclass. It is mainly used to transfer
data between detectors and trackers.
"""

from __future__ import annotations

__all__ = [
    "Instance",
]

import uuid
from timeit import default_timer as timer
from typing import Any

import cv2
import numpy as np

import mon


# region Instance

class Instance:
    """An instances of a moving object in a given frame. This class is mainly
    used to pass data between detectors and trackers.
    
    Notes:
        In the future, if we can extend this class to support keypoints
        detection.
    
    Attributes:
        id_: An unique ID. Defaults to None.
        roi_id: The unique ID of the ROI containing the
        bbox: A bounding bbox in XYXY format.
        polygon: A list of points representing an instance mask. Defaults to
            None.
        confidence: A confidence score. Defaults to None.
        classlabel: A :class:`mon.Classlabel` object. Defaults to None.
        frame_index: The current frame index. Defaults to None.
        timestamp: The creating time of the current instance.
    """
    
    def __init__(
        self,
        bbox       : np.ndarray,
        id_        : int | str         = uuid.uuid4().int,
        roi_uid    : int | str  | None = None,
        polygon    : np.ndarray | None = None,
        confidence : float      | None = None,
        classlabel : dict       | None = None,
        frame_index: int        | None = None,
        timestamp  : float             = timer(),
    ):
        self.id_         = id_
        self.roi_id      = roi_uid
        self.bbox        = np.array(bbox)
        self.polygon     = np.array(polygon)
        self.confidence  = confidence
        self.classlabel  = classlabel
        self.frame_index = frame_index
        self.timestamp   = timestamp
    
    @classmethod
    def from_value(cls, value: Any) -> Instance:
        if isinstance(value, Instance):
            return value
        elif isinstance(value, dict):
            return cls(**value)
        else:
            raise TypeError(
                f"value must be an Instance object or a dict, but got "
                f"{type(value)}."
            )
    
    @property
    def box_center(self):
        return mon.get_bbox_center(bbox=self.bbox)
    
    @property
    def box_tl(self):
        """The bbox's top left corner."""
        return self.bbox[0:2]
    
    @property
    def box_corners_points(self) -> np.ndarray:
        return mon.get_bbox_corners_points(bbox=self.bbox)
    
    def draw(
        self,
        image  : np.ndarray,
        bbox   : bool             = False,
        polygon: bool             = False,
        label  : bool             = True,
        color  : list[int] | None = None
    ) -> np.ndarray:
        """Draw the current object on the :param:`image`."""
        color = color or (self.classlabel["color"]
                          if self.classlabel is not None else (255, 255, 255))
        
        if bbox:
            cv2.rectangle(
                img       = image,
                pt1       = (self.bbox[0], self.bbox[1]),
                pt2       = (self.bbox[2], self.bbox[3]),
                color     = color,
                thickness = 2
            )
        if polygon:
            pts = self.polygon.reshape((-1, 1, 2))
            cv2.polylines(
                img       = image,
                pts       = pts,
                isClosed  = True,
                color     = color,
                thickness = 2,
            )
        if label:
            font = cv2.FONT_HERSHEY_SIMPLEX
            org = (self.box_tl[0] + 5, self.box_tl[1])
            cv2.putText(
                img       = image,
                text      = self.classlabel["name"],
                fontFace  = font,
                fontScale = 1.0,
                org       = org,
                color     = color,
                thickness = 2
            )
        return image

# endregion

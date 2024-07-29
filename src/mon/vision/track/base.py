#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the base classes for tracks and trackers."""

from __future__ import annotations

__all__ = [
    "Detection",
    "Track",
    "Tracker",
]

from timeit import default_timer as timer
from abc import ABC, abstractmethod
from typing import Any, Callable

import numpy as np

from mon import core, data
from mon.globals import MOTIONS, OBJECTS, TrackState
from mon.vision import geometry

console = core.console


# region Track

class Detection:
    """An instance of a track in a frame. This class is mainly used to wrap and
    pass data between detectors and trackers.
    """
    
    def __int__(
        self,
        id_       : int               = -1,
        frame_id  : int               = -1,
        roi_id    : int               = -1,
        bbox      : np.ndarray | None = None,
        polygon   : np.ndarray | None = None,
        feature   : np.ndarray | None = None,
        confidence: float             = -1.0,
        classlabel: dict       | None = None,
        timestamp : int | float       = timer(),
        *args, **kwargs
    ):
        self.id_         = id_
        self.frame_id    = frame_id
        self.roi_id      = roi_id
        self.bbox        = np.array(bbox)    if bbox    is not None else None
        self.polygon     = np.array(polygon) if polygon is not None else None
        self.feature     = np.array(feature) if feature is not None else None
        self.confidence  = confidence
        self.classlabel  = classlabel
        self.timestamp   = timestamp
    
    @classmethod
    def from_value(cls, value: Detection | dict) -> Detection:
        """Create a :class:`BBoxAnnotation` object from an arbitrary :param:`value`.
        """
        if isinstance(value, dict):
            return Detection(**value)
        elif isinstance(value, Detection):
            return value
        else:
            raise ValueError(
                f":param:`value` must be a :class:`Detection` class or "
                f"a :class:`dict`, but got {type(value)}."
            )
    
    @property
    def bbox_center(self):
        return geometry.get_bbox_center(bbox=self.bbox)
    
    @property
    def bbox_tl(self):
        """The bbox's top left corner."""
        return self.bbox[0:2]
    
    @property
    def bbox_corners_points(self) -> np.ndarray:
        return geometry.get_bbox_corners_points(bbox=self.bbox)


class Track(ABC):
    """The base class for all tracks.
    
    Definition: A track represents the trajectory or path that an object takes
    as it moves through a sequence of frames in a video or across multiple
    sensor readings. It consists of a series of positional data points
    corresponding to the object's location at different times.
    
    Components of a Track:
        Object Identification: Identifying the object of interest to be tracked.
        Positional Data: The coordinates (e.g., x and y positions in 2D space)
            of the object in each frame.
        Time Stamps: The specific times at which the object's positions are recorded.
        Track ID: A unique identifier for each tracked object to distinguish
            it from others in the scene.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._history = []

# endregion


# region Tracker

class Tracker(ABC):
    """The base class for all trackers."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    @abstractmethod
    def update(self, *args, **kwargs):
        pass

# endregion

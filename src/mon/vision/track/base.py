#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the base classes for tracks and trackers."""

from __future__ import annotations

__all__ = [
    "Detection",
    "Track",
    "Tracker",
]

from abc import ABC, abstractmethod
from timeit import default_timer as timer

import numpy as np

from mon import core
from mon.globals import TrackState
from mon.vision import geometry

console = core.console


# region Track

class Detection:
    """An instance of a track in a frame. This class is mainly used to wrap and
    pass data between detectors and trackers.
    
    Args:
        frame_id: The frame ID or index.
        bbox: The bounding box.
        polygon: The polygon resulted from instance segmentation models.
        feature: The feature used in deep tracking methods.
        confidence: The confidence score.
        classlabel: The class label.
        timestamp: The timestamp when the detection is created.
    """
    
    def __init__(
        self,
        frame_id  : int        | None = None,
        bbox      : np.ndarray | None = None,
        polygon   : np.ndarray | None = None,
        feature   : np.ndarray | None = None,
        confidence: float      | None = None,
        classlabel: dict | int | None = None,
        timestamp : int  | float      = timer(),
        *args, **kwargs
    ):
        self.frame_id   = frame_id
        self.bbox       = np.array(bbox)    if bbox    is not None else None
        self.polygon    = np.array(polygon) if polygon is not None else None
        self.feature    = np.array(feature) if feature is not None else None
        self.confidence = confidence
        self.classlabel = classlabel
        self.timestamp  = timestamp
    
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
        return geometry.bbox_center(bbox=self.bbox)
    
    @property
    def bbox_tl(self):
        """The bbox's top left corner."""
        return self.bbox[0:2]
    
    @property
    def bbox_corners_points(self) -> np.ndarray:
        return geometry.bbox_corners_points(bbox=self.bbox)


class Track(ABC):
    """The base class for all tracks.
    
    Definition: A track represents the trajectory or path that an object takes
    as it moves through a sequence of frames in a video or across multiple
    sensor readings. It consists of a series of positional data points
    corresponding to the object's location at different times.
    
    Args:
        id_: The unique ID of the track. Default: ``None``.
        state: The state of the track. Default: :class:`TrackState.NEW`.
        detections: The list of detections associated with the track.
            Default: ``[]``.
    """
    
    _count: int = 0
    
    def __init__(
        self,
        id_       : int | None = None,
        state     : TrackState = TrackState.NEW,
        detections: Detection | list[Detection] = [],
    ):
        self.id_   = id_ or Track._count
        Track._count += 1
        self.state = state
        detections = [detections] if not isinstance(detections, list) else detections
        assert all(isinstance(d, Detection) for d in detections)
        self.history: list[Detection] = detections
    
    @staticmethod
    def next_id() -> int:
        """This function keeps track of the total number of tracking objects,
        which is also the track ID of the new tracking object.
        """
        return Track._count + 1
    
    @abstractmethod
    def update(self, *args, **kwargs):
        """Updates the state vector of the tracking object."""
        pass
    
    @abstractmethod
    def predict(self):
        """Predict the next state of the tracking object."""
        pass
        
# endregion


# region Tracker

class Tracker(ABC):
    """The base class for all trackers."""
    
    def __init__(self):
        super().__init__()
        self.frame_count = 0
        
    @abstractmethod
    def update(self, *args, **kwargs):
        pass

# endregion

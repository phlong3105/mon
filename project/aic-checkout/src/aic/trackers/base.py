#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Base class for all tracker.
"""

from __future__ import annotations

import abc
from typing import Optional
from typing import Union

import numpy as np

from aic.builder import MOTIONS
from aic.builder import OBJECTS
from aic.trackers.motion import KFBoxMotion
from aic.trackers.motion import Motion
from onevision import Callable
from onevision import ListOrTuple3T

__all__ = [
    "BaseTracker",
]


# MARK: - BaseTracker

class BaseTracker(metaclass=abc.ABCMeta):
    """Base Tracker.

    Attributes:
        name (str):
            Name of the tracker.
        max_age (int):
            Time to store the track before deleting, that mean track could
            live in `max_age` frame with no match bounding box, consecutive
            frame that track disappears. Default: `1`.
        min_hits (int):
            Number of frame which has matching bounding box of the detected
            object before the object is considered becoming the track.
            Default: `3`.
        iou_threshold (float):
            Intersection over Union threshold between two track with their
            bounding box. Default: `0.3`.
        motion_model (FuncCls):
            Motion model class. Default: `KFBoxMotion`.
        object_type (Callable, optional):
            Type of tracked object. Default: `None`.
        frame_count (int):
            Current index of reading frame.
        tracks (list):
            List of tracked objects.
    """

    # MARK: Magic Functions

    def __init__(
        self,
        name         : str,
        max_age      : int   = 1,
        min_hits     : int   = 3,
        iou_threshold: float = 0.3,
        motion_model : Union[dict, Motion, Callable] = "kf_box_motion",
        object_type  : Optional[Callable]            = None,
        *args, **kwargs
    ):
        super().__init__()
        from aic.objects.base import BaseMovingObject

        self.name          = name
        self.max_age       = max_age
        self.min_hits      = min_hits
        self.iou_threshold = iou_threshold
        self.frame_count   = 0
        self.motion_model  = KFBoxMotion
        self.object_type   = BaseMovingObject
        self.tracks        = []
        
        self.init_motion_model(motion_model=motion_model)
        self.init_object_type(object_type=object_type)

    # MARK: Configure
    
    def init_motion_model(self, motion_model: Union[dict, Motion, Callable]):
        """Initialize the motion model for tracked objects."""
        if isinstance(motion_model, str):
            self.motion_model  = MOTIONS.get(motion_model)
        elif isinstance(motion_model, dict):
            self.motion_model  = MOTIONS.build_from_dict(cfg=motion_model).__class__
        elif isinstance(motion_model, Motion):
            self.motion_model  = motion_model.__class__
        else:
            raise ValueError()

    def init_object_type(self, object_type: Optional[Callable] = None):
        """Initialize the type for tracked objects."""
        from aic.objects.base import BaseMovingObject

        if object_type is None:
            self.object_type = BaseMovingObject
        elif isinstance(object_type, str):
            self.object_type = OBJECTS.get(object_type)
        elif isinstance(object_type, BaseMovingObject):
            self.object_type = object_type.__class__
        else:
            raise ValueError()

    # MARK: Update

    @abc.abstractmethod
    def update(self, detections: list):
        """Update `self.tracks` with new detections.

        Args:
            detections (list):
                List of newly detected detections.

        Requires:
            This method must be called once for each frame even with empty
            detections, just call update with empty container.
        """
        pass

    @abc.abstractmethod
    def associate_instances_to_tracks(self, instances: np.ndarray, tracks: np.ndarray) -> ListOrTuple3T[np.ndarray]:
        """Assigns `detections` to `self.tracks`.

        Args:
            instances (np.ndarray):
                Newly detected detections.
            tracks (np.ndarray):
                Current tracks.

        Returns:
            matched_indexes (np.ndarray):
            unmatched_inst_indexes (np.ndarray):
            unmatched_trks_indexes (np.ndarray):
        """
        pass

    @abc.abstractmethod
    def update_matched_tracks(self, matched_indexes: np.ndarray, instances: list):
        """Update tracks that have been matched with new detected detections.

        Args:
            matched_indexes (np.ndarray):
                Indexes of `self.tracks` that have not been matched with new
                detections.
            instances (list):
                Newly detected detections.
        """
        pass

    @abc.abstractmethod
    def create_new_tracks(self, unmatched_inst_indexes: np.ndarray, instances: list):
        """Create new tracks.

        Args:
            unmatched_inst_indexes (np.ndarray):
                Indexes of `detections` that have not been matched with any
                tracks.
            instances (list):
                Newly detected detections.
        """
        pass

    @abc.abstractmethod
    def delete_dead_tracks(self):
        """Delete dead tracks."""
        pass

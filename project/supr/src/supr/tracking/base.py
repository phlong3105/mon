#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the base class for all trackers."""

from __future__ import annotations

__all__ = [
    "Tracker",
]

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from supr import constant, data
from supr.tracking import motion

if TYPE_CHECKING:
    from supr.typing import CallableType, MotionType


# region Tracker

class Tracker(ABC):
    """The base class for all trackers.

    Args:
        max_age: The time to store the track before deleting, that mean a track
            could live upto :param:`max_age` frames with no match bounding box,
            consecutive frame that track disappears. Defaults to 1.
        min_hits: A number of frames, which has matching bounding box of the
            detected object before the object is considered becoming the track.
            Defaults to 3.
        iou_threshold: An Intersection-over-Union threshold between two tracks.
            Defaults to 0.3.
        motion_model: A motion model. Defaults to 'KFBoxMotion'.
        object_type: An object type. Defaults to None.
    """

    def __init__(
        self,
        max_age      : int                 = 1,
        min_hits     : int                 = 3,
        iou_threshold: float               = 0.3,
        motion_model : MotionType          = "kf_box_motion",
        object_type  : CallableType | None = data.MovingObject,
    ):
        super().__init__()
        self.max_age       = max_age
        self.min_hits      = min_hits
        self.iou_threshold = iou_threshold
        self.frame_count   = 0
        self.motion_model  = constant.MOTION.build(cfg=motion_model).__class__
        self.object_type   = object_type
        self.tracks        = []
        
        self.init_motion_model(motion_model=motion_model)
        self.init_object_type(object_type=object_type)
    
    def init_motion_model(self, motion_model: MotionType):
        """Initialize the motion model for tracked objects."""
        motion_model = motion_model or self.motion_model
        if isinstance(motion_model, str):
            motion_model = constant.MOTION.get(motion_model)
        elif isinstance(motion_model, dict):
            motion_model = constant.MOTION.build(cfg=motion_model).__class__
        elif isinstance(motion_model, motion.Motion):
            motion_model = motion_model.__class__
        self.motion_model = motion_model

    def init_object_type(self, object_type: CallableType | None = None):
        """Initialize the type for tracked objects."""
        object_type = object_type or self.object_type
        if object_type is None:
            object_type = data.MovingObject,
        elif isinstance(object_type, str):
            object_type = constant.OBJECT.get(object_type)
        elif isinstance(object_type, data.MovingObject):
            object_type = object_type.__class__
        self.object_type = object_type

    @abstractmethod
    def update(self, instances: list | np.ndarray = ()):
        """Update :attr:`tracks` with new detections. This method will call the
        following methods:
            - :meth:`assign_instances_to_tracks`
            - :meth:`update_matched_tracks`
            - :meth:`create_new_tracks`
            - :meth`:delete_dead_tracks`

        Args:
            instances: A list of new instances. Defaults to ().

        Requires:
            This method must be called once for each frame even with empty
            instances, just call update with an empty list.
        """
        pass

    @abstractmethod
    def assign_instances_to_tracks(
        self,
        instances: list | np.ndarray,
        tracks   : list | np.ndarray,
    ) -> tuple[
        list | np.ndarray,
        list | np.ndarray,
        list | np.ndarray
    ]:
        """Assigns new :param:`instances` to :param:`tracks`.

        Args:
            instances: A list of new instances
            tracks: A list of existing tracks.

        Returns:
            A list of tracks' indexes that have been matched with new instances.
            A list of new instances' indexes of that have NOT been matched with
                any tracks.
            A list of tracks' indexes that have NOT been matched with new
                instances.
        """
        pass

    @abstractmethod
    def update_matched_tracks(
        self,
        matched_indexes: list | np.ndarray,
        instances      : list | np.ndarray
    ):
        """Update existing tracks that have been matched with new instances.

        Args:
            matched_indexes: A list of tracks' indexes that have been matched
                with new instances.
            instances: A list of new instances.
        """
        pass

    @abstractmethod
    def create_new_tracks(
        self,
        unmatched_inst_indexes: list | np.ndarray,
        instances             : list | np.ndarray
    ):
        """Create new tracks for new instances that haven't been matched to any
        existing tracks.

        Args:
            unmatched_inst_indexes: A list of new instances' indexes of that
                haven't been matched with any tracks.
            instances: A list of new instances.
        """
        pass

    @abstractmethod
    def delete_dead_tracks(self):
        """Delete all dead tracks."""
        pass

# endregion

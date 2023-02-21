#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the base class for all trackers."""

from __future__ import annotations

__all__ = [
    "Tracker",
]

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from supr import data, motion, motion as mo
from supr.globals import MOTIONS, OBJECTS


# region Tracker

class Tracker(ABC):
    """The base class for all trackers.

    Args:
        max_age: The time to store the track before deleting, that mean a track
            could live upto :param:`max_age` frames with no match bounding bbox,
            consecutive frame that track disappears. Defaults to 1.
        min_hits: A number of frames, which has matching bounding bbox of the
            detected object before the object is considered becoming the track.
            Defaults to 3.
        iou_threshold: An Intersection-over-Union threshold between two tracks.
            Defaults to 0.3.
        motion_type: A motion model. Defaults to 'KFBoxMotion'.
        object_type: An object type. Defaults to None.
    """

    def __init__(
        self,
        max_age      : int   = 1,
        min_hits     : int   = 3,
        iou_threshold: float = 0.3,
        motion_type  : Any   = "kf_box_motion",
        object_type  : Any   = data.MovingObject,
    ):
        super().__init__()
        self.max_age       = max_age
        self.min_hits      = min_hits
        self.iou_threshold = iou_threshold
        self.motion_type   = motion_type
        self.object_type   = object_type
        self.frame_count   = 0
        self.tracks        = []
    
    @property
    def motion_type(self) -> type(motion.Motion):
        return self._motion_type
    
    @motion_type.setter
    def motion_type(self, motion_type: Any):
        if isinstance(motion_type, str):
            motion_type = MOTIONS.get(motion_type)
        elif isinstance(motion_type, dict):
            if not hasattr(motion_type, "name"):
                raise ValueError(f"motion_type must contain a key 'name'.")
            motion_type = MOTIONS.get(motion_type["name"]).__class__
        elif isinstance(motion_type, mo.Motion):
            motion_type = motion_type.__class__
        self._motion_type = motion_type
    
    @property
    def object_type(self) -> type(data.MovingObject):
        return self._object_type
    
    @object_type.setter
    def object_type(self, object_type: Any):
        if isinstance(object_type, str):
            object_type = OBJECTS.get(object_type)
        elif isinstance(object_type, dict ):
            if not hasattr(object_type, "name"):
                raise ValueError(f"object_type must contain a key 'name'.")
            object_type = OBJECTS.get(object_type["name"]).__class__
        elif isinstance(object_type, data.MovingObject):
            object_type = object_type.__class__
        self._object_type = object_type
    
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
        for i in unmatched_inst_indexes:
            new_trk = self.object_type(
                instances = instances[i],
                motion    = self.motion_type,
            )
            self.tracks.append(new_trk)

    @abstractmethod
    def delete_dead_tracks(self):
        """Delete all dead tracks."""
        pass

# endregion

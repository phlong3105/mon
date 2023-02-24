#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the base class for all moving objects."""

from __future__ import annotations

__all__ = [
    "MovingObject", "Object", "StaticObject",
]

import uuid
from abc import ABC, abstractmethod
from collections import Counter
from timeit import default_timer as timer
from typing import Any, Callable, Union

import cv2
import numpy as np

import mon
from supr import motion as mo, rmoi
from supr.data import instance
from supr.globals import MOTIONS, MovingState


# region Object

class Object(ABC):
    """The base class for all objects."""
    pass

# endregion


# region Static Object

class StaticObject(Object):
    """The base class for all static objects (i.e., trees, poles, traffic
    lights, etc.).
    """
    pass

# endregion


# region Moving Object

class MovingObject(list[instance.Instance], Object):
    """The base class for all moving objects. It is a list that contains all
    instances of the object in consecutive frames. The list is sorted in a
    timely manner.
    
    See more: :class:`Object`.
    
    Args:
        instances: The first instance of this object, or a list of instances.
        id_: The object unique ID.
        motion: A motion model.
        moving_state: The current state of the moving object with respect to the
            ROIs. Defaults to 'Candidate'.
        moi_id: The unique ID of the MOI. Defaults to None.
        timestamp: The time when the object is created.
        frame_index: The frame index when the object is created.
    """
    
    min_entering_distance: float = 0.0    # Minimum distance when an object enters the ROI to be `Confirmed`. Defaults to 0.0.
    min_traveled_distance: float = 100.0  # Minimum distance between first trajectory point with last trajectory point. Defaults to 10.0.
    min_hit_streak       : int   = 10     # Minimum number of `consecutive` frames that track appears. Defaults to 10.
    max_age              : int   = 1      # Maximum frame to wait until a dead track can be counted. Defaults to 1.

    def __init__(
        self,
        instances   : Any,
        motion      : mo.Motion,
        id_         : int | str         = uuid.uuid4().int,
        moving_state: MovingState | str = MovingState.CANDIDATE,
        moi_id      : int | str | None  = None,
        timestamp   : float             = timer(),
        frame_index : int               = -1,
    ):
        instances = [instances] if not isinstance(instances, list | tuple) else instances
        super().__init__(instance.Instance.from_value(i) for i in instances)
        self.uid          = id_
        self.motion       = motion
        self.moving_state = moving_state
        self.moi_uid      = moi_id
        self.timestamp    = timestamp
        self.frame_index  = frame_index
        self.trajectory   = [self.current.box_center]
    
    def __setitem__(self, index: int, item: instance.Instance):
        super().__setitem__(index, instance.Instance.from_value(item))
    
    def insert(self, index: int, item: instance.Instance):
        super().insert(index, instance.Instance.from_value(item))
    
    def append(self, item: instance.Instance):
        super().append(instance.Instance.from_value(item))
    
    def extend(self, other: Any):
        if isinstance(other, type(self)):
            super().extend(other)
        else:
            super().extend(instance.Instance.from_value(item) for item in other)
    
    @property
    def classlabels(self) -> list:
        return [d.classlabel for d in self]
    
    @property
    def first(self) -> instance.Instance:
        return self[0]
    
    @property
    def last(self) -> instance.Instance:
        return self[-1]
    
    @property
    def current(self) -> instance.Instance:
        """An alias to :meth:`last`."""
        return self.last
    
    @property
    def majority_label(self) -> dict:
        return mon.majority_voting(labels=self.classlabels)
    
    @property
    def majority_label_id(self) -> int:
        return self.majority_label["id"]
    
    @property
    def majority_roi_id(self) -> Union[int, str]:
        """The major ROI's uid of the object."""
        roi_id = Counter(self.roi_ids).most_common(1)
        return roi_id[0][0]
    
    @property
    def roi_ids(self) -> list[int | str]:
        return [d.roi_id for d in self]
    
    @property
    def moving_state(self) -> MovingState:
        return self._moving_state
    
    @moving_state.setter
    def moving_state(self, moving_state: MovingState | str):
        self._moving_state = MovingState.from_value(value=moving_state)
    
    @property
    def is_candidate(self) -> bool:
        return self.moving_state == MovingState.CANDIDATE
    
    @property
    def is_confirmed(self) -> bool:
        return self.moving_state == MovingState.CONFIRMED
    
    @property
    def is_counting(self) -> bool:
        return self.moving_state == MovingState.COUNTING
    
    @property
    def is_countable(self) -> bool:
        return True if (self.moi_uid is not None) else False
    
    @property
    def is_to_be_counted(self) -> bool:
        return self.moving_state == MovingState.TO_BE_COUNTED
    
    @property
    def is_counted(self) -> bool:
        return self.moving_state == MovingState.COUNTED
    
    @property
    def is_exiting(self) -> bool:
        return self.moving_state == MovingState.EXITING
    
    @property
    def motion(self) -> mo.Motion:
        return self._motion

    @motion.setter
    def motion(self, motion: mo.Motion):
        if isinstance(motion, mo.Motion):
            self._motion = motion
        elif isinstance(motion, Callable):
            self._motion = motion(instance=self.current)
        elif isinstance(motion, str):
            self._motion = MOTIONS.build(name=motion, instance=self.current)
        elif isinstance(motion, dict):
            self._motion = MOTIONS.build(cfg=motion, instance=self.current)
        else:
            raise TypeError
        
    @property
    def age(self) -> int:
        """The number of frames while the track is alive, from
        Candidate -> Deleted.
        """
        return self.motion.age
    
    @property
    def hits(self) -> int:
        """The number of frames has that track appear."""
        return self.motion.hits
    
    @property
    def hit_streak(self) -> int:
        """The number of `consecutive` frames has that track appear."""
        return self.motion.hit_streak
    
    @property
    def time_since_update(self) -> int:
        """The number of `consecutive` frames that track disappears."""
        return self.motion.time_since_update
    
    @property
    def traveled_distance(self) -> float:
        """The total traveled distance of the object."""
        if len(self.trajectory) < 2:
            return 0.0
        return mon.distance.euclidean(u=self.trajectory[0], v=self.trajectory[-1])
    
    def traveled_distance_between(self, start: int = -1, end: int = -2) -> float:
        """The traveled distance of the object between :param:`start` and
        :param:`end` frames.
        """
        step = abs(end - start)
        if len(self.trajectory) < step:
            return 0.0
        return mon.distance.euclidean(u=self.trajectory[start], v=self.trajectory[end])
    
    def update(self, instance: instance.Instance | None, **_):
        """Update the object with a new detected instances."""
        self.append(instance)
        self.update_trajectory()
        self.motion.update(instance=instance)
     
    def update_trajectory(self):
        """Update trajectory with a new instance's center point."""
        d = mon.distance.euclidean(u=self.trajectory[-1], v=self.current.box_center)
        if d >= self.min_traveled_distance:
            self.trajectory.append(list(self.current.box_center))
            
    @abstractmethod
    def update_moving_state(self, rois: list[rmoi.ROI], **kwargs):
        """Update the current state of the road_objects. One recommendation of
        the state diagram is as follows:
        _____________      _____________      ____________      ___________      ________
        | Candidate | ---> | Confirmed | ---> | Counting | ---> | Counted | ---> | Exit |
        -------------      -------------      ------------      -----------      --------
              |                 |                                                   ^
              |_________________|___________________________________________________|
                    (mark by a tracker when road_objects's max age > threshold)
        """
        pass
     
    def draw(
        self,
        image  : np.ndarray,
        bbox   : bool = True,
        polygon: bool = False,
        label  : bool = True,
        color  : list[int] | None = None
    ) -> np.ndarray:
        """Draw the current object and its trajectory on the :param:`image`."""
        if self.moi_uid is not None:
            color = mon.AppleRGB.values()[self.moi_uid]
        else:
            color = color or self.majority_label["color"]
        
        if self.is_candidate:
            pass
        elif self.is_confirmed:
            image = self.draw_instance(
                image   = image,
                bbox    = bbox,
                polygon = polygon,
                label   = label,
                color   = [255, 255, 255]
            )
            image = mon.draw_trajectory(
                image      = image,
                trajectory = self.trajectory,
                color      = [255, 255, 255],
                thickness  = 3,
            )
        elif self.is_counting:
            image = self.draw_instance(
                image   = image,
                bbox    = bbox,
                polygon = polygon,
                label   = label,
                color   = [255, 255, 255]
            )
            image = mon.draw_trajectory(
                image      = image,
                trajectory = self.trajectory,
                color      = [255, 255, 255],
                thickness  = 3,
            )
        elif self.is_counted:
            image = self.draw_instance(
                image   = image,
                bbox    = bbox,
                polygon = polygon,
                label   = label,
                color   = color
            )
            image = mon.draw_trajectory(
                image      = image,
                trajectory = self.trajectory,
                color      = color,
                thickness  = 3,
            )
        elif self.is_exiting:
            image = self.draw_instance(
                image   = image,
                bbox    = bbox,
                polygon = polygon,
                label   = label,
                color   = color
            )
            image = mon.draw_trajectory(
                image      = image,
                trajectory = self.trajectory,
                color      = color,
                thickness  = 3,
            )
        return image
    
    def draw_instance(
        self,
        image  : np.ndarray,
        bbox   : bool             = True,
        polygon: bool             = False,
        label  : bool             = True,
        color  : list[int] | None = None
    ) -> np.ndarray:
        """Draw the current object on the :param:`image`."""
        color = color or self.majority_label["color"]
        if bbox:
            b = self.current.bbox
            cv2.rectangle(
                img       = image,
                pt1       = (int(b[0]), int(b[1])),
                pt2       = (int(b[2]), int(b[3])),
                color     = color,
                thickness = 2
            )
            b_center = self.current.box_center.astype(int)
            cv2.circle(
                img       = image,
                center    = tuple(b_center),
                radius    = 3,
                thickness = -1,
                color     = color
            )
        if polygon:
            pts = self.current.polygon.reshape((-1, 1, 2))
            cv2.polylines(img=image, pts=pts, isClosed=True, color=color, thickness=2)
        if label is not None:
            box_tl     = self.current.bbox[0:2]
            curr_label = self.majority_label
            font       = cv2.FONT_HERSHEY_SIMPLEX
            org        = (int(box_tl[0]) + 5, int(box_tl[1]))
            cv2.putText(
                img       = image,
                text      = curr_label["name"],
                fontFace  = font,
                fontScale = 1.0,
                org       = org,
                color     = color,
                thickness = 2,
            )
        return image
    
    
# endregion

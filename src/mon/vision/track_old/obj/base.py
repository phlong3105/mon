#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the base classes for all tracking objects."""

from __future__ import annotations

__all__ = [
    "Instance", "MovingObject", "Object", "StaticObject", "Track",
]

import uuid
from abc import ABC, abstractmethod
from collections import Counter
from timeit import default_timer as timer
from typing import Any, Callable

import cv2
import numpy as np

from mon import core, nn
from mon.globals import AppleRGB, MOTIONS, MovingState
from mon.vision import geometry, utils
from mon.vision.track_old import motion as mmotion

console = core.console


# region Instance

class Instance:
    """An instance of a track (i.e, moving object) in a given frame. This class
    is mainly used to wrap and pas data between detectors and trackers.
    
    Attributes:
        id_: A unique ID. Default: ``None``.
        roi_id: The unique ID of the ROI containing the
        bbox: A bounding bbox in XYXY format.
        polygon: A list of points representing an instance mask. Default: ``None``.
        feature: A feature vector that describes the object contained in this
            image.
        confidence: A confidence score. Default: ``None``.
        classlabel: A :class:`mon.Classlabel` object. Default: ``None``.
        frame_index: The current frame index. Default: ``None``.
        timestamp: The creating time of the current instance.
    """
    
    def __init__(
        self,
        id_        : int | str         = uuid.uuid4().int,
        roi_uid    : int | str  | None = None,
        bbox       : np.ndarray | None = None,
        polygon    : np.ndarray | None = None,
        feature    : np.ndarray | None = None,
        confidence : float      | None = None,
        classlabel : dict       | None = None,
        frame_index: int        | None = None,
        timestamp  : int | float       = timer(),
    ):
        self.id_         = id_
        self.roi_id      = roi_uid
        self.bbox        = np.array(bbox)    if bbox    is not None else None
        self.polygon     = np.array(polygon) if polygon is not None else None
        self.feature     = np.array(feature) if feature is not None else None
        self.confidence  = confidence
        self.classlabel  = classlabel
        self.frame_index = (frame_index + 1) if frame_index is not None else None
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
    def bbox_center(self):
        return geometry.get_bbox_center(bbox=self.bbox)
    
    @property
    def bbox_tl(self):
        """The bbox's top left corner."""
        return self.bbox[0:2]
    
    @property
    def bbox_corners_points(self) -> np.ndarray:
        return geometry.get_bbox_corners_points(bbox=self.bbox)
        
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
            org = (self.bbox_tl[0] + 5, self.bbox_tl[1])
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


# region Object

class Object(ABC):
    """The base class for all objects."""
    pass


class StaticObject(Object):
    """The base class for all static objects (i.e., trees, poles, traffic
    lights, etc.).
    """
    pass


class MovingObject(list[Instance], Object):
    """The base class for all moving objects. It is a list that contains all
    instances of the object in consecutive frames. The list is sorted in a
    timely manner.
    
    See more: :class:`Object`.
    
    Args:
        instances: The first instance of this object, or a list of instances.
        id_: The object unique ID.
        motion: A motion model.
        moving_state: The current state of the moving object with respect to the
            ROIs. Default: ``'Candidate'``.
        moi_id: The unique ID of the MOI. Default: ``None``.
        timestamp: The time when the object is created.
        frame_index: The frame index when the object is created.
    """
    
    min_entering_distance: float = 0.0    # Minimum distance when an object enters the ROI to be `Confirmed`. Default: 0.0.
    min_traveled_distance: float = 100.0  # Minimum distance between first trajectory point with last trajectory point. Default: 10.0.
    min_hit_streak       : int   = 10     # Minimum number of `consecutive` frames that track appears. Default: 10.
    max_age              : int   = 1      # Maximum frame to wait until a dead track can be counted. Default: 1.

    def __init__(
        self,
        instances   : Any,
        motion      : mmotion.Motion,
        id_         : int | str         = uuid.uuid4().int,
        moving_state: MovingState | str = MovingState.CANDIDATE,
        moi_id      : int | str | None  = None,
        timestamp   : int | float       = timer(),
        frame_index : int               = -1,
    ):
        instances = [instances] if not isinstance(instances, list | tuple) else instances
        super().__init__(Instance.from_value(i) for i in instances)
        self.id_          = id_
        self.motion       = motion
        self.moving_state = moving_state
        self.moi_id       = moi_id
        self.timestamp    = timestamp
        self.frame_index  = frame_index
        self.trajectory   = [self.current.bbox_center]
    
    def __setitem__(self, index: int, item: Instance):
        super().__setitem__(index, Instance.from_value(item))
    
    def insert(self, index: int, item: Instance):
        super().insert(index, Instance.from_value(item))
    
    def append(self, item: Instance):
        super().append(Instance.from_value(item))
    
    def extend(self, other: Any):
        if isinstance(other, type(self)):
            super().extend(other)
        else:
            super().extend(Instance.from_value(item) for item in other)
    
    @property
    def classlabels(self) -> list:
        return [d.classlabel for d in self]
    
    @property
    def first(self) -> Instance:
        return self[0]
    
    @property
    def last(self) -> Instance:
        return self[-1]
    
    @property
    def current(self) -> Instance:
        """An alias to :meth:`last`."""
        return self.last
    
    @property
    def majority_label(self) -> dict:
        return nn.majority_voting(labels=self.classlabels)
    
    @property
    def majority_label_id(self) -> int:
        return self.majority_label["id"]
    
    @property
    def majority_roi_id(self) -> int | str:
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
        return True if (self.moi_id is not None) else False
    
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
    def motion(self) -> mmotion.Motion:
        return self._motion

    @motion.setter
    def motion(self, motion: mmotion.Motion):
        if isinstance(motion, mmotion.Motion):
            self._motion = motion
        elif isinstance(motion, Callable):
            self._motion = motion(instance=self.current)
        elif isinstance(motion, str):
            self._motion = MOTIONS.build(name=motion, instance=self.current)
        elif isinstance(motion, dict):
            self._motion = MOTIONS.build(config=motion, instance=self.current)
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
        return geometry.distance.euclidean(u=self.trajectory[0], v=self.trajectory[-1])
    
    def traveled_distance_between(self, start: int = -1, end: int = -2) -> float:
        """The traveled distance of the object between :param:`start` and
        :param:`end` frames.
        """
        step = abs(end - start)
        if len(self.trajectory) < step:
            return 0.0
        return geometry.distance.euclidean(u=self.trajectory[start], v=self.trajectory[end])
    
    def update(self, instance: Instance | None, **_):
        """Update the object with a new detected instances."""
        self.append(instance)
        self.update_trajectory()
        self.motion.update(instance=instance)
     
    def update_trajectory(self):
        """Update trajectory with a new instance's center point."""
        d = geometry.distance.euclidean(u=self.trajectory[-1], v=self.current.bbox_center)
        if d >= self.min_traveled_distance:
            self.trajectory.append(list(self.current.bbox_center))
            
    @abstractmethod
    def update_moving_state(self, **kwargs):
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
        bbox   : bool             = True,
        polygon: bool             = False,
        label  : bool             = True,
        color  : list[int] | None = None
    ) -> np.ndarray:
        """Draw the current object and its trajectory on the :param:`image`."""
        if self.moi_id is not None:
            color = AppleRGB.values()[self.moi_id]
        else:
            color = color or self.majority_label["color"]
        
        if self.is_candidate:
            image = self.draw_instance(
                image   = image,
                bbox    = bbox,
                polygon = polygon,
                label   = False,
                color   = [255, 255, 255]
            )
            image = utils.draw_trajectory(
                image      = image,
                trajectory = self.trajectory,
                color      = [255, 255, 255],
                thickness  = 3,
            )
        elif self.is_confirmed:
            image = self.draw_instance(
                image   = image,
                bbox    = bbox,
                polygon = polygon,
                label   = False,
                color   = [255, 255, 255]
            )
            image = utils.draw_trajectory(
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
            image = utils.draw_trajectory(
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
            image = utils.draw_trajectory(
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
            image = utils.draw_trajectory(
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
            b_center = self.current.bbox_center.astype(int)
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
        if label:
            box_tl     = self.current.bbox[0:2]
            curr_label = self.majority_label
            text       = f"{curr_label['name']}, {curr_label['id'] + 1}"
            font       = cv2.FONT_HERSHEY_SIMPLEX
            org        = (int(box_tl[0]) + 5, int(box_tl[1]))
            cv2.putText(
                img       = image,
                text      = text,
                fontFace  = font,
                fontScale = 1.0,
                org       = org,
                color     = color,
                thickness = 2,
            )
        return image


Track = MovingObject

# endregion

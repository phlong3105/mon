#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the base class for all moving objects."""

from __future__ import annotations

__all__ = [
    "Instance",
    "MovingObject",
    "Object",
]

import uuid
from abc import ABC, abstractmethod
from collections import Counter
from timeit import default_timer as timer
from typing import Sequence, TYPE_CHECKING, Union

import cv2
import numpy as np

import mon
from supr import constant, rmoi, tracking

if TYPE_CHECKING:
    from supr.typing import Ints, UIDType


# region Instance

class Instance:
    """An instances of a moving object in a given frame. This class is mainly
    used to pass data between detectors and trackers.
    
    Attributes:
        uid: An unique ID. Defaults to None.
        roi_uid: The unique ID of the ROI containing the 
        box: A bounding box in (x1, y1, x2, y2) format.
        polygon: A list of points representing an instances mask. Defaults to
            None.
        confidence: A confidence score. Defaults to None.
        classlabel: A :class:`mon.Classlabel` object. Defaults to None.
        frame_index: The frame index of a  Defaults to None.
        timestamp: The creating time of the current object.
    """
    
    def __init__(
        self,
        uid        : UIDType,
        roi_uid    : UIDType,
        box        : np.ndarray,
        polygon    : np.ndarray | None = None,
        confidence : float      | None = None,
        classlabel : dict       | None = None,
        frame_index: int        | None = None,
        timestamp  : float             = timer(),
    ):
        super().__init__()
        self.uid         = uid
        self.roi_uid     = roi_uid
        self.box         = box
        self.polygon     = polygon
        self.confidence  = confidence
        self.classlabel  = classlabel
        self.frame_index = frame_index
        self.timestamp   = timestamp
        
    @property
    def box_cxcyrh(self):
        return mon.box_xyxy_to_cxcyrh(box=self.box)

    @property
    def box_center(self):
        return mon.get_box_center(box=self.box)
    
    @property
    def box_tl(self):
        """The box's top left corner."""
        return self.box[0:2]
        
    def draw(
        self,
        drawing: np.ndarray,
        box    : bool        = False,
        polygon: bool        = False,
        label  : bool        = True,
        color  : Ints | None = None
    ) -> np.ndarray:
        """Draw the current object on the :param:`drawing`."""
        color = color \
            or (self.classlabel["color"] if self.classlabel is not None
                else (255, 255, 255))
        
        if box:
            cv2.rectangle(
                img       = drawing,
                pt1       = (self.box[0], self.box[1]),
                pt2       = (self.box[2], self.box[3]),
                color     = color,
                thickness = 2
            )
        
        if polygon:
            pts = self.polygon.reshape((-1, 1, 2))
            cv2.polylines(
                img       = drawing,
                pts       = pts,
                isClosed  = True,
                color     = color,
                thickness = 2,
            )
        
        if label:
            font = cv2.FONT_HERSHEY_SIMPLEX
            org  = (self.box_tl[0] + 5, self.box_tl[1])
            cv2.putText(
                img       = drawing,
                text      = self.classlabel["name"],
                fontFace  = font,
                fontScale = 1.0,
                org       = org,
                color     = color,
                thickness = 2
            )
        return drawing

# endregion


# region Moving Object

class Object(ABC):
    """The base class for all objects."""
    pass


class MovingObject(Object, list):
    """The base class for all moving objects. It is a list that contains all
    instances of the object in consecutive frames. The list is sorted in a
    timely manner.
    
    See more: :class:`Object`.
    
    Args:
        instances: The first instance of this object, or a list of instances.
        uid: The object unique ID.
        motion: A motion model.
        moving_state: The current state of the moving object with respect to the
            ROIs. Defaults to 'Candidate'.
        moi_uid: The unique ID of the MOI. Defaults to None.
        timestamp: The time when the object is created.
        frame_index: The frame index when the object is created.
    """
    
    min_entering_distance: float = 0.0    # Min distance when an object enters the ROI to be `Confirmed`. Default: `0.0`.
    min_traveled_distance: float = 100.0  # Min distance between first trajectory point with last trajectory point. Default: `10.0`.
    min_hit_streak       : int   = 10     # Min number of `consecutive` frame has that track appear. Default: `10`.
    max_age              : int   = 1      # Max frame to wait until a dead track can be counted. Default: `1`.

    def __init__(
        self,
        instances   : Instance | Sequence[Instance],
        motion      : tracking.Motion,
        uid         : UIDType              = uuid.uuid4().int,
        moving_state: constant.MovingState = constant.MovingState.Candidate,
        moi_uid     : UIDType | None       = None,
        timestamp   : float 	           = timer(),
        frame_index : int                  = -1,
    ):
        instances = [instances] if not isinstance(instances, list | tuple) else instances
        assert all(i for i in instances if isinstance(i, Instance))
        super().__init__(i for i in instances)
        
        self.uid          = uid
        self.motion       = motion
        self.moving_state = moving_state
        self.moi_uid      = moi_uid
        self.timestamp    = timestamp
        self.frame_index  = frame_index
        self.trajectory   = [self.current_box_center]
    
    def __setitem__(self, index: int, item: Instance):
        assert isinstance(item, Instance)
        super().__setitem__(index, item)
    
    def insert(self, index: int, item: Instance):
        assert isinstance(item, Instance)
        super().insert(index, item)
    
    def append(self, item: Instance):
        assert isinstance(item, Instance)
        super().append(item)
    
    def extend(self, other: Instance | Sequence[Instance]):
        if isinstance(other, type(self)):
            super().extend(other)
        else:
            super().extend(item for item in other)
    
    @property
    def classlabels(self) -> list:
        return [d.classlabel for d in self]
    
    @property
    def first(self) -> np.ndarray:
        return self[0]
    
    @property
    def first_box(self) -> np.ndarray:
        return self[0].box
    
    @property
    def current(self) -> Instance:
        return self[-1]
    
    @property
    def current_box(self) -> np.ndarray:
        return self[-1].box
    
    @property
    def current_box_center(self) -> np.ndarray:
        return self[-1].box_center
    
    @property
    def current_box_corners_points(self) -> np.ndarray:
        return mon.get_box_corners_points(box=self[-1].box)
    
    @property
    def current_confidence(self) -> float:
        return self[-1].confidence
    
    @property
    def current_classlabel(self) -> dict:
        return self[-1].classlabel
    
    @property
    def current_frame_index(self) -> int:
        return self[-1].frame_index
    
    @property
    def current_polygon(self) -> np.ndarray:
        return self[-1].polygon
    
    @property
    def current_roi_id(self) -> Union[int, str]:
        return self[-1].roi_uid
    
    @property
    def current_timestamp(self) -> float:
        return self[-1].timestamp
    
    @property
    def majority_label(self) -> dict:
        return mon.majority_voting(labels=self.classlabels)
    
    @property
    def majority_label_id(self) -> int:
        return self.majority_label["id"]
    
    @property
    def majority_roi_uid(self) -> Union[int, str]:
        """The major ROI's uid of the object."""
        roi_id = Counter(self.roi_uids).most_common(1)
        return roi_id[0][0]
    
    @property
    def roi_uids(self) -> list[UIDType]:
        return [d.roi_uid for d in self]
    
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
    
    @property
    def moving_state(self) -> constant.MovingState:
        return self._moving_state
    
    @moving_state.setter
    def moving_state(self, moving_state: constant.MovingState):
        if moving_state not in constant.MovingState.keys():
            raise ValueError(
                f"Moving state should be one of: {constant.MovingState.keys()}. "
                f"But got: {moving_state}."
            )
        self._moving_state = moving_state
    
    @property
    def is_candidate(self) -> bool:
        return self.moving_state == constant.MovingState.Candidate
    
    @property
    def is_confirmed(self) -> bool:
        return self.moving_state == constant.MovingState.Confirmed
    
    @property
    def is_counting(self) -> bool:
        return self.moving_state == constant.MovingState.Counting
    
    @property
    def is_countable(self) -> bool:
        return True if (self.moi_uid is not None) else False
    
    @property
    def is_to_be_counted(self) -> bool:
        return self.moving_state == constant.MovingState.ToBeCounted
    
    @property
    def is_counted(self) -> bool:
        return self.moving_state == constant.MovingState.Counted
    
    @property
    def is_exiting(self) -> bool:
        return self.moving_state == constant.MovingState.Exiting
    
    def update(self, instance: Instance | None, **_):
        """Update the object with a new detected instances."""
        self.append(instance)
        self.motion.update_motion_state(instance=instance)
        self.update_trajectory()
    
    @abstractmethod
    def update_trajectory(self):
        """Update trajectory with measurement's center point."""
        pass
    
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
        drawing: np.ndarray,
        box    : bool        = True,
        polygon: bool        = False,
        label  : bool        = True,
        color  : Ints | None = None
    ) -> np.ndarray:
        """Draw the current object and its trajectory on the :param:`drawing`.
        """
        if self.moi_uid is not None:
            color = mon.AppleRGB.values()[self.moi_uid]
        else:
            color = color or self.majority_label["color"]
        if self.is_confirmed:
            self.draw_trajectory(drawing=drawing)
        elif self.is_counting:
            self.draw_trajectory(drawing=drawing)
        elif self.is_counted:
            self.draw_instance(
                drawing = drawing,
                box     = box,
                polygon = polygon,
                label   = label,
                color   = color
            )
            self.draw_trajectory(drawing=drawing)
        elif self.is_exiting:
            self.draw_instance(
                drawing = drawing,
                box     = box,
                polygon = polygon,
                label   = label,
                color   = color
            )
            self.draw_trajectory(drawing=drawing)
        return drawing
    
    def draw_instance(
        self,
        drawing: np.ndarray,
        box    : bool        = True,
        polygon: bool        = False,
        label  : bool        = True,
        color  : Ints | None = None
    ) -> np.ndarray:
        """Draw the current object on the :param:`drawing`."""
        color = color or self.majority_label["color"]
        if box:
            b = self.current_box
            cv2.rectangle(
                img       = drawing,
                pt1       = (b[0], b[1]),
                pt2       = (b[2], b[3]),
                color     = color,
                thickness = 2
            )
            b_center = self.current_box_center.astype(int)
            cv2.circle(
                img       = drawing,
                center    = tuple(b_center),
                radius    = 3,
                thickness = -1,
                color     = color
            )
        if polygon:
            pts = self.current_polygon.reshape((-1, 1, 2))
            cv2.polylines(
                img=drawing, pts=pts, isClosed=True, color=color, thickness=2
            )
        if label is not None:
            box_tl     = box[0:2]
            curr_label = self.majority_label
            font       = cv2.FONT_HERSHEY_SIMPLEX
            org        = (box_tl[0] + 5, box_tl[1])
            cv2.putText(
                img       = drawing,
                text      = curr_label["name"],
                fontFace  = font,
                fontScale = 1.0,
                org       = org,
                color     = color,
                thickness = 2,
            )
        return drawing
    
    def draw_trajectory(self, drawing: np.ndarray) -> np.ndarray:
        """Draw the current trajectory on the :param:`drawing`."""
        if self.moi_uid is not None:
            color = mon.AppleRGB.values()[self.moi_uid]
        else:
            color = self.majority_label["color"]
    
        if self.trajectory is not None:
            pts = np.array(self.trajectory).reshape((-1, 1, 2))
            cv2.polylines(
                img       = drawing,
                pts       = [pts.astype(int)],
                isClosed  = False,
                color     = color,
                thickness = 2
            )
            for point in self.trajectory:
                cv2.circle(
                    img       = drawing,
                    center    = tuple(point.astype(int)),
                    radius    = 3,
                    thickness = 2,
                    color     = color
                )
        return drawing
    
# endregion

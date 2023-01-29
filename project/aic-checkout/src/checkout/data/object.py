#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the base class for all persistent objects."""

from __future__ import annotations

__all__ = [
    "MovingObject", "MovingState", "Object", "TemporalObject",
]

import uuid
from abc import ABC, abstractmethod
from collections import Counter
from timeit import default_timer as timer
from typing import Sequence, TYPE_CHECKING, Union

import cv2
import numpy as np

import mon
from checkout import rmoi, tracking
from checkout.data import detection

if TYPE_CHECKING:
    from checkout.typing import Ints, UIDType


# region MovingState

class MovingState(mon.Enum):
    """The counting state of an object when moving through the camera."""
    Candidate   = 1  # Preliminary state.
    Confirmed   = 2  # Confirmed the Detection is a road_objects eligible for counting.
    Counting    = 3  # Object is in the counting zone/counting state.
    ToBeCounted = 4  # Mark object to be counted somewhere in this loop iteration.
    Counted     = 5  # Mark object has been counted.
    Exiting     = 6  # Mark object for exiting the ROI or image frame. Let's it die by itself.

# endregion


# region Object

class Object(ABC):
    """The base class for all objects."""
    pass


class TemporalObject(Object, list):
    """An object that persists throughout several frames. It contains all
    detected instances in consecutive frames. The list is sorted in a timely
    manner.
    
    Args:
        iterable: The first detected instance of this object, or a list of
            detected instances.
        uid: The object unique ID.
        timestamp: The time when the object is created.
        frame_index: The frame index when the object is created.
    """
    
    def __init__(
        self,
        iterable   : Sequence[detection.Detection],
        uid        : UIDType = uuid.uuid4().int,
        timestamp  : float 	 = timer(),
        frame_index: int     = -1,
    ):
        assert isinstance(iterable, list | tuple) \
               and all(isinstance(i, detection.Detection) for i in iterable)
        super().__init__(i for i in iterable)
        self.uid         = uid
        self.timestamp   = timestamp
        self.frame_index = frame_index
    
    def __setitem__(self, index: int, item: detection.Detection):
        assert isinstance(item, detection.Detection)
        super().__setitem__(index, item)
    
    def insert(self, index: int, item: detection.Detection):
        assert isinstance(item, detection.Detection)
        super().insert(index, item)

    def append(self, item: detection.Detection):
        assert isinstance(item, detection.Detection)
        super().append(item)

    def extend(self, other):
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
    def current(self) -> detection.Detection:
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
    
    def update(self, detection: detection.Detection | None, **_):
        """Update the object with a new detected instance."""
        self.append(detection)
        
    def draw(
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

# endregion


# region Moving Object

class MovingObject(TemporalObject, ABC):
    """The base class for all moving objects. Extend the :class:`Temporal` class
    with additional functionalities: motion, moving state, and MOI.

    Args:
        motion: A motion model.
        moving_state: The current state of the moving object with respect to the
            ROIs. Defaults to 'Candidate'.
        moi_uid: The unique ID of the MOI. Defaults to None.
    """

    min_entering_distance: float = 0.0    # Min distance when an object enters the ROI to be `Confirmed`. Default: `0.0`.
    min_traveled_distance: float = 100.0  # Min distance between first trajectory point with last trajectory point. Default: `10.0`.
    min_hit_streak       : int   = 10     # Min number of `consecutive` frame has that track appear. Default: `10`.
    max_age              : int   = 1      # Max frame to wait until a dead track can be counted. Default: `1`.

    def __init__(
        self,
        motion      : tracking.Motion,
        moving_state: MovingState    = MovingState.Candidate,
        moi_uid     : UIDType | None = None,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.motion       = motion
        self.moving_state = moving_state
        self.moi_uid       = moi_uid
        self.trajectory   = [self.current_box_center]

    @property
    def traveled_distance(self) -> float:
        """Return the traveled distance of the object."""
        if len(self.trajectory) < 2:
            return 0.0
        return mon.euclidean_distance(self.trajectory[0], self.trajectory[-1])
    
    def traveled_distance_between(self, start: int = -1, end: int = -2) -> float:
        """Return the recently traveled distance of the object between previous
        and current frames.
        """
        step = abs(end - start)
        if len(self.trajectory) < step:
            return 0.0
        return mon.euclidean_distance(self.trajectory[start], self.trajectory[end])
    
    @property
    def hits(self) -> int:
        """The number of frames has that track appear."""
        return self.motion.hits

    @property
    def hit_streak(self) -> int:
        """The number of `consecutive` frames has that track appear."""
        return self.motion.hit_streak

    @property
    def age(self) -> int:
        """The number of frames while the track is alive, from
        Candidate -> Deleted.
        """
        return self.motion.age

    @property
    def time_since_update(self) -> int:
        """The number of `consecutive` frames that track disappears."""
        return self.motion.time_since_update

    @property
    def moving_state(self) -> MovingState:
        return self._moving_state

    @moving_state.setter
    def moving_state(self, moving_state: MovingState):
        if moving_state not in MovingState.keys():
            raise ValueError(
                f"Moving state should be one of: {MovingState.keys()}. "
                f"But got: {moving_state}."
            )
        self._moving_state = moving_state

    @property
    def is_candidate(self) -> bool:
        return self.moving_state == MovingState.Candidate

    @property
    def is_confirmed(self) -> bool:
        return self.moving_state == MovingState.Confirmed

    @property
    def is_counting(self) -> bool:
        return self.moving_state == MovingState.Counting

    @property
    def is_countable(self) -> bool:
        return True if (self.moi_uid is not None) else False

    @property
    def is_to_be_counted(self) -> bool:
        return self.moving_state == MovingState.ToBeCounted

    @property
    def is_counted(self) -> bool:
        return self.moving_state == MovingState.Counted

    @property
    def is_exiting(self) -> bool:
        return self.moving_state == MovingState.Exiting
    
    def update(self, detection: detection.Detection, **kwargs):
        """Update the object with a new detected instance."""
        super(MovingObject, self).update(detection=detection, **kwargs)
        self.motion.update_motion_state(detection=detection)
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

    def draw(self, drawing: np.ndarray, **kwargs) -> np.ndarray:
        """Draw the current object on the :param:`drawing`."""
        if self.moi_uid is not None:
            color = mon.AppleRGB.values()[self.moi_uid]
        else:
            color = self.majority_label["color"]
            
        if self.is_confirmed:
            # TemporalObject.draw(self, drawing=drawing, label=False, **kwargs)
            self.draw_trajectory(drawing=drawing)
        elif self.is_counting:
            # TemporalObject.draw(self, drawing=drawing, label=True, **kwargs)
            self.draw_trajectory(drawing=drawing)
        elif self.is_counted:
            TemporalObject.draw(self, drawing=drawing, label=True, color=color, **kwargs)
            self.draw_trajectory(drawing=drawing)
        elif self.is_exiting:
            TemporalObject.draw(self, drawing=drawing, label=True, color=color, **kwargs)
            self.draw_trajectory(drawing=drawing)
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

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the base class for all persistent objects."""

from __future__ import annotations

__all__ = [
    "Object",
    "MovingObject",
    "MovingState",
]

import uuid
from abc import ABC
from collections import Counter
from timeit import default_timer as timer
from typing import Optional, Union

import cv2
import numpy as np

import mon
from checkout.typing import UIDType


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

class Object:
    """The base class for all persistent objects.
    
    Requires:
        It is required to be subclassed with the motion model. If you want
        to use it without tracking or counting functions.
    
    Args:
        uid: The object unique ID.
        detections: A list of all detection instances of this object.
        timestamp: The time when the object is created.
        frame_index: The frame index when the object is created.
    """

    # MARK: Magic Functions

    def __init__(
        self,
        uid        : UIDType                 = uuid.uuid4().int,
        detection  : Union[list, Detection] | None = None,
        timestamp  : float 					   	      = timer(),
        frame_index: int                              = -1,
    ):
        super().__init__()
        self.id_         = uid
        self.timestamp   = timestamp
        self.frame_index = frame_index
        self.detections: list[Detection] = []
        
        if isinstance(detection, Detection):
            self.detections = [detection]
        else:
            self.detections = (detection if (detection is not None) else [])

    # MARK: Properties

    @property
    def class_labels(self) -> list:
        """Get the list of all class_labels of the object."""
        return [d.classlabel for d in self.detections]

    @property
    def first_box(self) -> np.ndarray:
        """Get the first det_box of the object."""
        return self.detections[0].box
    
    @property
    def current_box(self) -> np.ndarray:
        """Get the latest det_box of the object."""
        return self.detections[-1].box
    
    @property
    def current_box_center(self) -> np.ndarray:
        """Get the latest center of the object."""
        return self.detections[-1].box_center
    
    @property
    def current_box_corners_points(self) -> np.ndarray:
        """Get the latest corners of bounding boxes as points."""
        return get_box_corners_points(self.detections[-1].box)

    @property
    def current_confidence(self) -> float:
        """Get the latest confidence score."""
        return self.detections[-1].confidence

    @property
    def current_class_label(self) -> dict:
        """Get the latest label of the object."""
        return self.detections[-1].classlabel

    @property
    def current_frame_index(self) -> int:
        """Get the latest frame index of the object."""
        return self.detections[-1].frame_index

    @property
    def current_instance(self) -> Detection:
        """Get the latest measurement of the object."""
        return self.detections[-1]

    @property
    def current_polygon(self) -> np.ndarray:
        """Get the latest polygon of the object."""
        return self.detections[-1].polygon

    @property
    def current_roi_id(self) -> Union[int, str]:
        """Get the latest ROI's id of the object."""
        return self.detections[-1].roi_uid

    @property
    def current_timestamp(self) -> float:
        """Get the last time the object has been updated."""
        return self.detections[-1].timestamp

    @property
    def label_by_majority(self) -> dict:
        """Get the major class_label of the object."""
        return majority_voting(self.class_labels)
    
    @property
    def label_id_by_majority(self) -> int:
        """Get the most popular label's id of the object."""
        return self.label_by_majority["id"]

    @property
    def roi_id_by_majority(self) -> Union[int, str]:
        """Get the major ROI's id of the object."""
        roi_id = Counter(self.roi_ids).most_common(1)
        return roi_id[0][0]

    @property
    def roi_ids(self) -> list[Union[int, str]]:
        """Get the list ROI's ids of the object."""
        return [d.roi_uid for d in self.detections]

    # MARK: Update
    
    def update(self, detection: Optional[Detection], **kwargs):
        """Update with new measurement.
        
        Args:
            detection (Detection, optional):
                Detection of the object.
        """
        self.detections.append(detection)

    # MARK: Visualize
    
    def draw(
        self,
        drawing: np.ndarray,
        box    : bool            = True,
        polygon: bool            = False,
        label  : bool            = True,
        color  : Optional[Color] = None
    ) -> np.ndarray:
        """Draw the object into the `drawing`.
        
        Args:
            drawing (np.ndarray):
                Drawing canvas.
            box (bool):
                Should draw the detected box? Default: `True`.
            polygon (bool):
                Should draw polygon? Default: `False`.
            label (bool):
                Should draw label? Default: `True`.
            color (tuple):
                Primary color. Default: `None`.
        """
        color = (color if (color is not None) else self.label_by_majority["color"])
        box   = self.current_box

        if box is not None:
            cv2.rectangle(
                img       = drawing,
                pt1       = (box[0], box[1]),
                pt2       = (box[2], box[3]),
                color     = color,
                thickness = 2
            )
            box_center = self.current_box_center.astype(int)
            cv2.circle(
                img       = drawing,
                center    = tuple(box_center),
                radius    = 3,
                thickness = -1,
                color     = color
            )
        
        """
        if polygon is not None:
            pts = self.current_polygon.reshape((-1, 1, 2))
            cv2.polylines(
                img=drawing, pts=pts, isClosed=True, color=color, thickness=2
            )
        """
        
        if label is not None:
            box_tl     = box[0:2]
            curr_label = self.label_by_majority
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


# MARK: - BaseMovingObject

class MovingObject(Object, ABC):
    """Base Moving Object.

    Attributes:
        motion (Motion):
            Motion model.
        moving_state (MovingState):
            Current state of the moving object with respect to camera's
            ROIs. Default: `Candidate`.
        moi_id (int, str, optional):
            ID of the MOI that the current moving object is best fitted to.
            Default: `None`.
        trajectory (np.ndarray):
            Object trajectory as an array of detections' center points.
    """

    min_entering_distance: float = 0.0    # Min distance when an object enters the ROI to be `Confirmed`. Default: `0.0`.
    min_traveled_distance: float = 100.0  # Min distance between first trajectory point with last trajectory point. Default: `10.0`.
    min_hit_streak       : int   = 10     # Min number of `consecutive` frame has that track appear. Default: `10`.
    max_age              : int   = 1      # Max frame to wait until a dead track can be counted. Default: `1`.

    def __init__(
        self,
        motion      : Motion,
        moving_state: MovingState               = MovingState.Candidate,
        moi_id      : Optional[Union[int, str]] = None,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.motion       = motion
        self.moving_state = moving_state
        self.moi_id       = moi_id
        self.trajectory   = np.array([self.current_box_center])

    @property
    def traveled_distance(self) -> float:
        """Return the traveled distance of the object."""
        if len(self.trajectory) < 2:
            return 0.0
        return euclidean_distance(self.trajectory[0], self.trajectory[-1])
    
    def traveled_distance_between(self, start: int = -1, end: int = -2) -> float:
        """Return the recently traveled distance of the object between previous
        and current frames."""
        step = abs(end - start)
        if len(self.trajectory) < step:
            return 0.0
        return euclidean_distance(self.trajectory[start], self.trajectory[end])
    
    @property
    def hits(self) -> int:
        """Return the number of frame has that track appear."""
        return self.motion.hits

    @property
    def hit_streak(self) -> int:
        """Return the number of `consecutive` frame has that track appear."""
        return self.motion.hit_streak

    @property
    def age(self) -> int:
        """Return the number of frame while the track is alive,
        from Candidate -> Deleted."""
        return self.motion.age

    @property
    def time_since_update(self) -> int:
        """Return the number of `consecutive` frame that track disappear."""
        return self.motion.time_since_update

    @property
    def moving_state(self) -> MovingState:
        return self._moving_state

    @moving_state.setter
    def moving_state(self, moving_state: MovingState):
        """Assign moving state.

        Args:
            moving_state (MovingState):
                Object's moving state.
        """
        if moving_state not in MovingState.keys():
            raise ValueError(f"Moving state should be one of: "
                             f"{MovingState.keys()}. But given {moving_state}.")
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
        return True if (self.moi_id is not None) else False

    @property
    def is_to_be_counted(self) -> bool:
        return self.moving_state == MovingState.ToBeCounted

    @property
    def is_counted(self) -> bool:
        return self.moving_state == MovingState.Counted

    @property
    def is_exiting(self) -> bool:
        return self.moving_state == MovingState.Exiting
    
    def update(self, detection: Detection, **kwargs):
        """Update with value from a `Detection` object.

        Args:
            detection (Detection):
                Detection of the object.
        """
        super(MovingObject, self).update(detection=detection, **kwargs)
        self.motion.update_motion_state(detection=detection)
        self.update_trajectory()
    
    @abc.abstractmethod
    def update_trajectory(self):
        """Update trajectory with measurement's center point."""
        pass
    
    @abc.abstractmethod
    def update_moving_state(self, rois: list[ROI], **kwargs):
        """Update the current state of the road_objects.
        One recommendation of the state diagram is as follow:

                (exist >= 10 frames)  (road_objects cross counting line)   (after being counted
                (in roi)                                               by counter)
        _____________          _____________                  ____________        ___________        ________
        | Candidate | -------> | Confirmed | ---------------> | Counting | -----> | Counted | -----> | Exit |
        -------------          -------------                  ------------        -----------        --------
              |                       |                                                                  ^
              |_______________________|__________________________________________________________________|
                                (mark by tracker when road_objects's max age > threshold)
        """
        pass

    def draw(self, drawing: np.ndarray, **kwargs) -> np.ndarray:
        """Draw the object into the `drawing`.

        Args:
            drawing (np.ndarray):
                Drawing canvas.
        """
        if self.moi_id is not None:
            color = AppleRGB.values()[self.moi_id]
        else:
            color = self.label_by_majority["color"]
            
        if self.is_confirmed:
            # BaseObject.draw(self, drawing=drawing, label=False, **kwargs)
            self.draw_trajectory(drawing=drawing)
        elif self.is_counting:
            # BaseObject.draw(self, drawing=drawing, label=True, **kwargs)
            self.draw_trajectory(drawing=drawing)
        elif self.is_counted:
            Object.draw(self, drawing=drawing, label=True, color=color, **kwargs)
            self.draw_trajectory(drawing=drawing)
        elif self.is_exiting:
            Object.draw(self, drawing=drawing, label=True, color=color, **kwargs)
            self.draw_trajectory(drawing=drawing)

    def draw_trajectory(self, drawing: np.ndarray) -> np.ndarray:
        if self.moi_id is not None:
            color = AppleRGB.values()[self.moi_id]
        else:
            color = self.label_by_majority["color"]
            
        if self.trajectory is not None:
            pts = self.trajectory.reshape((-1, 1, 2))
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

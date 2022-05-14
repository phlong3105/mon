#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Data class to store tracks.
"""

from __future__ import annotations

import enum
import uuid
from abc import abstractmethod
from collections import Counter
from timeit import default_timer as timer
from typing import Optional
from typing import Union

import cv2
import numpy as np

from one.core import Color
from one.data import Detection
from one.data import majority_voting
from one.imgproc import AppleRGB
from one.imgproc import euclidean_distance
from one.imgproc import get_box_corners_points
from one.vision.object_tracking.motion import Motion

__all__ = [
	"Track",
	"TrackState",
]


# MARK: - Modules

class TrackState(enum.Enum):
    """Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.
    """
    Tentative   = 1  # Not certain or fixed.
    Confirmed   = 2  # Confirmed the Detection is a road_objects eligible for counting.
    Counting    = 3  # Object is in the counting zone/counting state.
    ToBeCounted = 4  # Mark object to be counted somewhere in this loop iteration.
    Counted     = 5  # Mark object has been counted.
    Deleted     = 6  # Mark object for exiting the ROI or image frame. Let's it die by itself.

    @staticmethod
    def values() -> list[int]:
        """Return the list of all values."""
        return [s.value for s in TrackState]

    @staticmethod
    def keys():
        """Return the list of all enum keys."""
        return [s for s in TrackState]


class Track:
    """Single target track. It contains both track's info and track management
    scheme.
    
    Requires:
        It is required to be subclassed with the motion model. In case you want
        to use it without tracking or counting functions.
        
    Attributes:
    	motion (Motion):
            Motion model.
        track_id (int, str):
            Unique track identifier.
        moi_id (int, str, optional):
            Unique MOI identifier that the current moving object is best fitted to.
            Default: `None`.
        track_state (TrackState):
            Current state of the moving object. Default: `Tentative`.
        timestamp (float):
            Time when the object is created.
        frame_index (int):
            Frame index when the object is created.
        detections (list):
            List of all detections of this object.
        trajectory (np.ndarray):
            Trajectory as an array of detections' center points.
    """
    
    # MARK: Class Attributes

    min_entering_distance: float = 0.0   # Min distance when an object enters the ROI to be `Confirmed`. Default: `0.0`.
    min_traveled_distance: float = 10.0  # Min distance between first trajectory point with last trajectory point. Default: `10.0`.
    min_hit_streak       : int   = 10    # Min number of `consecutive` frame has that track appear. Default: `10`.
    max_age              : int   = 1     # Max frame to wait until a dead track can be counted. Default: `1`.

    # MARK: Magic Functions

    def __init__(
        self,
	    motion     : Motion,
        track_id   : Union[int, str]                  = uuid.uuid4().int,
	    moi_id     : Optional[Union[int, str]]        = None,
	    track_state: TrackState                       = TrackState.Tentative,
        timestamp  : float 					   	      = timer(),
        frame_index: int                              = -1,
	    detection  : Optional[Union[list, Detection]] = None,
        *args, **kwargs
    ):
        super().__init__()
        self.motion      = motion
        self.track_id    = track_id
        self.moi_id      = moi_id
        self.track_state = track_state
        self.timestamp   = timestamp
        self.frame_index = frame_index
      
        self.detections: list[Detection] = []
        if isinstance(detection, Detection):
            self.detections = [detection]
        else:
            self.detections = (detection if (detection is not None) else [])
        
        self.trajectory = np.array([self.current_box_center])
		
    # MARK: Properties

    @property
    def age(self) -> int:
        """Return the number of frame while the track is alive, from Tentative -> Deleted."""
        return self.motion.age
    
    @property
    def class_labels(self) -> list:
        """Get the list of all class_labels of the object."""
        return [d.class_label for d in self.detections]

    @property
    def current(self) -> Detection:
        """Get the latest Detection of the object."""
        return self.detections[-1]
    
    @property
    def current_box(self) -> np.ndarray:
        """Get the latest det_box of the object."""
        return self.current.box
    
    @property
    def current_box_center(self) -> np.ndarray:
        """Get the latest center of the object."""
        return self.current.box_center
    
    @property
    def current_box_corners_points(self) -> np.ndarray:
        """Get the latest corners of bounding boxes as points."""
        return get_box_corners_points(self.current.box)

    @property
    def current_confidence(self) -> float:
        """Get the latest confidence score."""
        return self.current.confidence

    @property
    def current_class_label(self) -> dict:
        """Get the latest label of the object."""
        return self.current.class_label

    @property
    def current_frame_index(self) -> int:
        """Get the latest frame index of the object."""
        return self.current.frame_index

    @property
    def current_polygon(self) -> np.ndarray:
        """Get the latest polygon of the object."""
        return self.current.polygon

    @property
    def current_roi_id(self) -> Union[int, str]:
        """Get the latest ROI's id of the object."""
        return self.current.roi_id

    @property
    def current_timestamp(self) -> float:
        """Get the last time the object has been updated."""
        return self.current.timestamp

    @property
    def first_box(self) -> np.ndarray:
        """Get the first det_box of the object."""
        return self.detections[0].box

    @property
    def hits(self) -> int:
        """Return the number of frame has that track appear."""
        return self.motion.hits

    @property
    def hit_streak(self) -> int:
        """Return the number of `consecutive` frame has that track appear."""
        return self.motion.hit_streak
    
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
        return [d.roi_id for d in self.detections]

    @property
    def time_since_update(self) -> int:
        """Return the number of `consecutive` frame that track disappears."""
        return self.motion.time_since_update
    
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
    
    # MARK: Properties (TrackState)

    @property
    def track_state(self) -> TrackState:
        """Return track state."""
        return self._track_state

    @track_state.setter
    def track_state(self, track_state: TrackState):
        """Assign track state.

        Args:
            track_state (TrackState):
                Object's moving state.
        """
        if track_state not in TrackState.keys():
            raise ValueError(f"Moving state should be one of: "
                             f"{TrackState.keys()}. But given {track_state}.")
        self._track_state = track_state

    @property
    def is_tentative(self) -> bool:
        """Return `True` if the current moving state is `Tentative`."""
        return self.track_state == TrackState.Tentative

    @property
    def is_confirmed(self) -> bool:
        """Return `True` if the current moving state is `Confirmed`."""
        return self.track_state == TrackState.Confirmed

    @property
    def is_counting(self) -> bool:
        """Return `True` if the current moving state is `Counting`."""
        return self.track_state == TrackState.Counting

    @property
    def is_countable(self) -> bool:
        """Return `True` if the current vehicle is countable."""
        return True if (self.moi_id is not None) else False

    @property
    def is_to_be_counted(self) -> bool:
        """Return `True` if the current moving state is `ToBeCounted`."""
        return self.track_state == TrackState.ToBeCounted

    @property
    def is_counted(self) -> bool:
        """Return `True` if the current moving state is `Counted`."""
        return self.track_state == TrackState.Counted

    @property
    def is_deleted(self) -> bool:
        """Return `True` if the current moving state is `Deleted`."""
        return self.track_state == TrackState.Deleted
    
    # MARK: Update
    
    def update(self, detection: Optional[Detection], **kwargs):
        """Update with new measurement.
        
        Args:
            detection (Detection, optional):
                Object measurement.
        """
        self.detections.append(detection)
        self.motion.update(measurement=detection)
        self.update_track_state(**kwargs)
    
    @abstractmethod
    def update_trajectory(self):
        """Update trajectory with measurement's center point."""
        pass
    
    @abstractmethod
    def update_track_state(self, **kwargs):
        """Update the current state of the track."""
        pass
    
    # MARK: Visualize

    def draw(self, drawing: np.ndarray, index: int = -1, **kwargs) -> np.ndarray:
        """Draw the object into the `drawing`.

        Args:
            drawing (np.ndarray):
                Drawing canvas.
            index (int):
                Detection index. Default: `-1` means the current one.
        """
        if self.moi_id is not None:
            color = AppleRGB.values()[self.moi_id]
        else:
            color = self.label_by_majority["color"]
        
        if self.is_tentative:
            self.draw_detection(drawing=drawing, index=index, label=False, color=AppleRGB.WHITE.value, **kwargs)
        elif self.is_confirmed:
            self.draw_detection(drawing=drawing, index=index, label=False, **kwargs)
            self.draw_trajectory(drawing=drawing, index=index)
        elif self.is_counting:
            self.draw_detection(drawing=drawing, index=index, label=True, **kwargs)
            self.draw_trajectory(drawing=drawing, index=index)
        elif self.is_counted:
            self.draw_detection(drawing=drawing, index=index, label=True, color=color, **kwargs)
            self.draw_trajectory(drawing=drawing, index=index)
        elif self.is_deleted:
            self.draw_detection(drawing=drawing, index=index, label=True, color=color, **kwargs)
            self.draw_trajectory(drawing=drawing, index=index)
        
        return drawing
        
    def draw_detection(
        self,
        drawing: np.ndarray,
        index  : int             = -1,
        box    : bool            = True,
        polygon: bool            = False,
        label  : bool            = True,
        color  : Optional[Color] = None,
    ) -> np.ndarray:
        """Draw the object into the `drawing`.
        
        Args:
            drawing (np.ndarray):
                Drawing canvas.
            index (int):
                Detection index. Default: `-1` means the current one.
            box (bool):
                Should draw the detected box? Default: `True`.
            polygon (bool):
                Should draw polygon? Default: `False`.
            label (bool):
                Should draw label? Default: `True`.
            color (tuple):
                Primary color. Default: `None`.
        
        Returns:
            drawing (np.ndarray):
                Drawing canvas.
        """
        color     = (color if (color is not None) else self.label_by_majority["color"])
        detection = self.detections[index]
        
        if box:
            cv2.rectangle(
                img       = drawing,
                pt1       = (detection.box[0], detection.box[1]),
                pt2       = (detection.box[2], detection.box[3]),
                color     = color,
                thickness = 2
            )
            box_center = detection.box_center.astype(int)
            cv2.circle(
                img       = drawing,
                center    = tuple(box_center),
                radius    = 3,
                thickness = -1,
                color     = color
            )
            
        if polygon:
            pts = detection.polygon.reshape((-1, 1, 2))
            cv2.polylines(img=drawing, pts=pts, isClosed=True, color=color, thickness=2)
        
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
        
        return drawing
        
    def draw_trajectory(self, drawing: np.ndarray, index: int = -1) -> np.ndarray:
        """Draw the object's trajectory into the `drawing`.
        
        Args:
            drawing (np.ndarray):
                Drawing canvas.
           index (int):
                Detection index. Default: `-1` means the current one.
                
        Returns:
            drawing (np.ndarray):
                Drawing canvas.
        """
        if self.moi_id is not None:
            color = AppleRGB.values()[self.moi_id]
        else:
            color = self.label_by_majority["color"]
            
        if self.trajectory is not None:
            trajectory = self.trajectory[0:index]
            pts        = trajectory.reshape((-1, 1, 2))
            cv2.polylines(
                img       = drawing,
                pts       = [pts.astype(int)],
                isClosed  = False,
                color     = color,
                thickness = 2
            )
            for point in trajectory:
                cv2.circle(
                    img       = drawing,
                    center    = tuple(point.astype(int)),
                    radius    = 3,
                    thickness = 2,
                    color     = color
                )
        
        return drawing

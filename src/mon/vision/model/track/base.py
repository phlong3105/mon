#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the base class for all trackers."""

from __future__ import annotations

__all__ = [
    "Instance", "MovingObject", "Object", "StaticObject", "Tracker",
]

import uuid
from abc import ABC, abstractmethod
from collections import Counter
from timeit import default_timer as timer
from typing import Any, Callable

import cv2
import numpy as np

from mon.globals import AppleRGB, MOTIONS, MovingState, OBJECTS
from mon.vision import drawing, geometry
from mon.coreml import data as md
from mon.vision.model.track import motion as mm


# region Instance

class Instance:
    """An instance of a track (i.e, moving object) in a given frame. This class
    is mainly used to wrap and pas data between detectors and trackers.
    
    Attributes:
        id_: A unique ID. Defaults to None.
        roi_id: The unique ID of the ROI containing the
        bbox: A bounding bbox in XYXY format.
        polygon: A list of points representing an instance mask. Defaults to
            None.
        confidence: A confidence score. Defaults to None.
        classlabel: A :class:`mon.Classlabel` object. Defaults to None.
        frame_index: The current frame index. Defaults to None.
        timestamp: The creating time of the current instance.
    """
    
    def __init__(
        self,
        id_        : int | str         = uuid.uuid4().int,
        roi_uid    : int | str  | None = None,
        bbox       : np.ndarray | None = None,
        polygon    : np.ndarray | None = None,
        confidence : float      | None = None,
        classlabel : dict       | None = None,
        frame_index: int        | None = None,
        timestamp  : float             = timer(),
    ):
        self.id_         = id_
        self.roi_id      = roi_uid
        self.bbox        = np.array(bbox)    if bbox    is not None else None
        self.polygon     = np.array(polygon) if polygon is not None else None
        self.confidence  = confidence
        self.classlabel  = classlabel
        self.frame_index = frame_index
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
    def box_center(self):
        return geometry.get_bbox_center(bbox=self.bbox)
    
    @property
    def box_tl(self):
        """The bbox's top left corner."""
        return self.bbox[0:2]
    
    @property
    def box_corners_points(self) -> np.ndarray:
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
            org = (self.box_tl[0] + 5, self.box_tl[1])
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


# region Track

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
        motion      : mm.Motion,
        id_         : int | str         = uuid.uuid4().int,
        moving_state: MovingState | str = MovingState.CANDIDATE,
        moi_id      : int | str | None  = None,
        timestamp   : float             = timer(),
        frame_index : int               = -1,
    ):
        instances = [instances] if not isinstance(instances, list | tuple) else instances
        super().__init__(Instance.from_value(i) for i in instances)
        self.uid          = id_
        self.motion       = motion
        self.moving_state = moving_state
        self.moi_uid      = moi_id
        self.timestamp    = timestamp
        self.frame_index  = frame_index
        self.trajectory   = [self.current.box_center]
    
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
        return md.majority_voting(labels=self.classlabels)
    
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
    def motion(self) -> mm.Motion:
        return self._motion

    @motion.setter
    def motion(self, motion: mm.Motion):
        if isinstance(motion, mm.Motion):
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
        d = geometry.distance.euclidean(u=self.trajectory[-1], v=self.current.box_center)
        if d >= self.min_traveled_distance:
            self.trajectory.append(list(self.current.box_center))
            
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
        if self.moi_uid is not None:
            color = AppleRGB.values()[self.moi_uid]
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
            image = drawing.draw_trajectory(
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
            image = drawing.draw_trajectory(
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
            image = drawing.draw_trajectory(
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
            image = drawing.draw_trajectory(
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
            text       = f"{curr_label['id']} {curr_label['name']}"
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

# endregion


# region Tracker

class Tracker(ABC):
    """The base class for all trackers."""
    
    def __init__(
        self,
        max_age      : int       = 1,
        min_hits     : int       = 3,
        iou_threshold: float     = 0.3,
        motion_type  : mm.Motion = "kf_box_motion",
        object_type  : Callable  = MovingObject,
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
    def motion_type(self) -> type(mm.Motion):
        return self._motion_type
    
    @motion_type.setter
    def motion_type(self, motion_type: Any):
        if isinstance(motion_type, str):
            motion_type = MOTIONS.get(motion_type)
        elif isinstance(motion_type, dict):
            if not hasattr(motion_type, "name"):
                raise ValueError(f"motion_type must contain a key 'name'.")
            motion_type = MOTIONS.get(motion_type["name"]).__class__
        elif isinstance(motion_type, mm.Motion):
            motion_type = motion_type.__class__
        self._motion_type = motion_type
    
    @property
    def object_type(self) -> type(MovingObject):
        return self._object_type
    
    @object_type.setter
    def object_type(self, object_type: Any):
        if isinstance(object_type, str):
            object_type = OBJECTS.get(object_type)
        elif isinstance(object_type, dict):
            if not hasattr(object_type, "name"):
                raise ValueError(f"object_type must contain a key 'name'.")
            object_type = OBJECTS.get(object_type["name"]).__class__
        elif isinstance(object_type, MovingObject):
            object_type = object_type.__class__
        self._object_type = object_type
        
    @abstractmethod
    def update(self, instances: list | np.ndarray = ()):
        """Update :attr:`tracks` with new detections. This method will call the
        following methods:
            1. :meth:`assign_instances_to_tracks`
            2. :meth:`update_matched_tracks`
            3. :meth:`create_new_tracks`
            4. :meth`:delete_dead_tracks`

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

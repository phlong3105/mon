#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements a Region of Interest and Movement of Interest in
images.
"""

from __future__ import annotations

__all__ = [
    "MOI", "MovementOfInterest", "ROI", "RegionOfInterest",
    "assign_detections_to_rois", "assign_moving_objects_to_mois",
    "get_best_matched_moi", "get_moi_for_box", "get_roi_for_box",
]

from typing import Any, Sequence

import cv2
import numpy as np

import mon
from supr import data

error_console = mon.error_console


# region ROI

class RegionOfInterest:
    """The Region of Interest.
    
    Args:
        id_: An unique ID. Defaults to None.
        points: A list of points defining the ROI. Defaults to None.
        shape_type: The ROI type. Defaults to None.
    """
    
    def __init__(self, id_: int | str, points: np.ndarray, shape_type: str):
        super().__init__()
        self.id_        = id_
        self.points     = points
        self.shape_type = shape_type
    
    def __len__(self) -> int:
        return int(self._points.shape[0])
    
    @property
    def points(self) -> np.ndarray:
        """The array of points defining the ROI."""
        return self._points
    
    @points.setter
    def points(self, points: np.ndarray | list):
        points = np.array(points, np.int32)
        if not points.ndim == 2:
            raise ValueError(
                f"points' number of dimensions must be == 2, but got {points.ndim}."
            )
        self._points = points
    
    @property
    def has_valid_points(self) -> bool:
        """Return True if there are more than 3 points."""
        if int(self._points.shape[0]) >= 3:
            return True
        else:
            error_console.log(f"Number of points in the ROI must be >= 3.")
            return False
    
    @classmethod
    def from_dict(cls, value: dict) -> list[ROI]:
        """Create a list of :class:`RegionOfInterest` from a dictionary."""
        if "roi" not in value:
            raise ValueError("value must contains a 'roi' key.")
        value = value["roi"]
        if not isinstance(value, list | tuple):
            raise TypeError(
                f"value must be a list or tuple, but got {type(value)}."
            )
        return [cls(**v) for v in value]
    
    @classmethod
    def from_file(cls, value: mon.Path) -> list[ROI]:
        """Create a list of :class:`RegionOfInterest` from the content of a
        ".json" file specified by the :param:`path`.
        """
        value = mon.Path(value)
        if not value.is_json_file():
            raise ValueError(
                f"path must be a valid path to a .json file, but got {value}."
            )
        data = mon.load_config(value)
        return cls.from_dict(value=data)

    @classmethod
    def from_value(cls, value: Any) -> list[ROI] | None:
        """Create a :class:`RegionOfInterest` object from an arbitrary
        :param:`value`."""
        if isinstance(value, ROI):
            return [value]
        if isinstance(value, dict):
            return cls.from_dict(value=value)
        if isinstance(value, list | tuple):
            assert all(isinstance(v, dict | ROI) for v in value)
            return [cls(**v) if isinstance(v, dict) else v for v in value]
        if isinstance(value, str | mon.Path):
            return cls.from_file(value=value)
        return None

    def is_box_in_roi(self, bbox: np.ndarray, compute_distance: bool = False) -> int:
        """Check a bounding bbox touches the current ROI.
        
        Args:
            bbox: Bounding boxes in XYXY format.
            compute_distance: If True, calculate the distance from bbox
                coordinates to the ROI? Defaults to False.
        
        Returns:
            Positive (inside), negative (outside), or zero (on an edge).
        """
        tl = cv2.pointPolygonTest(self.points, (bbox[0], bbox[1]), compute_distance)
        tr = cv2.pointPolygonTest(self.points, (bbox[2], bbox[1]), compute_distance)
        br = cv2.pointPolygonTest(self.points, (bbox[2], bbox[3]), compute_distance)
        bl = cv2.pointPolygonTest(self.points, (bbox[0], bbox[3]), compute_distance)
        if tl > 0 and tr > 0 and br > 0 and bl > 0:
            return 1
        elif tl < 0 and tr < 0 and br < 0 and bl < 0:
            return min(tl, tr, br, bl)
        else:
            return 0

    def is_box_center_in_roi(self, bbox: np.ndarray, compute_distance: bool = False) -> int:
        """Check a bounding bbox touches the current ROI.
        
        Args:
            bbox: Bounding boxes in XYXY format.
            compute_distance: If True, calculate the distance from bbox
                coordinates to the ROI? Defaults to False.
        
        Returns:
            Positive (inside), negative (outside), or zero (on an edge).
        """
        c_x = (bbox[0] + bbox[2]) / 2
        c_y = (bbox[1] + bbox[3]) / 2
        return int(cv2.pointPolygonTest(self.points, (c_x, c_y), compute_distance))
    
    def draw(self, image: np.ndarray) -> np.ndarray:
        """Draw the current ROI on the :param:`image`."""
        color = mon.BasicRGB.GREEN.value
        pts = self.points.reshape((-1, 1, 2))
        cv2.polylines(
            img       = image,
            pts       = [pts],
            isClosed  = True,
            color     = color,
            thickness = 2
        )
        return image


ROI = RegionOfInterest


def get_roi_for_box(
    bbox     : np.ndarray,
    rois     : Sequence[ROI],
    threshold: int = -50
) -> int | str | None:
    """Check if bounding boxes belong to one of the many ROIs in the image.

    Args:
        bbox: Bounding boxes in XYXY format.
        rois: A list of :class:`ROI` objects.
        threshold: A threshold value determining whether if the bbox is in a
            ROI. Defaults to -50.
        
    Returns:
        The matching ROI's id.
    """
    for r in rois:
        d = r.is_box_center_in_roi(bbox=bbox, compute_distance=True)
        if d >= threshold:
            return r.id_
    return None


def assign_detections_to_rois(instances: list[data.Instance], rois: list[ROI]):
    """Assign :class:`data.Detection` objects to ROIs.

    Args:
        instances: A list :class:`data.Instance` objects.
        rois: A list of :class:`ROI` objects.
    """
    for i in instances:
        i.roi_id = get_roi_for_box(bbox=i.bbox, rois=rois)


# endregion


# region MOI

class MovementOfInterest:
    """The Movement of Interest
    
    Args:
        id_: An unique ID. Defaults to None.
        points: A list of points defining the MOI. Defaults to None.
        shape_type: The MOI type. Defaults to None.
        offset: Defaults to None.
        distance_function: A distance function. Defaults to 'hausdorff'.
        distance_threshold: The maximum distance for counting a track. Defaults
            to 300.0.
        angle_threshold: The maximum angle for counting a track. Defaults to
            45.0.
        color: The color of the MOI. Defaults to [255, 255, 255].
    """
    
    def __init__(
        self,
        id_               : int | str,
        points            : np.ndarray,
        shape_type        : str,
        offset            : int | None = None,
        distance_function : str        = "hausdorff",
        distance_threshold: float      = 300.0,
        angle_threshold   : float      = 45.0,
        color             : list[int]  = mon.AppleRGB.WHITE.value,
    ):
        super().__init__()
        self.id_                = id_
        self.points             = points
        self.shape_type         = shape_type
        self.offset             = offset
        self.distance_function  = distance_function  # mon.DISTANCES.build(name=distance_function)
        self.distance_threshold = distance_threshold
        self.angle_threshold    = angle_threshold
        
        self.color = mon.AppleRGB.values()[id_] \
            if isinstance(id_, int) and id_ < len(mon.AppleRGB.values()) \
            else color
    
    @property
    def points(self) -> np.ndarray:
        """The array of points defining the MOI."""
        return self._points
    
    @points.setter
    def points(self, points: np.ndarray | list):
        points = np.array(points, np.int32)
        """
        if not points.ndim >= 2:
            raise ValueError(
                f"points' number of dimensions must be >= 2, but got {points.ndim}."
            )
        """
        self._points = points
    
    @property
    def has_valid_points(self) -> bool:
        """Return True if there are more than 2 points."""
        if int(self._points.shape[0]) \
            and not all(len(t) >= 2 for t in self.points):
            return True
        else:
            error_console.log(f"Number of points in each track must be >= 2.")
            return False
    
    @classmethod
    def from_dict(cls, value: dict) -> list[MOI]:
        """Create a list of :class:`MotionOfInterest` from a dictionary."""
        if "moi" not in value:
            raise ValueError("value must contains a 'moi' key.")
        value = value["roi"]
        if not isinstance(value, list | tuple):
            raise TypeError(
                f"value must be a list or tuple, but got {type(value)}."
            )
        return [cls(**v) for v in value]
    
    @classmethod
    def from_file(cls, value: mon.Path) -> list[MOI]:
        """Create a list of :class:`MotionOfInterest` from the content of a
        ".json" file specified by the :param:`path`.
        """
        value = mon.Path(value)
        if not value.is_json_file():
            raise ValueError(
                f"path must be a valid path to a .json file, but got {value}."
            )
        data = mon.load_config(value)
        return cls.from_dict(value=data)

    @classmethod
    def from_value(cls, value: Any) -> list[MOI] | None:
        """Create a :class:`MotionOfInterest` object from an arbitrary
        :param:`value`."""
        if isinstance(value, MOI):
            return [value]
        if isinstance(value, dict):
            return cls.from_dict(value=value)
        if isinstance(value, list | tuple):
            return [cls(**v) if isinstance(v, dict) else v for v in value]
        if isinstance(value, str | mon.Path):
            return cls.from_file(value=value)
        return None
    
    def calculate_distance_with_track(self, object_track: np.ndarray) -> float:
        """Calculate the distance between an object's track to the MOI's tracks.
        
        Args:
            object_track: An object's trajectory as an array of points.
            
        Returns:
            Distance value between an object's track with a MOI's track. If
            distance > :attr:`distance_threshold`, return None.
        """
        d = self.distance_function(self.points, object_track)
        return None if (d > self.distance_threshold) else d
    
    def calculate_angle_with_track(self, object_track: np.ndarray) -> float:
        """Calculate the angle between an object's track to the MOI's tracks.
        
        Args:
            object_track: An object's trajectory as an array of points.
            
        Returns:
            Angle value between an object's track with a MOI's track. If
            angle > :attr:`angle_threshold`, return None.
        """
        a = mon.angle(self.points, object_track)
        return None if (a > self.angle_threshold) else a
    
    def is_box_center_in_moi(
        self,
        bbox            : np.ndarray,
        compute_distance: bool = False
    ) -> int:
        """Check a bounding bbox touches the current MOI.
        
        Args:
            bbox: Bounding boxes in XYXY format.
            compute_distance: If True, calculate the distance from bbox
                coordinates to the ROI? Defaults to False.
        
        Returns:
            Positive (inside), negative (outside), or zero (on an edge).
        """
        c_x = (bbox[0] + bbox[2]) / 2
        c_y = (bbox[1] + bbox[3]) / 2
        return int(cv2.pointPolygonTest(self.points, (c_x, c_y), compute_distance))
    
    def draw(self, image: np.ndarray) -> np.ndarray:
        """Draw the current MOI on the :param:`image`."""
        # NOTE: Draw MOI's direction
        if self.points.ndim < 2:
            return image
        
        pts = self.points.reshape((-1, 1, 2))
        if self.shape_type == "polygon":
            cv2.polylines(
                img       = image,
                pts       = [pts],
                isClosed  = True,
                color     = self.color,
                thickness = 1,
                lineType  = cv2.LINE_AA
            )
        elif self.shape_type == "line":
            cv2.polylines(
                img       = image,
                pts       = [pts],
                isClosed  = False,
                color     = self.color,
                thickness = 1,
                lineType  = cv2.LINE_AA
            )
            cv2.arrowedLine(
                img       = image,
                pt1       = tuple(self.points[-2]),
                pt2       = tuple(self.points[-1]),
                color     = self.color,
                thickness = 1,
                line_type = cv2.LINE_AA,
                tipLength = 0.2
            )
            for i in range(len(self.points) - 1):
                cv2.circle(
                    img       = image,
                    center    = tuple(self.points[i]),
                    radius    = 3,
                    color     = self.color,
                    thickness = -1,
                    lineType  = cv2.LINE_AA
                )
        
        # NOTE: Draw MOI's uid
        cv2.putText(
            img       = image,
            text      = f"{self.id_}",
            fontFace  = cv2.FONT_HERSHEY_SIMPLEX,
            fontScale = 0.75,
            org       = tuple(self.points[-1]),
            color     = self.color,
            thickness=2
        )
        return image


MOI = MovementOfInterest


def get_moi_for_box(
    bbox     : np.ndarray,
    mois     : list[MOI],
    threshold: int = 0
) -> int | str | None:
    """Check if bounding boxes belong to one of the many MOIs in the image.
    
    Args:
        bbox: Bounding boxes in XYXY format.
        mois: A list of :class:`MOI` objects.
        threshold: A threshold value determining whether if the bbox is in a
            ROI. Defaults to -50.
            
    Returns:
        The matching MOI's id.
    """
    for m in mois:
        d = m.is_box_center_in_moi(bbox=bbox)
        if d >= threshold:
            return m.id_
    return None


def get_best_matched_moi(
    object_track: np.ndarray,
    mois        : list[MOI],
) -> tuple[int, float]:
    """Find the MOI that best matched with an object's track.
    
    Args:
        object_track: An object's track as an array of points.
        mois: A list of :class:`MOI` objects.

    Returns:
        The best matching MOI's uid.
        The best matching MOI's distance.
    """
    # NOTE: Calculate distances between object track and all MOIs' tracks
    distances = []
    angles    = []
    for moi in mois:
        distances.append(
            moi.calculate_distance_with_track(object_track=object_track)
        )
        angles.append(
            moi.calculate_angle_with_track(object_track=object_track)
        )
    
    min_moi_id, min_distance = None, None
    for i, (d, a) in enumerate(zip(distances, angles)):
        if (d is None) or (a is None):
            continue
        if (min_distance is not None) and (min_distance < d):
            continue
        min_distance = d
        min_moi_id = mois[i].id_
    
    return min_moi_id, min_distance


def assign_moving_objects_to_mois(
    objects   : list,
    mois      : list,
    shape_type: str  = "line",
):
    """Assign :class:`Detection` objects to MOIs.

    Args:
        objects: A list of objects.
        mois: A list of :class:`MOI` objects.
        shape_type: The shape of MOI to check. One of: ['polygon', 'line'].
            Defaults to 'line'.
    """
    if len(objects) <= 0:
        return
    polygon_mois = [m for m in mois if m.shape_type == "polygon"]
    line_mois    = [m for m in mois if m.shape_type == "line"]
    
    if shape_type == "polygon":
        for o in objects:
            if o.moi_uid is None:
                o.moi_uid = get_moi_for_box(
                    bbox = o.current_box,
                    mois = polygon_mois
                )
    elif shape_type == "line":
        for o in objects:
            if o.moi_uid is None:
                o.moi_uid = get_best_matched_moi(
                    object_track = o.trajectory,
                    mois         = line_mois
                )[0]

# endregion

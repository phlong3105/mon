#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements a Region of Interest and Movement of Interest in
images.
"""

from __future__ import annotations

__all__ = [
	"MOI", "MovementOfInterest", "ROI", "RegionOfInterest",
	"assign_detections_to_rois", "assign_moving_objects_to_mois",
	"find_best_matched_moi", "find_moi_for_box", "find_roi_for_box",
]

from typing import Sequence

import cv2
import numpy as np

import mon
from supr import constant, data
from supr.typing import Ints, PathsType, PathType, PointsType, UIDType


# region MARK: - ROI

class RegionOfInterest:
	"""The Region of Interest.
	
	Args:
		uid: An unique ID. Defaults to None.
		points: A list of points defining the ROI. Defaults to None.
		shape_type: The ROI type. Defaults to None.
	"""
	
	def __init__(self, uid: UIDType, points: PointsType, shape_type: str):
		super().__init__()
		self.uid        = uid
		self.points	    = points
		self.shape_type = shape_type
	
	def __len__(self) -> int:
		return int(self._points.shape[0])
	
	@property
	def points(self) -> np.ndarray:
		"""The array of points defining the ROI."""
		return self._points
	
	@points.setter
	def points(self, points: PointsType):
		if isinstance(points, list | tuple):
			points = np.array(points, np.int32)
		assert isinstance(points, np.ndarray) and points.ndim == 2
		self._points = points

	@property
	def are_valid_points(self) -> bool:
		"""Return True if there are more than 3 points."""
		if int(self._points.shape[0]) >= 3:
			return True
		else:
			mon.error_console.log(f"Number of points in the ROI must be >= 3.")
			return False
		
	@classmethod
	def from_file(
		cls,
		path     : PathType,
		rmois_dir: PathsType = None,
		**kwargs
	) -> list:
		"""Create a list of :class:`RegionOfInterest` from the content of a
		".json" file specified by the :param:`path`.
		"""
		if mon.is_json_file(path=path):
			pass
		elif isinstance(path, str) \
			and rmois_dir is not None \
			and mon.Path(rmois_dir).is_dir():
			path = mon.Path(rmois_dir) / path
		assert mon.is_json_file(path=path)
		
		data = mon.load_config(path)
		assert hasattr(data, "roi")
		rois = [cls(**r, **kwargs) for r in data.roi]
		return rois
	
	def is_box_in_roi(
		self,
		box             : np.ndarray,
		compute_distance: bool = False
	) -> int:
		"""Check a bounding box touches the current ROI.
		
		Args:
			box: Bounding boxes in (x1, y1, x2, y2) format.
			compute_distance: If True, calculate the distance from box
				coordinates to the ROI? Defaults to False.
		
		Returns:
			Positive (inside), negative (outside), or zero (on an edge).
		"""
		tl = cv2.pointPolygonTest(self.points, (box[0], box[1]), compute_distance)
		tr = cv2.pointPolygonTest(self.points, (box[2], box[1]), compute_distance)
		br = cv2.pointPolygonTest(self.points, (box[2], box[3]), compute_distance)
		bl = cv2.pointPolygonTest(self.points, (box[0], box[3]), compute_distance)
		if tl > 0 and tr > 0 and br > 0 and bl > 0:
			return 1
		elif tl < 0 and tr < 0 and br < 0 and bl < 0:
			return min(tl, tr, br, bl)
		else:
			return 0
	
	def is_box_center_in_roi(
		self,
		box             : np.ndarray,
		compute_distance: bool = False
	) -> int:
		"""Check a bounding box touches the current ROI.
		
		Args:
			box: Bounding boxes in (x1, y1, x2, y2) format.
			compute_distance: If True, calculate the distance from box
				coordinates to the ROI? Defaults to False.
		
		Returns:
			Positive (inside), negative (outside), or zero (on an edge).
		"""
		c_x = (box[0] + box[2]) / 2
		c_y = (box[1] + box[3]) / 2
		return int(cv2.pointPolygonTest(self.points, (c_x, c_y), compute_distance))
		
	def draw(self, drawing: np.ndarray) -> np.ndarray:
		"""Draw the current ROI on the :param:`drawing`."""
		color = mon.BasicRGB.GREEN.value
		pts   = self.points.reshape((-1, 1, 2))
		cv2.polylines(img=drawing, pts=[pts], isClosed=True, color=color, thickness=2)
		return drawing


ROI = RegionOfInterest


def find_roi_for_box(
	box      : np.ndarray,
	rois     : Sequence[ROI],
	threshold: int = -50
) -> UIDType | None:
	"""Check if bounding boxes belong to one of the many ROIs in the image.

	Args:
		box: Bounding boxes in (x1, y1, x2, y2) format.
		rois: A list of :class:`ROI` objects.
		threshold: A threshold value determining whether if the box is in a ROI.
			Defaults to -50.
		
	Returns:
		The matching ROI's UID.
	"""
	for r in rois:
		d = r.is_box_center_in_roi(box=box, compute_distance=True)
		if d >= threshold:
			return r.uid
	return None


def assign_detections_to_rois(dets: list[data.Instance], rois: Sequence[ROI]):
	"""Assign :class:`data.Detection` objects to ROIs.

	Args:
		dets: A list :class:`data.Detection` objects.
		rois: A list of :class:`ROI` objects.
	"""
	for d in dets:
		d.roi_uid = find_roi_for_box(box=d.box, rois=rois)

# endregion


# region MARK: - MOI

class MovementOfInterest:
	"""The Movement of Interest
	
	Args:
		uid: An unique ID. Defaults to None.
		points: A list of points defining the MOI. Defaults to None.
		shape_type: The MOI type. Defaults to None.
		offset: Defaults to None.
		distance_function: A distance function. Defaults to 'hausdorff'.
		distance_threshold: The maximum distance for counting with track.
			Defaults to 300.0.
		angle_threshold: The maximum angle for counting with track. Defaults to
			45.0.
		color: The color for drawing the MOI. Defaults to (255, 255, 255).
	"""
	
	def __init__(
		self,
		uid               : UIDType,
		points            : PointsType,
		shape_type        : str,
		offset            : int | None = None,
		distance_function : str        = "hausdorff",
		distance_threshold: float      = 300.0,
		angle_threshold   : float      = 45.0,
		color             : Ints       = mon.AppleRGB.WHITE.value,
	):
		super().__init__()
		self.uid                = uid
		self.points             = points
		self.shape_type         = shape_type
		self.offset             = offset
		self.distance_function  = constant.DISTANCE.build(name=distance_function)
		self.distance_threshold = distance_threshold
		self.angle_threshold    = angle_threshold
		
		self.color = mon.AppleRGB.values()[uid] \
			if isinstance(uid, int) and uid < len(mon.AppleRGB.values()) \
			else color
			
	@property
	def points(self) -> np.ndarray:
		"""The array of points defining the MOI."""
		return self._points
	
	@points.setter
	def points(self, points: PointsType):
		if isinstance(points, list | tuple):
			points = np.array(points, np.int32)
		assert isinstance(points, np.ndarray) and points.ndim >= 2
		self._points = points

	@property
	def are_valid_points(self) -> bool:
		"""Return True if there are more than 2 points."""
		if int(self._points.shape[0]) \
			and not all(len(t) >= 2 for t in self.points):
			return True
		else:
			mon.error_console.log(
				f"Number of points in each track must be >= 2."
			)
			return False
	
	@classmethod
	def from_file(
		cls,
		path     : PathType,
		rmois_dir: PathsType = None,
		**kwargs
	) -> list:
		"""Create a list of :class:`MotionOfInterest` from the content of a
		".json" file specified by the :param:`path`.
		"""
		if mon.is_json_file(path=path):
			pass
		elif isinstance(path, str) \
			and rmois_dir is not None \
			and mon.Path(rmois_dir).is_dir():
			path = mon.Path(rmois_dir) / path
		assert mon.is_json_file(path=path)

		data = mon.load_config(path)
		assert hasattr(data, "moi")
		mois = [cls(**m, **kwargs) for m in data.moi]
		return mois
	
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
		a = mon.angle_between_vectors(self.points, object_track)
		return None if (a > self.angle_threshold) else a
	
	def is_box_center_in_moi(
		self,
		box             : np.ndarray,
		compute_distance: bool = False
	) -> int:
		"""Check a bounding box touches the current MOI.
		
		Args:
			box: Bounding boxes in (x1, y1, x2, y2) format.
			compute_distance: If True, calculate the distance from box
				coordinates to the ROI? Defaults to False.
		
		Returns:
			Positive (inside), negative (outside), or zero (on an edge).
		"""
		c_x = (box[0] + box[2]) / 2
		c_y = (box[1] + box[3]) / 2
		return int(cv2.pointPolygonTest(self.points, (c_x, c_y), compute_distance))
		
	def draw(self, drawing: np.ndarray) -> np.ndarray:
		"""Draw the current MOI on the :param:`drawing`."""
		# NOTE: Draw MOI's direction
		pts = self.points.reshape((-1, 1, 2))
		if self.shape_type == "polygon":
			cv2.polylines(
				img       = drawing,
				pts       = [pts],
				isClosed  = True,
				color     = self.color,
				thickness = 1,
				lineType  = cv2.LINE_AA
			)
		elif self.shape_type == "line":
			cv2.polylines(
				img       = drawing,
				pts       = [pts],
				isClosed  = False,
				color     = self.color,
				thickness = 1,
				lineType  = cv2.LINE_AA
			)
			cv2.arrowedLine(
				img       = drawing,
				pt1       = tuple(self.points[-2]),
				pt2       = tuple(self.points[-1]),
				color     = self.color,
				thickness = 1,
				line_type = cv2.LINE_AA,
				tipLength = 0.2
			)
			for i in range(len(self.points) - 1):
				cv2.circle(
					img       = drawing,
					center    = tuple(self.points[i]),
					radius    = 3,
					color     = self.color,
					thickness = -1,
					lineType  = cv2.LINE_AA
				)
				
		# NOTE: Draw MOI's uid
		cv2.putText(
			img       = drawing,
			text      = f"{self.uid}",
			fontFace  = cv2.FONT_HERSHEY_SIMPLEX,
			fontScale = 0.75,
			org       = tuple(self.points[-1]),
			color     = self.color,
			thickness = 2
		)
		return drawing


MOI = MovementOfInterest


def find_moi_for_box(
	box      : np.ndarray,
	mois     : Sequence[MOI],
	threshold: int = 0
) -> UIDType | None:
	"""Check if bounding boxes belong to one of the many MOIs in the image.
	
	Args:
		box: Bounding boxes in (x1, y1, x2, y2) format.
		mois: A list of :class:`MOI` objects.
		threshold: A threshold value determining whether if the box is in a ROI.
			Defaults to -50.
			
	Returns:
		The matching MOI's UID.
	"""
	for m in mois:
		d = m.is_box_center_in_moi(box=box)
		if d >= threshold:
			return m.uid
	return None


def find_best_matched_moi(
	object_track: np.ndarray,
	mois        : Sequence[MOI],
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
		min_moi_id   = mois[i].uid

	return min_moi_id, min_distance


def assign_moving_objects_to_mois(
	objs      : list,
	mois      : list,
	shape_type: str = "line",
):
	"""Assign :class:`Detection` objects to MOIs.

	Args:
		objs: A list of objects.
		mois: A list of :class:`MOI` objects.
		shape_type: The shape of MOI to check. One of: ["polygon", "line"].
			Defaults to 'line'.
	"""
	if len(objs) <= 0:
		return
	polygon_mois = [m for m in mois if m.shape_type == "polygon"]
	line_mois    = [m for m in mois if m.shape_type == "line"   ]
	
	if shape_type == "polygon":
		for o in objs:
			if o.moi_uid is None:
				o.moi_uid = find_moi_for_box(
					box  = o.current_box,
					mois = polygon_mois
				)
	elif shape_type == "line":
		for o in objs:
			if o.moi_uid is None:
				o.moi_uid = find_best_matched_moi(
					object_track = o.trajectory,
					mois         = line_mois
				)[0]
	
	
# endregion

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Region of interest.
"""

from __future__ import annotations

import os
from typing import Optional
from typing import Union

import cv2
import numpy as np

from one import BasicRGB
from one import error_console
from one import is_json_file
from one import load_config

__all__ = [
	"ROI",
]


# MARK: - ROI (Region of Interest)

class ROI:
	"""ROI (Region of Interest).
	
	Attributes:
		id_ (int, str, optional):
			ROI's unique id. Default: `None`.
		shape_type (str, optional):
			Shape type. Default: `None`.
	"""
	
	# MARK: Magic Functions
	
	def __init__(
		self,
		id_       : Optional[Union[int, str]]         = None,
		points    : Optional[Union[np.ndarray, list]] = None,
		shape_type: Optional[str]        		  	  = None,
		*args, **kwargs
	):
		"""

		Args:
			id_ (int, str, optional):
				ROI's unique id. Default: `None`.
			points (np.ndarray, list, optional):
				List of points in the ROI. Default: `None`.
			shape_type (str, optional):
				Shape type. Default: `None`.
		"""
		super().__init__()
		self.id_        = id_
		self.shape_type = shape_type
		self.points	    = points

	# MARK: Properties
	
	@property
	def points(self) -> Union[np.ndarray, None]:
		"""Returns the array of points in the ROI."""
		return self._points
	
	@points.setter
	def points(self, points: Union[np.ndarray, list, None]):
		"""Assign points in ROI.

		Args:
			points (np.ndarray, list, optional):
				List of points in the ROI. Default: `None`.
		"""
		if isinstance(points, list):
			points = np.array(points, np.int32)
		self._points = points

	@property
	def are_valid_points(self) -> bool:
		"""Returns `True` if the points are valid."""
		if self.points and len(self.points) >= 2:
			return True
		else:
			error_console.log(f"Insufficient number of points in the ROI.")
			return False

	# MARK: Configure
	
	@classmethod
	def load_from_file(cls, file: str, rmois_dir: Optional[str] = None, **kwargs) -> list:
		"""Load ROI from external `.json` file.
		
		Args:
			file (str):
				Give the roi file. Example a path
				"..data/<dataset>/rmois/cam_n.json", so provides `cam_n.json`.
			rmois_dir (str, optional):
				Location that stores all rmois.
			
		Returns:
			rois (list):
				Return the list of ROIs in the image.
		"""
		# NOTE: Get json file
		path = ""
		if is_json_file(path=file):
			path = file
		elif rmois_dir:
			path = os.path.join(rmois_dir, file)
		if not is_json_file(path=path):
			raise ValueError(f"File not found or given a wrong file type at {path}.")

		# NOTE: Create ROIs
		rois_data = load_config(path).roi
		rois      = []
		for roi_cfg in rois_data:
			rois.append(cls(**roi_cfg, **kwargs))
		return rois
	
	# MARK: API
	
	@staticmethod
	def associate_detections_to_rois(detections: list, rois: list):
		"""Static method to check if a given box belong to one of the many
		rois in the image.

		Args:
			detections (np.ndarray):
				Array of detections.
			rois (list):
				List of ROIs.
		"""
		for d in detections:
			d.roi_id = ROI.find_roi_for_box(box_xyxy=d.box, rois=rois)
	
	@staticmethod
	def find_roi_for_box(box_xyxy: np.ndarray, rois: list) -> Optional[int]:
		"""Static method to check if a given box belong to one of the many
		ROIs in the image.

		Args:
			box_xyxy (np.ndarray):
				Bounding boxes. They are expected to be in (x1, y1, x2, y2)
				format with `0 <= x1 < x2` and `0 <= y1 < y2`.
			rois (list):
				List of ROIs.
		
		Returns:
			roi_id (int):
				ROI's id that the object is in. Else `None`.
		"""
		for roi in rois:
			dist = roi.is_center_in_or_touch_roi(box_xyxy=box_xyxy, compute_distance=True)
			if dist >= -50:
				return roi.id_
		return None
	
	def is_box_in_or_touch_roi(self, box_xyxy: np.ndarray, compute_distance: bool = False) -> int:
		"""Check the bounding box touch ROI or not.
		
		Args:
			box_xyxy (np.ndarray):
				Bounding boxes. They are expected to be in (x1, y1, x2, y2)
				format with `0 <= x1 < x2` and `0 <= y1 < y2`.
			compute_distance (bool):
				Should calculate the distance from box coordinates to roi?
				Default: `False`.
		
		Returns:
			distance (int):
				positive (inside), negative (outside), or zero (on an edge).
		"""
		tl = cv2.pointPolygonTest(self.points, (box_xyxy[0], box_xyxy[1]), compute_distance)
		tr = cv2.pointPolygonTest(self.points, (box_xyxy[2], box_xyxy[1]), compute_distance)
		br = cv2.pointPolygonTest(self.points, (box_xyxy[2], box_xyxy[3]), compute_distance)
		bl = cv2.pointPolygonTest(self.points, (box_xyxy[0], box_xyxy[3]), compute_distance)
		
		if tl > 0 and tr > 0 and br > 0 and bl > 0:
			return 1
		elif tl < 0 and tr < 0 and br < 0 and bl < 0:
			return min(tl, tr, br, bl)
		else:
			return 0
	
	def is_center_in_or_touch_roi(self, box_xyxy: np.ndarray, compute_distance: bool = False) -> int:
		""" Check the bounding box touch ROI or not.
		
		Args:
			box_xyxy (np.ndarray):
				Bounding boxes. They are expected to be in (x1, y1, x2, y2)
				format with `0 <= x1 < x2` and `0 <= y1 < y2`.
			compute_distance (bool):
				Should calculate the distance from center to roi?
				Default: `False`.
		
		Returns:
			distance (int)
				positive (inside), negative (outside), or zero (on an edge).
		"""
		c_x = (box_xyxy[0] + box_xyxy[2]) / 2
		c_y = (box_xyxy[1] + box_xyxy[3]) / 2
		return int(cv2.pointPolygonTest(self.points, (c_x, c_y), compute_distance))
	
	# MARK: Visualize

	def draw(self, drawing: np.ndarray) -> np.ndarray:
		"""Draw the ROI.
		
		Args:
			drawing (np.ndarray):
				Drawing canvas.
		"""
		color = BasicRGB.GREEN.value
		pts   = self.points.reshape((-1, 1, 2))
		cv2.polylines(img=drawing, pts=[pts], isClosed=True, color=color, thickness=2)
		return drawing

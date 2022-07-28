#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Data class to store output from detectors.
"""

from __future__ import annotations

import uuid
from timeit import default_timer as timer
from typing import Optional
from typing import Union

import cv2
import numpy as np
from one.imgproc import box_xyxy_to_cxcyrh
from one.imgproc import get_box_center

from one.core import Color

__all__ = [
	"Detection",
]


# MARK: - Modules

class Detection:
	"""Detection converted from raw numpy output from detector.
	
	Attributes:
		detection_id (int, str):
			Unique measurement identifier.
		roi_id (int, str, optional):
			Unique ID of the ROI that the object is in. Else `None`.
			Default: `None`.
		box (np.ndarray, optional):
			Bounding box in (x1, y1, x2, y2) format. Default: `None`.
		polygon (np.ndarray, optional):
			List of points. Default: `None`.
		features (np.ndarray, optional):
			Feature vector that describes the object contained in this image.
			Default: `None`.
		confidence (float, optional):
			Confidence score. Default: `None`.
		class_label (dict, optional):
			Class-label dict. Default: `None`.
		frame_index (int, optional):
			Index of frame when the Detection is created. Default: `None`.
		timestamp (float):
			Time when the object is created.
	"""
	
	# MARK: Magic Functions
	
	def __init__(
		self,
		detection_id: Union[int, str]	        = uuid.uuid4().int,
		roi_id      : Optional[Union[int, str]] = None,
		box         : Optional[np.ndarray]      = None,
		polygon     : Optional[np.ndarray]      = None,
		features    : Optional[np.ndarray]      = None,
		confidence  : Optional[float]           = None,
		class_label : Optional[dict]            = None,
		frame_index : Optional[int]             = None,
		timestamp   : float                     = timer(),
		*args, **kwargs
	):
		super().__init__()
		self.detection_id = detection_id
		self.roi_id       = roi_id
		self.box          = box
		self.polygon      = polygon
		self.features     = features
		self.confidence   = confidence
		self.class_label  = class_label
		self.frame_index  = frame_index
		self.timestamp    = timestamp
	
	# MARK: Properties
	
	@property
	def box_cxcyrh(self):
		"""Return the box in (cx, cy, r, h) format."""
		return box_xyxy_to_cxcyrh(self.box)
	
	@property
	def box_center(self):
		"""Return the box's center."""
		return get_box_center(self.box)
	
	@property
	def box_tl(self):
		"""Return the box's top left corner."""
		return self.box[0:2]
	
	@property
	def box_br(self):
		"""Return the box's bottom right corner."""
		return self.box[2:4]
	
	# MARK: Visualize
	
	def draw(
		self,
		drawing: np.ndarray,
		box    : bool            = False,
		polygon: bool            = False,
		label  : bool            = True,
		color  : Optional[Color] = None
	) -> np.ndarray:
		"""Draw the road_objects into the `drawing`.
		
		Args:
			drawing (np.ndarray):
				Drawing canvas.
			box (bool):
				Should draw the detected boxes? Default: `False`.
			polygon (bool):
				Should draw polygon? Default: `False`.
			label (bool):
				Should draw label? Default: `True`.
			color (Color, optional):
				Primary color. Default: `None`.
				
		Returns:
			drawing (np.ndarray):
				Drawing canvas.
		"""
		color = color if (color is not None) else self.class_label["color"]
		
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
			cv2.polylines(img=drawing, pts=pts, isClosed=True, color=color, thickness=2)
		
		if label:
			font = cv2.FONT_HERSHEY_SIMPLEX
			org  = (self.box_tl[0] + 5, self.box_tl[1])
			cv2.putText(
				img       = drawing,
				text      = self.class_label["name"],
				fontFace  = font,
				fontScale = 1.0,
				org       = org,
				color     = color,
				thickness = 2
			)
		
		return drawing

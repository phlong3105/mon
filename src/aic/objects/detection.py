#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Data class to store results from detectors.
Attribute includes: bounding box, confident score, class, uuid, ...
"""

from __future__ import annotations

import uuid
from timeit import default_timer as timer
from typing import Optional
from typing import Union

import cv2
import numpy as np

from one import box_xyxy_to_cxcyrh
from one import Color
from one import get_box_center as box_center

__all__ = [
	"Detection",
]


# MARK: - Detection

class Detection:
	"""Detection Dataclass. Convert raw detected output from detector to easy
	to use namespace.
	
	Attributes:
		id_ (int, str):
			Object unique ID.
		roi_id (int, str, optional):
			Unique ID of the ROI that the object is in. Else `None`.
			Default: `None`.
		box (np.ndarray, optional):
			Bounding box points as
			[top_left x, top_left y, bottom_right x, bottom_right y].
			Default: `None`.
		polygon (np.ndarray, optional):
			List of points. Default: `None`.
		confidence (float, optional):
			Confidence score. Default: `None`.
		class_label (dict, optional):
			Classlabel dict. Default: `None`.
		frame_index (int, optional):
			Index of frame when the Detection is created. Default: `None`.
		timestamp (float):
			Time when the object is created.
	"""
	
	# MARK: Magic Functions
	
	def __init__(
		self,
		id_        : Union[int, str]	       = uuid.uuid4().int,
		roi_id     : Optional[Union[int, str]] = None,
		box        : Optional[np.ndarray]      = None,
		polygon    : Optional[np.ndarray]      = None,
		confidence : Optional[float]           = None,
		class_label: Optional[dict]            = None,
		frame_index: Optional[int]             = None,
		timestamp  : float                     = timer(),
		*args, **kwargs
	):
		super().__init__()
		self.id_         = id_
		self.roi_id      = roi_id
		self.box         = box
		self.polygon     = polygon
		self.confidence  = confidence
		self.class_label = class_label
		self.frame_index = frame_index
		self.timestamp   = timestamp
	
	# MARK: Properties
	
	@property
	def box_cxcyrh(self):
		"""Return the box as [center_x, center_y, ratio, height]."""
		return box_xyxy_to_cxcyrh(self.box)

	@property
	def box_center(self):
		"""Return the box's center."""
		return box_center(self.box)
	
	@property
	def box_tl(self):
		"""Return the box's top left corner."""
		return self.box[0:2]
	
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
			color (tuple):
				Primary color. Default: `None`.
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

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Label handler for YOLO label/data format.
"""

from __future__ import annotations

import numpy as np
from one.imgproc import box_cxcywh_norm_to_xyxy
from one.imgproc import compute_box_area
from one.io import is_txt_file

from one.core import LABEL_HANDLERS
from one.data1.data_class import ImageInfo
from one.data1.data_class import ObjectAnnotation
from one.data1.data_class import VisionData
from one.data1.label_handler.base import BaseLabelHandler

__all__ = [
	"YoloLabelHandler"
]


# MARK: - YoloLabelHandler

@LABEL_HANDLERS.register(name="yolo")
class YoloLabelHandler(BaseLabelHandler):
	"""Handler for loading and dumping labels from Yolo label format to our
	custom label format defined in `onedataset.core.vision_data`.
	
	YOLO format:
		<object_category> <x_center> <y_center> <bbox_width> <bbox_height>
		<score> ...
		
		Where:
			<object_category> : Object category indicates the type of
							    annotated object.
			<x_center_norm>   : Fx coordinate of the center of rectangle.
			<y_center_norm>   : Fy coordinate of the center of rectangle.
			<bbox_width_norm> : Width in pixels of the predicted object
								bounding box.
			<bbox_height_norm>: Height in pixels of the predicted object
								bounding box.
			<score>           : Fscore in the DETECTION result file
								indicates the confidence of the predicted
							    bounding box enclosing an object measurement.
	"""
	
	# MARK: Load
	
	def load_from_file(self, image_path: str, label_path: str, **kwargs) -> VisionData:
		"""Load data from file.

		Args:
			image_path (str):
				Image file.
			label_path (str):
				Label file.
				
		Return:
			visual_data (VisualData):
				A `VisualData` item.
		"""
		# NOTE: Parse image info
		image_info = ImageInfo.from_file(image_path=image_path)
		shape0     = image_info.shape0
		
		# NOTE: Load content from file
		if is_txt_file(path=label_path):
			with open(label_path, "r") as f:
				labels = np.array([x.split() for x in f.read().splitlines()], np.float32)  # labels
		if len(labels) == 0:
			return VisionData(image_info=image_info)
		
		# NOTE: Parse all annotations
		objs = []
		for i, l in enumerate(labels):
			class_id 	    = int(l[0])
			box_cxcywh_norm = l[1:5]
			box_xyxy		= box_cxcywh_norm_to_xyxy(
				box_cxcywh_norm, shape0[0], shape0[1]
			)
			confidence 		= l[5]
			objs.append(
				ObjectAnnotation(
					class_id   = class_id,
					box        = box_cxcywh_norm,
					area       = compute_box_area(box_xyxy),
					confidence = confidence
				)
			)
			
		return VisionData(image_info=image_info, objects=objs)
	
	# MARK: Dump
	
	def dump_to_file(self, data: VisionData, path: str, **kwargs):
		"""Dump data from object to file.
		
		Args:
			data (VisualData):
				`VisualData` item.
			path (str):
				Label filepath to dump the data.
		"""
		if not is_txt_file(path=path):
			path += ".txt"
		
		# NOTE: Dump to file
		with open(path, "w") as f:
			for b in data.box_labels:
				ss = f"{b[4]} {b[0]} {b[1]} {b[2]} {b[3]} {b[6]}\n"
				f.writelines(ss)

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Label handler for Cityscapes label/data format.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Optional

import cv2
import numpy as np
from one.imgproc import box_xywh_to_xyxy
from one.imgproc import box_xyxy_to_cxcywh_norm
from one.io import dump_file
from one.io import is_json_file
from one.io import load_file

from one.core import LABEL_HANDLERS
from one.data1.data_class import ClassLabels
from one.data1.data_class import ImageInfo
from one.data1.data_class import ObjectAnnotation as Annotation
from one.data1.data_class import VisionData
from one.data1.label_handler.base import BaseLabelHandler
from one.vision.shape import box

__all__ = [
	"CityscapesLabelHandler"
]


# MARK: - CityscapesLabelHandler

@LABEL_HANDLERS.register(name="cityscapes")
class CityscapesLabelHandler(BaseLabelHandler):
	"""Handler for loading and dumping labels from Cityscapes label format
	to our custom label format defined in `onedataset.core.vision_data`.
	
	Cityscapes format:
		{
			"imgHeight": ...
			"imgWidth": ...
			"objects": [
				{
					"labels": ...
					"polygon": [
						[<x>, <y>],
						...
					]
				}
				...
			]
		}
	"""
	
	# MARK: Load
	
	def load_from_file(
		self,
		image_path  : str,
		label_path  : str,
		class_labels: Optional[ClassLabels] = None,
		**kwargs
	) -> VisionData:
		"""Load data from file.

		Args:
			image_path (str):
				Image filepath.
			label_path (str):
				Label filepath.
			class_labels (ClassLabels, optional):
				`ClassLabels` object contains all class-labels defined in
				the dataset.
				
		Return:
			visual_data (VisionData):
				A `VisualData` item.
		"""
		# NOTE: Load content from file
		label_dict = load_file(label_path) if is_json_file(label_path) else None
		height0    = label_dict.get("imgHeight")
		width0     = label_dict.get("imgWidth")
		objects    = label_dict.get("objects")
		
		# NOTE: Parse image info
		image_info = ImageInfo.from_file(image_path)

		if height0 != image_info.height0:
			image_info.height0 = height0
		else:
			image_info.height0 = image_info.height0

		if width0 != image_info.width0:
			image_info.width0 = width0
		else:
			image_info.width0 = image_info.width0

		# NOTE: Parse all annotations
		objs = []
		for i, l in enumerate(objects):
			label = l.get("label")
			# If the label is not known, but ends with a 'group'
			# (e.g. cargroup) try to remove the s and see if that works
			if (class_labels.get_id_by_name(name=label) is None) and	\
				label.endswith("group"):
				label = label[:-len("group")]
			if class_labels is None:
				class_id = label
			else:
				class_id = class_labels.get_id_by_name(name=label)
			polygon         = l.get("polygon", None)
			polygon			= np.array(polygon)
			box_xywh        = cv2.boundingRect(polygon)
			box_xywh        = np.array(box_xywh, np.float32)
			box_xyxy        = box_xywh_to_xyxy(box_xywh)
			box_cxcywh_norm = box_xyxy_to_cxcywh_norm(box_xyxy, height0, width0)
			annotation 		= Annotation(
				class_id = class_id,
				box      = box_cxcywh_norm,
				area     = box.compute_box_area(box_xyxy),
			)
			if polygon is not None:
				annotation.polygon_vertices = [polygon]
			objs.append(annotation)
			
		return VisionData(image_info=image_info, objects=objs)
	
	# MARK: Dump
	
	def dump_to_file(
		self,
		data        : VisionData,
		path        : str,
		class_labels: Optional[ClassLabels] = None,
		**kwargs
	):
		"""Dump data from object to file.
		
		Args:
			data (VisionData):
				`VisualData` item.
			path (str):
				Label filepath to dump the data.
			class_labels (ClassLabels, optional):
				`ClassLabels` object contains all class-labels defined in
				the dataset. Default: `None`.
		"""
		# NOTE: Prepare output data
		label_dict              = OrderedDict()
		info                    = data.image_info
		label_dict["imgHeight"] = info.height0
		label_dict["imgWidth"]  = info.width0
		
		objs = []
		for obj in data.objects:
			if class_labels is None:
				label = obj.class_id
			else:
				label = class_labels.get_name(key="id", value=obj.class_id)
			obj_dict = {
				"label"  : label,
				"polygon": obj.polygon_vertices[0]
			}
			objs.append(obj_dict)
		label_dict["objects"] = objs
		
		# NOTE: Dump to file
		dump_file(obj=label_dict, path=path, file_format="json")

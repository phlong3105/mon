#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Label handler for YOLO label/data format.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from pathlib import Path

import numpy as np

from one.core import LABEL_HANDLERS
from one.data.data_class import ImageInfo
from one.data.data_class import ObjectAnnotation as Annotation
from one.data.data_class import VisionData
from one.data.label_handler.base import BaseLabelHandler
from one.imgproc import box_cxcywh_norm_to_xyxy
from one.imgproc import box_xyxy_to_cxcywh_norm
from one.imgproc import compute_box_area
from one.io import dump_file
from one.io import is_xml_file
from one.io import load_file

"""
from one.core import error_console
try:
	import pyvips
except ImportError:
	error_console.log(f"Cannot import `pyvips`.")
"""

__all__ = [
	"PascalLabelHandler"
]


# MARK: - PascalLabelHandler

@LABEL_HANDLERS.register(name="pascal")
class PascalLabelHandler(BaseLabelHandler):
	"""Handler for loading and dumping labels from Pascal label format to
	our custom label format defined in `onedataset.core.vision_data`.
	
	Pascal Format:
	
		<annotation>
			<folder>GeneratedData_Train</folder>
			<filename>000001.png</filename>
			<path>/my/path/GeneratedData_Train/000001.png</path>
			<source>
				<database>Unknown</database>
			</source>
			<size>
				<width>224</width>
				<height>224</height>
				<depth>3</depth>
			</size>
			<segmented>0</segmented>
			<object>
				<name>21</name>
				<pose>Frontal</pose>
				<truncated>0</truncated>
				<difficult>0</difficult>
				<occluded>0</occluded>
				<bndbox>
					<xmin>82</xmin>
					<xmax>172</xmax>
					<ymin>88</ymin>
					<ymax>146</ymax>
				</bndbox>
			</object>
		</annotation>
	
	- name:
		This is the name of the object that we are trying to identify
		(i.e., class_id).
	- truncated:
		Indicates that the bounding box specified for the object does not
		correspond to the full extent of the object. For example, if an
		object is visible partially in the image then we set truncated to 1.
		If the object is fully visible then set truncated to 0.
	- difficult:
		An object is marked as difficult when the object is considered
		difficult to recognize. If the object is difficult to recognize then
		we set difficult to 1 else set it to 0.
	- bndbox:
		Axis-aligned rectangle specifying the extent of the object visible in
		the image.
	"""
	
	# MARK: Load
	
	def load_from_file(self, image_path: str, label_path: str, **kwargs) -> VisionData:
		"""Load data from file.

		Args:
			image_path (str):
				Image filepath.
			label_path (str):
				Label filepath.
				
		Return:
			visual_data (VisionData):
				A `VisualData` item.
		"""
		# NOTE: Parse image info
		image_info = ImageInfo.from_file(image_path=image_path)
		shape0     = image_info.shape0

		if (shape0[0] <= 9) or (shape0[1] <= 9):
			raise ValueError(f"Image size (height and width) must >= 10 pixel."
			                 f"But got: {shape0}.")
		
		# NOTE: Load content from file
		label_dict = load_file(label_path) if is_xml_file(label_path) else None
		label_dict = label_dict.get("annotation")
		version    = label_dict.get("ver", 0)
		folder     = label_dict.get("folder")
		filename   = label_dict.get("filename")
		path       = label_dict.get("path")
		size       = label_dict.get("size")
		height     = int(size.get("height"))
		width      = int(size.get("width"))
		depth      = int(size.get("depth"))
		segmented  = label_dict.get("segmented")
		_object    = label_dict.get("object")
		if _object is None:
			_object = []
		else:
			_object = [_object] if not isinstance(_object, list) else _object

		# NOTE: Parse image info
		common_prefix = os.path.commonprefix([image_path, label_path])
		stem          = str(Path(image_path).stem)
		
		info         = ImageInfo()
		info.id      = info.id   if (info.id != stem) else stem
		info.name    = (filename if (filename is not None)
						else str(Path(image_path).name))
		info.path    = image_path.replace(common_prefix, "")
		info.height0 = height if (height is not None) else shape0[0]
		info.width0  = width  if (width is not None)  else shape0[1]
		info.depth   = depth  if (depth is not None)  else info.depth
		
		# NOTE: Parse all annotations
		_objects = []
		for i, l in enumerate(_object):
			height0   = info.height0
			width0    = info.width0
			name      = l.get("name")
			class_id  = (int(name) if isinstance(name, str) and name.isnumeric() else name)
			pose      = l.get("pose")
			truncated = l.get("truncated")
			difficult = l.get("difficult")
			occluded  = l.get("occluded")
			bndbox    = l.get("bndbox")
			box_xyxy  = np.array([int(bndbox["xmin"]),
								  int(bndbox["ymin"]),
								  int(bndbox["ymin"]),
								  int(bndbox["ymax"])], np.float32)
			box_cxcywh_norm = box_xyxy_to_cxcywh_norm(box_xyxy, height0, width0)
			_objects.append(
				Annotation(
					class_id   = class_id,
					box        = box_cxcywh_norm,
					area       = compute_box_area(box_xyxy),
					truncation = truncated,
					occlusion  = occluded,
					difficult  = difficult
				)
			)
			
		return VisionData(image_info=info, objects=_objects)
	
	# MARK: Dump
	
	def dump_to_file(self, data: VisionData, path: str, **kwargs):
		"""Dump data from object to file.
		
		Args:
			data (VisionData):
				`VisualData` item.
			path (str):
				Label filepath to dump the data.
		"""
		# NOTE: Prepare output data
		label_dict              = OrderedDict()
		info                    = data.image_info
		label_dict["folder"]    = str(Path(info.path).parent)
		label_dict["filename"]  = info.name
		label_dict["path"]      = info.path
		label_dict["source"]    = {"database": "Unknown"}
		label_dict["size"] 	    = {"width" : info.width0,
								   "height": info.height0,
								   "depth" : info.depth}
		label_dict["segmented"] = 0
		
		objs = []
		for obj in data.objects:
			xyxy = box_cxcywh_norm_to_xyxy(obj.box, info.height0, info.width0)
			ann_dict = {
				"name"     : obj.class_id,
				"pose"     : "Unknown",
				"truncated": int(obj.truncation),
				"difficult": int(obj.difficult),
				"occluded" : int(obj.occlusion),
				"bndbox"   : {"xmin": int(xyxy[0]),
							  "xmax": int(xyxy[1]),
							  "ymin": int(xyxy[2]),
							  "ymax": int(xyxy[3])}
			}
			objs.append(ann_dict)
			
		label_dict["object"]     = objs
		label_dict["annotation"] = label_dict
		
		# NOTE: Dump to file
		dump_file(obj=label_dict, path=path, file_format="xml")

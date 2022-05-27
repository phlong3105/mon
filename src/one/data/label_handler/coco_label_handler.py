#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Label handler for COCO label/data format.
"""

import uuid
from collections import OrderedDict
from pathlib import Path

import numpy as np
from munch import Munch

from one.core import LABEL_HANDLERS
from one.core import progress_bar
from one.data.data_class import ImageInfo
from one.data.data_class import ObjectAnnotation as Annotation
from one.data.data_class import VisionData
from one.data.label_handler.base import BaseLabelHandler
from one.imgproc import box_xywh_to_cxcywh_norm
from one.io import dump_file
from one.io import is_json_file
from one.io import load_file

__all__ = [
	"CocoDetectionLabelHandler"
]


# MARK: - CocoLabelHandler

@LABEL_HANDLERS.register(name="coco_detection")
class CocoDetectionLabelHandler(BaseLabelHandler):
	"""In COCO dataset, all labels are concatenated into a single ,json file.
	Format is as follows:
	
	{
		"info": {
			"year": "2021",
			"version": "1.0",
			"description": "Exported from FiftyOne",
			"contributor": "Voxel51",
			"url": "https://fiftyone.ai",
			"date_created": "2021-01-19T09:48:27"
		},
		"licenses": [
			{
			  "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
			  "id": 1,
			  "name": "Attribution-NonCommercial-ShareAlike License"
			},
			...
		],
		"categories": [
			...
			{
				"id": 2,
				"name": "cat",
				"supercategory": "animal"
			},
			...
		],
		"images": [
			{
				"id": 0,
				"license": 1,
				"file_name": "<filename0>.<ext>",
				"height": 480,
				"width": 640,
				"date_captured": null
			},
			...
		],
		"annotations": [
			{
				"id": 0,
				"image_id": 0,
				"category_id": 2,
				"Ωbox": [260, 177, 231, 199],
				"segmentation": [...],
				"area": 45969,
				"iscrowd": 0
			},
			...
		]
	}
	
	- info: Description and versioning information about your dataset.
	- licenses: List of licenses with unique IDs to be specified by your
	  images.
	- categories: Classification categories each with a unique ID. Optionally
	  associated with a supercategory that can span multiple classes. These
	  categories can be whatever you want, but note that if you’d need to
	  follow the COCO classes if you want to use a model pretrained on COCO
	  out of the box (or follow other dataset categories to use other models).
	- images: List of images in your dataset and relevant metadata including
	  unique image ID, filepath, height, width, and optional attributes like
	  license, URL, date captured, etc.
	- annotations: List of annotations each with a unique ID and the image ID
	  it relates to. This is where you will store the bounding box information
	  in our case or segmentation/keypoint/other label information for other
	  tasks. This also stores bounding box area and iscrowd indicating a large
	  bounding box surrounding multiple objects of the same category which is
	  used for evaluation.
	  
	- FCOCO bounding box format is [top left x position, top left y
	  position, width, height].
	"""
	
	# MARK: Load
	
	def load_from_file(self, image_paths: list[str], label_path: str, **kwargs) -> dict[str, VisionData]:
		"""Load a list of `VisualData` objects from a .json file.

		Args:
			image_paths (list[str]):
				List of image paths.
			label_path (str):
				Label filepath.
				
		Return:
			data (dict[str, VisionData]):
				A list of `VisualData` objects.
		"""
		# NOTE: Load content from file
		if not is_json_file(label_path):
			raise AssertionError(f"`label_path` must be a JSON file. "
			                     f"But got: {label_path}")
		
		label_dict  = load_file(label_path)
		info	    = label_dict.get("info", 	    None)
		licenses    = label_dict.get("licenses",    None)
		categories  = label_dict.get("categories",  None)
		images	    = label_dict.get("images",	    None)
		annotations = label_dict.get("annotations", None)
		
		# NOTE: Parse image info
		with progress_bar() as pbar:
			data = Munch()
			temp = Munch()
			for image in pbar.track(
				images, description=f"[bright_yellow]Reading image info"
			):
				id		 = image.get("id", uuid.uuid4().int)
				filename = image.get("file_name", "")
				info 	 = ImageInfo(
					id      = id,
					name	= filename,
					path    = "",
					height0 = image.get("height", 0),
					width0  = image.get("width",  0),
					depth   = 3,
				)
				info.coco_url      = image.get("coco_url", "")
				info.flickr_url    = image.get("flickr_url", "")
				info.license 	   = image.get("license", 0)
				info.date_captured = image.get("date_captured", "")
				data[id] 		   = VisionData(image_info=info)
				temp[filename]     = id
		
		# NOTE: Parse all annotations
		with progress_bar() as pbar:
			for a in pbar.track(
				annotations, description=f"[bright_yellow]Reading annotations"
			):
				id		 		= a["id"]
				image_id 		= a["image_id"]
				class_id 		= a["category_id"]
				height0  		= data[image_id].image_info.height0
				width0   		= data[image_id].image_info.width0
				box_xywh 		= np.array(a["bbox"], np.float32)
				box_cxcywh_norm = box_xywh_to_cxcywh_norm(box_xywh, height0, width0)
				segmentation    = a["segmentation"]
				area 	 		= a["area"]
				is_crowd 		= a["iscrowd"]
				
				annotation = Annotation(
					id           = id,
					image_id     = image_id,
					class_id     = class_id,
					box          = box_cxcywh_norm,
					segmentation = segmentation,
					area         = area,
					is_crowd     = is_crowd,
				)
				data[image_id].objects.append(annotation)
		
		# NOTE: Associate labels with image_paths
		with progress_bar() as pbar:
			filenames = [str(Path(p).name) for p in image_paths]
			labels    = {}
			for i in pbar.track(
				range(len(filenames)),
				description=f"[bright_yellow]Create labels"
			):
				name = filenames[i]
				if name in temp:
					id                = temp[name]
					path              = image_paths[i]
					v                 = data[id]
					v.image_info.path = path
					labels[path]      = v
		
		return labels
	
	# MARK: Dump
	
	def dump_to_file(self, data: list[VisionData], path: str, **kwargs):
		"""Dump data from object to file.
		
		Args:
			data (list[VisionData]):
				A list of `VisualData` objects.
			path (str):
				Label filepath to dump the data.
		"""
		# NOTE: Prepare output data
		label_dict         = OrderedDict()
		label_dict["info"] = {
			"description" : "",
			"url"         : "",
			"year"        : "",
			"contributor" : "",
			"date_created": ""
		}
		label_dict["licenses"] = [
			dict(url="", id=0, name="")
		]
		label_dict["categories"] = [
			dict(id=0, name="", supercategory="")
		]
		
		images 	    = []
		annotations = []
		for d in data:
			image = {
				"id"           : d.image_info.id,
				"license"      : 0,
				"file_name"    : d.image_info.name,
				"coco_url"     : "",
				"flickr_url"   : "",
				"height"       : d.image_info.height0,
				"width"        : d.image_info.width0,
				"date_captured": ""
			}
			images.append(image)
			
			for o in d.objects:
				annotation = {
					"id"          : o.id,
					"image_id"    : o.image_id,
					"category_id" : o.class_id,
					"bbox"        : o.box.tolist(),
					"segmentation": o.segmentation,
					"area"		  : o.area,
					"iscrowd"     : o.is_crowd
				}
				annotations.append(annotation)
				
		# NOTE: Dump to file
		dump_file(obj=label_dict, path=path, file_format="json")

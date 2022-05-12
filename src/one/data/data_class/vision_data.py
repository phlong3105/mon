#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Data class for several vision tasks such as image classification, object
measurement, segmentation, ...
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Optional
from typing import Union

import numpy as np

from one.data.data_class.image_info import ImageInfo
from one.data.data_class.object_annotation import ObjectAnnotation

__all__ = [
	"VisionData"
]


# MARK: - VisionData

def _make_default_objects() -> list[ObjectAnnotation]:
	"""Returns an empty list of object annotations."""
	return []


@dataclass
class VisionData:
	"""Visual Data implements a data class for all vision tasks. This is a
	generalization of the COCO format.
	
	Attributes:
		image (np.ndarray, optional):
			Image.
		image_info (onedataset.core.image_info.ImageInfo):
			Image information.
		image_annotation (int, str, optional):
			Image annotation/label (usually, `class_id`). This is used
			for classification task.
		semantic (np.ndarray, optional):
			Each pixel has an ID that represents the ground truth label. This
			is used for semantic segmentation task.
		semantic_info (ImageInfo, optional):
			Semantic segmentation mask information.
		measurement (np.ndarray, optional):
			Pixel values encode both, class and the individual measurement.
			Let's say your labels.py assigns the ID 26 to the class `car`.
			Then, the individual cars in an image get the IDs 26000, 26001,
			26002, ... . A group of cars, where our annotators could not
			identify the individual detections anymore, is assigned to the ID 26.
			This is used for measurement segmentation class.
		instance_info (ImageInfo, optional):
			Detection image information.
		panoptic (np.ndarray, optional):
			This is used for panoptic segmentation task.
		panoptic_info (ImageInfo, optional):
			Panoptic image information.
		eimage (np.ndarray, optional):
			Short for "enhanced image". Fgood quality image. When this is
			used, `image` will have poor quality. This is used in image
			enhancement task.
		eimage_info (ImageInfo, optional):
			Enhanced image information.
		objects (list):
			List of all object annotations in the image. This is used for
			object measurement and measurement segmentation tasks.
	
	References:
		https://towardsdatascience.com/how-to-work-with-object-detection-datasets-in-coco-format-9bf4fb5848a4
	"""

	image           : Optional[np.ndarray]   = None
	image_info      : ImageInfo				 = ImageInfo()
	image_annotation: Union[int, str, None]  = None
	semantic        : Optional[np.ndarray]   = None
	semantic_info   : Optional[ImageInfo]    = None
	instance        : Optional[np.ndarray]   = None
	instance_info   : Optional[ImageInfo]    = None
	panoptic        : Optional[np.ndarray]   = None
	panoptic_info   : Optional[ImageInfo]    = None
	eimage          : Optional[np.ndarray]   = None
	eimage_info     : Optional[ImageInfo]    = None
	objects         : list[ObjectAnnotation] = field(default_factory=_make_default_objects)
	
	# MARK: Properties
	
	@property
	def box_labels(self) -> np.ndarray:
		"""Return bounding box labels for measurement task:
		<image_id> <class_id> <x1> <y1> <x2> <y2> <confidence> <area>
		<truncation> <occlusion>
		"""
		return np.array([obj.box_label for obj in self.objects], dtype=np.float32)
	
	@property
	def class_id(self) -> Optional[Union[int, str]]:
		"""An alias of `image_annotation`."""
		return self.image_annotation

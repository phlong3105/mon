#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements annotations that take the form of a bounding box."""

from __future__ import annotations

__all__ = [
	"BBoxAnnotation",
	"BBoxesAnnotation",
	"BBoxesAnnotationCOCO",
	"BBoxesAnnotationKITTI",
	"BBoxesAnnotationVOC",
	"BBoxesAnnotationYOLO",
	"TemporalBBoxAnnotation",
]

import numpy as np
import torch

from mon import core
from mon.core import _size_2_t
from mon.data.datastruct.annotation import base
from mon.data.datastruct.annotation.category import ClassLabels

console = core.console


# region BBox Annotation

class BBoxAnnotation(base.Annotation):
	"""An object detection data. Usually, it is represented as a list of
	bounding boxes (for an object with multiple parts created by an occlusion),
	and an instance mask.
	
	See Also: :class:`mon.data.datastruct.annotation.base.Annotation`.
	
	Args:
		id_: A class ID of the detection data. Default: ``-1`` means unknown.
		index: An index for the object. Default: ``-1``.
		label: base.Annotation string. Default: ``''``.
		confidence: A confidence value for the data. Default: ``1.0``.
		bbox: A bounding box's coordinates.
		mask: Instance segmentation masks for the object within its bounding
			bbox, which should be a binary (0/1) 2D sequence or a binary integer
			tensor. Default: ``None``.
	"""
	
	def __init__(
		self,
		id_       : int   = -1,
		index     : int   = -1,
		label     : str   = "",
		confidence: float = 1.0,
		bbox      : list  = [],
		mask      : list | None = None,
		*args, **kwargs
	):
		super().__init__(*args, **kwargs)
		if not 0.0 <= confidence <= 1.0:
			raise ValueError(f":param:`conf` must be between ``0.0`` and ``1.0``, but got {confidence}.")
		if id_ <= 0 and label == "":
			raise ValueError(f"Either :param:`id` or name must be defined, but got {id_} and {label}.")
		self._id         = id_
		self._index      = index
		self._label      = label
		self._confidence = confidence
		self._bbox       = bbox
		self._mask       = mask if mask is not None else None
	
	@classmethod
	def from_mask(cls, mask: np.ndarray, label: str, **kwargs) -> BBoxAnnotation:
		"""Create a :class:`BBoxAnnotation` object with its :param:`mask`
		attribute populated from the given full image mask. The instance mask
		for the object is extracted by computing the bounding rectangle of the
		non-zero values in the image mask.
		
		Args:
			mask: A binary (0/1) 2D sequence or a binary integer tensor.
			label: A label string.
			**kwargs: Additional attributes for the :class:`BBoxAnnotation`.
		
		Return:
			A :class:`BBoxAnnotation` object.
		"""
		raise NotImplementedError(f"This function has not been implemented!")
	
	@classmethod
	def from_value(cls, value: BBoxAnnotation | dict) -> BBoxAnnotation:
		"""Create a :class:`BBoxAnnotation` object from an arbitrary :param:`value`.
		"""
		if isinstance(value, dict):
			return BBoxAnnotation(**value)
		elif isinstance(value, BBoxAnnotation):
			return value
		else:
			raise ValueError(
				f":param:`value` must be a :class:`BBoxAnnotation` class or "
				f"a :class:`dict`, but got {type(value)}."
			)
	
	@property
	def data(self) -> list | None:
		"""The label's data."""
		return [
			self._bbox[0],
			self._bbox[1],
			self._bbox[2],
			self._bbox[3],
			self._id,
			self._label,
			self._confidence,
			self._index,
		]
	
	def to_polyline(self, tolerance: int = 2, filled: bool = True) -> "PolylineAnnotation":
		"""Return a :class:`PolylineAnnotation` object of this instance. If the
		detection has a mask, the returned polyline will trace the boundary of
		the mask. Otherwise, the polyline will trace the bounding bbox itself.
		
		Args:
			tolerance: A tolerance, in pixels, when generating an approximate
				polyline for the instance mask. Typical values are 1-3 pixels.
				Default: ``2``.
			filled: If ``True``, the polyline should be filled. Default: ``True``.
		
		Return:
			A :class:`PolylineAnnotation` object.
		"""
		raise NotImplementedError(f"This function has not been implemented!")
	
	def to_segmentation(
		self,
		mask      : np.ndarray | None = None,
		image_size: _size_2_t  | None = None,
		target    : int               = 255
	) -> "SegmentationAnnotation":
		"""Return a :class:`SegmentationAnnotation` object of this instance. The
		detection must have an instance mask, i.e., :param:`mask` attribute must
		be populated. You must give either :param:`mask` or :param:`frame_size`
		to use this method.
		
		Args:
			mask: An optional 2D integer numpy array to use as an initial mask
				to which to add this object. Default: ``None``.
			image_size: The size of the segmentation mask to render. This
				parameter has no effect if a :param:`mask` is provided. Defaults
				to ``None``.
			target: The pixel value to use to render the object. If you want
				color mask, just pass in the :param:`id` attribute.
				Default: ``255``.
		
		Return:
			A :class:`SegmentationAnnotation` object.
		"""
		raise NotImplementedError(f"This function has not been implemented!")


class BBoxesAnnotation(list[BBoxAnnotation], base.Annotation):
	"""A list of object detection labels in an image.
	
	See Also: :class:`mon.data.datastruct.annotation.base.Annotation`.
	
	Args:
		seq: A list of :class:`BBoxAnnotation` objects.
	"""
	
	def __init__(self, seq: list[BBoxAnnotation | dict]):
		super().__init__(BBoxAnnotation.from_value(value=i) for i in seq)
	
	def __setitem__(self, index: int, item: BBoxAnnotation | dict):
		super().__setitem__(index, BBoxAnnotation.from_value(item))
	
	def insert(self, index: int, item: BBoxAnnotation | dict):
		super().insert(index, BBoxAnnotation.from_value(item))
	
	def append(self, item: BBoxAnnotation | dict):
		super().append(BBoxAnnotation.from_value(item))
	
	def extend(self, other: list[BBoxAnnotation | dict]):
		super().extend([BBoxAnnotation.from_value(item) for item in other])
	
	@property
	def data(self) -> list | None:
		return [i.data for i in self]
	
	@property
	def ids(self) -> list[int]:
		return [i._id for i in self]
	
	@property
	def labels(self) -> list[str]:
		return [i._label for i in self]
	
	@property
	def bboxes(self) -> list:
		return [i._bbox for i in self]
	
	def to_polylines(self, tolerance: int = 2, filled: bool = True) -> "PolylinesAnnotation":
		"""Return a :class:`PolylinesAnnotation` object of this instance. For
		detections with masks, the returned polylines will trace the boundaries
		of the masks. Otherwise, the polylines will trace the bounding boxes
		themselves.
		
		Args:
			tolerance: A tolerance, in pixels, when generating an approximate
				polyline for the instance mask. Typical values are 1-3 pixels.
				Default: ``2``.
			filled: If ``True``, the polyline should be filled. Default: ``True``.
	  
		Return:
			A :class:`PolylinesAnnotation` object.
		"""
		raise NotImplementedError(f"This function has not been implemented!")
	
	def to_segmentation(
		self,
		mask      : np.ndarray | None = None,
		image_size: _size_2_t  | None = None,
		target    : int               = 255
	) -> "SegmentationAnnotation":
		"""Return a :class:`SegmentationAnnotation` object of this instance. Only
		detections with instance masks (i.e., their :param:`mask` attributes
		populated) will be rendered.
		
		Args:
			mask: An optional 2D integer numpy array to use as an initial mask
				to which to add this object. Default: ``None``.
			image_size: The shape of the segmentation mask to render. This
				parameter has no effect if a :param:`mask` is provided. Defaults
				to ``None``.
			target: The pixel value to use to render the object. If you want
				color mask, just pass in the :param:`id` attribute. Default:
				``255``.
		
		Return:
			A :class:`SegmentationAnnotation` object.
		"""
		raise NotImplementedError(f"This function has not been implemented!")


class BBoxesAnnotationCOCO(BBoxesAnnotation):
	"""A list of object detection labels in COCO format.
	
	See Also: :class:`BBoxesAnnotation`.
	"""
	pass


class BBoxesAnnotationKITTI(BBoxesAnnotation):
	"""A list of object detection labels in KITTI format.
	
	See Also: :class:`BBoxesAnnotation`.
	"""
	pass


class BBoxesAnnotationVOC(BBoxesAnnotation):
	"""A list of object detection labels in VOC format. One VOCDetections
	corresponds to one image and one annotation `.xml` file.
	
	See Also: :class:`BBoxesAnnotation`.
	
	Args:
		path: Absolute path where the image file is present.
		source: Specify the original location of the file in a database. Since
			we don't use a database, it is set to ``'Unknown'`` by default.
		size: Specify the width, height, depth of an image. If the image is
			black and white, then the depth will be ``1``. For color images,
			depth will be ``3``.
		segmented: Signify if the images contain annotations that are non-linear
			(irregular) in shape—commonly called polygons. Default:
			``0`` (linear shape).
		object: Contains the object details. If you have multiple annotations,
			then the object tag with its contents is repeated. The components of
			the object tags are:
			- name: This is the name of the object that we're trying to
			  identify (i.e., class_id).
			- pose: Specify the skewness or orientation of the image. Defaults
			  to ``'Unspecified'``, which means that the image isn't skewed.
			- truncated: Indicates that the bounding bbox specified for the
			  object doesn't correspond to the full extent of the object. For
			  example, if an object is visible partially in the image, then we
			  set truncated to ``1``. If the object is fully visible, then the
			  set truncated to ``0``.
			- difficult: An object is marked as difficult when the object is
			  considered difficult to recognize. If the object is difficult to
			   recognize, then we set difficult to ``1`` else set it to ``0``.
			- bndbox: Axis-aligned rectangle specifying the extent of the object
			  visible in the image.
		classlabels: ClassLabel object. Default: ``None``.
	"""
	
	def __init__(
		self,
		path       : core.Path = "",
		source     : dict      = {"database": "Unknown"},
		size       : dict      = {"width": 0, "height": 0, "depth": 3},
		segmented  : int       = 0,
		classlabels: ClassLabels | None = None,
		*args, **kwargs
	):
		super().__init__(*args, **kwargs)
		self._path        = core.Path(path)
		self._source      = source
		self._size        = size
		self._segmented   = segmented
		self._classlabels = classlabels
	
	@classmethod
	def from_file(
		cls,
		path       : core.Path | str,
		classlabels: ClassLabels | None = None
	) -> BBoxesAnnotationVOC:
		"""Create a :class:`VOCDetections` object from a `.xml` file.
		
		Args:
			path: Path to the `.xml` file.
			classlabels: :class:`ClassLabels` object. Default: ``None``.
			
		Return:
			A :class:`VOCDetections` object.
		"""
		from mon.vision import geometry
		path = core.Path(path)
		if not path.is_xml_file():
			raise ValueError(f":param:`path` must be a valid path to an ``.xml`` file, but got {path}.")
		
		xml_data = core.read_from_file(path=path)
		if "annotation" not in xml_data:
			raise ValueError("xml_data must contain the ``'annotation'`` key.")
		
		annotation = xml_data["annotation"]
		folder     = annotation.get("folder", "")
		filename   = annotation.get("file_name", "")
		image_path = annotation.get("path", "")
		source     = annotation.get("source", {"database": "Unknown"})
		size       = annotation.get("size", {"width": 0, "height": 0, "depth": 3})
		width      = int(size.get("width", 0))
		height     = int(size.get("height", 0))
		depth      = int(size.get("depth", 0))
		segmented  = annotation.get("segmented", 0)
		objects    = annotation.get("object", [])
		objects    = [objects] if not isinstance(objects, list) else objects
		
		detections: list[BBoxAnnotation] = []
		for i, o in enumerate(objects):
			name       = o.get["name"]
			bndbox     = o.get["bndbox"]
			bbox       = torch.FloatTensor([bndbox["xmin"], bndbox["ymin"], bndbox["xmax"], bndbox["ymax"]])
			bbox       = geometry.bbox_xyxy_to_cxcywhn(bbox=bbox, height=height, width=width)
			confidence = o.get("confidence", 1.0)
			truncated  = o.get("truncated",  0)
			difficult  = o.get("difficult" , 0)
			pose       = o.get("pose", "Unspecified")
			
			if name.isnumeric():
				_id = int(name)
			elif isinstance(classlabels, ClassLabels):
				_id = classlabels.get_id(key="name", value=name)
			else:
				_id = -1
			
			detections.append(
				BBoxAnnotation(
					id_       = _id,
					label     = name,
					bbox      = bbox,
					confidence= confidence,
					truncated = truncated,
					difficult = difficult,
					pose      = pose,
				)
			)
		return cls(
			path        = image_path,
			source      = source,
			size        = size,
			segmented   = segmented,
			detections  = detections,
			classlabels = classlabels
		)


class BBoxesAnnotationYOLO(BBoxesAnnotation):
	"""A list of object detection labels in YOLO format. YOLO label consists of
	several bounding boxes. One YOLO label corresponds to one image and one
	annotation file.
	
	See Also: :class:`BBoxesAnnotation`.
	"""
	
	@classmethod
	def from_file(cls, path: core.Path) -> BBoxesAnnotationYOLO:
		"""Create a :class:`YOLODetectionsLabel` object from a `.txt` file.
		
		Args:
			path: Path to the annotation `.txt` file.
		
		Return:
			A :class:`YOLODetections` object.
		"""
		path = core.Path(path)
		if not path.is_txt_file():
			raise ValueError(f":param:`path` must be a valid path to an ``.txt`` file, but got {path}.")
		
		detections: list[BBoxAnnotation] = []
		lines = open(path, "r").readlines()
		for l in lines:
			d          = l.split(" ")
			bbox       = [float(b) for b in d[1:5]]
			confidence = float(d[5]) if len(d) >= 6 else 1.0
			detections.append(
				BBoxAnnotation(
					id_        = int(d[0]),
					bbox       = np.array(bbox),
					confidence= confidence
				)
			)
		return cls(detections)


class TemporalBBoxAnnotation(base.Annotation):
	"""An object detection annotation in a video whose support is defined by a start
	and end frame. Usually, it is represented as a list of bounding boxes (for
	an object with multiple parts created by an occlusion), and an instance
	mask.
	
	See Also: :class:`mon.data.datastruct.annotation.base.Annotation`.
	"""
	
	@property
	def data(self) -> list | None:
		raise NotImplementedError(f"This function has not been implemented!")

# endregion

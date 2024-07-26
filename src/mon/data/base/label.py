#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements multiple label types. We try to support all possible
data types: :class:`torch.Tensor`, :class:`numpy.ndarray`, or :class:`Sequence`,
but we prioritize :class:`torch.Tensor`.
"""

from __future__ import annotations

__all__ = [
	"ClassLabel",
	"ClassLabels",
	"ClassificationAnnotation",
	"ClassificationsAnnotation",
	"DetectionAnnotation",
	"DetectionsAnnotation",
	"DetectionsLabelCOCO",
	"DetectionsLabelKITTI",
	"DetectionsLabelVOC",
	"DetectionsLabelYOLO",
	"FrameAnnotation",
	"HeatmapAnnotation",
	"ImageAnnotation",
	"KeypointAnnotation",
	"KeypointsAnnotation",
	"KeypointsLabelCOCO",
	"Annotation",
	"PolylineAnnotation",
	"PolylinesAnnotation",
	"RegressionAnnotation",
	"SegmentationAnnotation",
	"TemporalDetectionAnnotation",
]

import uuid
from abc import ABC, abstractmethod
from typing import Any

import cv2
import numpy as np
import torch

from mon import core
from mon.core import _size_2_t

console = core.console


# region Base

class Annotation(ABC):
	"""The base class for all label classes. A label instance represents a
	logical collection of data associated with a particular task.
	"""
	
	@property
	@abstractmethod
	def data(self) -> list | None:
		"""The label's data."""
		pass
	
	@property
	def nparray(self) -> np.ndarray | None:
		"""The label's data as a :class:`numpy.ndarray`."""
		data = self.data
		if isinstance(data, list):
			data = np.array([i for i in data if isinstance(i, int | float)])
		return data
	
	@property
	def tensor(self) -> torch.Tensor | None:
		"""The label's data as a :class:`torch.Tensor`."""
		data = self.data
		if isinstance(data, list):
			data = torch.Tensor([i for i in data if isinstance(i, int | float)])
		return data

# endregion


# region ClassLabel

class ClassLabel(dict, Annotation):
	"""A class-label represents a class pre-defined in a dataset. It consists of
	basic attributes such as ID, name, and color.
	"""
	
	@classmethod
	def from_value(cls, value: ClassLabel | dict) -> ClassLabel:
		"""Create a :class:`ClassLabels` object from an arbitrary :param:`value`.
		"""
		if isinstance(value, dict):
			return ClassLabel(value)
		elif isinstance(value, ClassLabel):
			return value
		else:
			raise ValueError(
				f":param:`value` must be a :class:`ClassLabel` class or a "
				f":class:`dict`, but got {type(value)}."
			)
	
	@property
	def data(self) -> list | None:
		return None


class ClassLabels(list[ClassLabel]):
	"""A :class:`list` of all the class-labels defined in a dataset.
	
	Notes:
		We inherit the standard Python :class:`list` to take advantage of the
		built-in functions.
	"""
	
	def __init__(self, seq: list[ClassLabel | dict]):
		super().__init__(ClassLabel.from_value(value=i) for i in seq)
	
	def __setitem__(self, index: int, item: ClassLabel | dict):
		super().__setitem__(index, ClassLabel.from_value(item))
	
	def insert(self, index: int, item: ClassLabel | dict):
		super().insert(index, ClassLabel.from_value(item))
	
	def append(self, item: ClassLabel | dict):
		super().append(ClassLabel.from_value(item))
	
	def extend(self, other: list[ClassLabel | dict]):
		super().extend([ClassLabel.from_value(item) for item in other])
	
	@classmethod
	def from_dict(cls, value: dict) -> ClassLabels:
		"""Create a :class:`ClassLabels` object from a :class:`dict` :param:`d`.
		The :class:`dict` must contain the key ``'classlabels'``, and it's
		corresponding value is a list of dictionary. Each item in the list
		:param:`d["classlabels"]` is a dictionary describing a
		:class:`ClassLabel` object.
		"""
		if "classlabels" not in value:
			raise ValueError("value must contains a 'classlabels' key.")
		classlabels = value["classlabels"]
		if not isinstance(classlabels, list | tuple):
			raise TypeError(
				f":param:`classlabels` must be a :class:`list` or "
				f":class:`tuple`, but got {type(classlabels)}."
			)
		return cls(seq=classlabels)
	
	@classmethod
	def from_file(cls, path: core.Path) -> ClassLabels:
		"""Create a :class:`ClassLabels` object from the content of a ``.json``
		file specified by the :param:`path`.
		"""
		path = core.Path(path)
		if not path.is_json_file():
			raise ValueError(f":param:`path` must be a ``.json`` file, but got {path}.")
		return cls.from_dict(core.read_from_file(path=path))
	
	@classmethod
	def from_value(cls, value: Any) -> ClassLabels | None:
		"""Create a :class:`ClassLabels` object from an arbitrary
		:param:`value`.
		"""
		if isinstance(value, ClassLabels):
			return value
		if isinstance(value, dict):
			return cls.from_dict(value)
		if isinstance(value, list | tuple):
			return cls(value)
		if isinstance(value, str | core.Path):
			return cls.from_file(value)
		return None
	
	@property
	def classes(self) -> list[ClassLabel]:
		"""An alias."""
		return self
	
	def color_legend(self, height: int | None = None) -> np.array:
		"""Create a legend figure of all the classlabels.
		
		Args:
			height: The height of the legend. If None, it will be
				25px * :meth:`__len__`.
		
		Return:
			An RGB color legend figure.
		"""
		num_classes = len(self)
		row_height = 25 if (height is None) else int(height / num_classes)
		legend = np.zeros(
			((num_classes * row_height) + 25, 300, 3),
			dtype=np.uint8
		)
		
		# Loop over the class names + colors
		for i, label in enumerate(self):
			color = label.color  # Draw the class name + color on the legend
			color = color[::-1]  # Convert to BGR format since OpenCV operates on
			# BGR format.
			cv2.putText(
				img       = legend,
				text      = label.name,
				org       = (5, (i * row_height) + 17),
				fontFace  = cv2.FONT_HERSHEY_SIMPLEX,
				fontScale = 0.5,
				color     = (0, 0, 255),
				thickness = 2
			)
			cv2.rectangle(
				img       = legend,
				pt1       = (150, (i * 25)),
				pt2       = (300, (i * row_height) + 25),
				color     = color,
				thickness = -1
			)
		return legend
	
	def colors(
		self,
		key: str = "id",
		exclude_negative_key: bool = True,
	) -> list:
		"""Return a :class:`list` of colors corresponding to the items in
		:attr:`self`.
		
		Args:
			key: The key to search for. Default: ``'id'``.
			exclude_negative_key: If ``True``, excludes the key with negative
				value. Default: ``True``.
			
		Return:
			A list of colors.
		"""
		colors = []
		for c in self:
			key_value = c.get(key, None)
			if (key_value is None) or (exclude_negative_key and key_value < 0):
				continue
			color = c.get("color", [255, 255, 255])
			colors.append(color)
		return colors
	
	@property
	def id2label(self) -> dict[int, dict]:
		"""A :class:`dict` mapping items' IDs (keys) to items (values)."""
		return {label["id"]: label for label in self}
	
	def ids(
		self,
		key: str = "id",
		exclude_negative_key: bool = True,
	) -> list:
		"""Return a :class:`list` of IDs corresponding to the items in
		:attr:`self`.
		
		Args:
			key: The key to search for. Default: ``'id'``.
			exclude_negative_key: If ``True``, excludes the key with negative
				value. Default: ``True``.
			
		Return:
			A :class:`list` of IDs.
		"""
		ids = []
		for c in self:
			key_value = c.get(key, None)
			if (id is None) or (exclude_negative_key and key_value < 0):
				continue
			ids.append(key_value)
		return ids
	
	@property
	def name2label(self) -> dict[str, dict]:
		"""A dictionary mapping items' names (keys) to items (values)."""
		return {c["name"]: c for c in self.classes}
	
	def names(self, exclude_negative_key: bool = True) -> list:
		"""Return a list of names corresponding to the items in :attr:`self`.
		
		Args:
			exclude_negative_key: If ``True``, excludes the key with negative
				value. Default: ``True``.
			
		Return:
			A list of IDs.
		"""
		names = []
		for c in self:
			key_value = c.get("id", None)
			if (key_value is None) or (exclude_negative_key and key_value < 0):
				continue
			name = c.get("name", "")
			names.append(name)
		return names
	
	def num_classes(
		self,
		key: str = "id",
		exclude_negative_key: bool = True,
	) -> int:
		"""Counts the number of items.
		
		Args:
			key: The key to search for. Default: ``'id'``.
			exclude_negative_key: If ``True``, excludes the key with negative
				value. Default: ``True``.
			
		Return:
			The number of items (classes) in the dataset.
		"""
		count = 0
		for c in self:
			key_value = c.get(key, None)
			if (key_value is None) or (exclude_negative_key and key_value < 0):
				continue
			count += 1
		return count
	
	def get_class(self, key: str = "id", value: Any = None) -> dict | None:
		"""Return the item (class-label) matching the given :param:`key` and
		:param:`value`.
		"""
		for c in self:
			key_value = c.get(key, None)
			if (key_value is not None) and (value == key_value):
				return c
		return None
	
	def get_class_by_name(self, name: str) -> dict | None:
		"""Return the item (class-label) with the :param:`key` is ``'name'`` and
		value matching the given :param:`name`.
		"""
		return self.get_class(key="name", value=name)
	
	def get_id(self, key: str = "id", value: Any = None) -> int | None:
		"""Return the ID of the item (class-label) matching the given
		:param:`key` and :param:`value`.
		"""
		classlabel: dict = self.get_class(key=key, value=value)
		return classlabel["id"] if classlabel is not None else None
	
	def get_id_by_name(self, name: str) -> int | None:
		"""Return the name of the item (class-label) with the :param:`key` is
		'name' and value matching the given :param:`name`.
		"""
		classlabel = self.get_class_by_name(name=name)
		return classlabel["id"] if classlabel is not None else None
	
	def get_name(self, key: str = "id", value: Any = None) -> str | None:
		"""Return the name of the item (class-label) with the :param:`key` is
		'name' and value matching the given :param:`name`.
		"""
		c = self.get_class(key=key, value=value)
		return c["name"] if c is not None else None
	
	@property
	def tensor(self) -> torch.Tensor | None:
		return None
	
	def print(self):
		"""Print all items (class-labels) in a rich format."""
		if len(self) <= 0:
			console.log("[yellow]No class is available.")
			return
		console.log("Classlabels:")
		core.print_table(self.classes)


def majority_voting(labels: list[ClassLabel]) -> ClassLabel:
	"""Counts the number of appearances of each class-label, and returns the
	label with the highest count.
	
	Args:
		labels: A :class:`list` of :class:`ClassLabel`s.
	
	Return:
		The :class:`ClassLabel` object that has the most votes.
	"""
	# Count number of appearances of each label.
	unique_labels = {}
	label_voting  = {}
	for label in labels:
		k = label.get("id")
		v = label_voting.get(k)
		if v:
			label_voting[k]  = v + 1
		else:
			unique_labels[k] = label
			label_voting[k]  = 1
	
	# Get k (label's id) with max v
	max_id = max(label_voting, key=label_voting.get)
	return unique_labels[max_id]

# endregion


# region Image Annotation

class ImageAnnotation(Annotation):
	"""A ground-truth image label for an image.
	
	Args:
		id_: An ID of the image. This can be an integer or a string. This
			attribute is useful for batch processing where you want to keep the
			objects in the correct frame sequence.
		name: A name of the image. Default: ``None``.
		path: A path to the image file. Default: ``None``.
		image: A ground-truth image to be loaded. Default: ``None``.
		load: If ``True``, the image will be loaded into memory when
			the object is created. Default: ``False``.
		cache: If ``True``, the image will be loaded into memory and
			kept there. Default: ``False``.

	References:
		`<https://www.tensorflow.org/datasets/api_docs/python/tfds/features/Image>`__
		
	 See Also: :class:`Annotation`.
	"""
	
	def __init__(
		self,
		id_  : int               = uuid.uuid4().int,
		name : str        | None = None,
		path : core.Path  | None = None,
		image: np.ndarray | None = None,
		load : bool              = False,
		cache: bool              = False,
		*args, **kwargs
	):
		super().__init__(*args, **kwargs)
		self._id             = id_
		self._image          = None
		self._keep_in_memory = cache
		
		self._path = core.Path(path) if path is not None else None
		if (self.path is None or not self.path.is_image_file()) and image is None:
			raise ValueError(f":param:`path` must be a valid path to an image file, but got {path}.")
		
		if name is None:
			name = str(core.Path(path).name) if path.is_image_file() else f"{id_}"
		self._name = name
		self._stem = core.Path(self._name).stem
		
		if load and image is None:
			image = self.load()
		
		self._shape = core.get_image_shape(input=image) if image is not None else None
		
		if self._keep_in_memory:
			self._image = image
	
	def load(
		self,
		path : core.Path | None = None,
		cache: bool 			= False,
	) -> np.ndarray | None:
		"""Loads image into memory.
		
		Args:
			path: The path to the image file. Default: ``None``.
			cache: If ``True``, the image will be loaded into memory
				and kept there. Default: ``False``.
			
		Return:
			An image of shape :math:`[H, W, C]`.
		"""
		self._keep_in_memory = cache
		
		if path is not None:
			path = core.Path(path)
			if path.is_image_file():
				self._path = path
		if self.path is None or not self.path.is_image_file():
			raise ValueError(f":param:`path` must be a valid path to an image file, but got {self.path}.")
		
		image = core.read_image(path=self.path, to_rgb=True, to_tensor=False, normalize=False)
		self._shape = core.get_image_shape(input=image) if (image is not None) else self._shape
		
		if self._keep_in_memory:
			self._image = image
		return image
	
	@property
	def path(self) -> core.Path:
		"""The path to the image file."""
		return self._path
	
	@property
	def data(self) -> np.ndarray | None:
		if self._image is None:
			return self.load()
		else:
			return self._image
	
	@property
	def meta(self) -> dict:
		"""Return a dictionary of metadata about the object. The dictionary
		includes ID, name, path, and shape of the image.
		"""
		return {
			"id"   : self._id,
			"name" : self._name,
			"stem" : self._stem,
			"path" : self.path,
			"shape": self._shape,
			"hash" : self.path.stat().st_size if isinstance(self.path, core.Path) else None,
		}


class FrameAnnotation(Annotation):
	"""A ground-truth image label for a video frame.
	
	Args:
		id_: An ID of the image. This can be an integer or a string. This
			attribute is useful for batch processing where you want to keep the
			objects in the correct frame sequence.
		index: An index of the frame. Default: ``None``.
		path: A path to the video file. Default: ``None``.
		frame: A ground-truth image to be loaded. Default: ``None``.
		cache: If ``True``, the image will be loaded into memory and
			kept there. Default: ``False``.

	References:
		`<https://www.tensorflow.org/datasets/api_docs/python/tfds/features/Image>`__
		
	 See Also: :class:`Annotation`.
	"""
	
	def __init__(
		self,
		id_  : int               = uuid.uuid4().int,
		index: str        | None = None,
		path : core.Path  | None = None,
		frame: np.ndarray | None = None,
		cache: bool              = False,
		*args, **kwargs
	):
		super().__init__(*args, **kwargs)
		self._id             = id_
		self._index			 = index
		self._frame          = frame
		self._keep_in_memory = cache
		self._path 		     = core.Path(path) if path is not None else None
		self._name  		 = str(self.path.name) if self.path is not None else f"{id_}"
		self._stem  		 = core.Path(self._name).stem
		self._shape			 = core.get_image_shape(input=frame) if frame is not None else None
	
	@property
	def path(self) -> core.Path:
		"""The path to the video file."""
		return self._path
	
	@property
	def data(self) -> np.ndarray | None:
		return self._frame

	@property
	def meta(self) -> dict:
		"""Return a dictionary of metadata about the object. The dictionary
		includes ID, name, path, and shape of the image.
		"""
		return {
			"id"   : self._id,
			"name" : self._name,
			"stem" : self._stem,
			"path" : self.path,
			"shape": self._shape,
			"hash" : self.path.stat().st_size if isinstance(self.path, core.Path) else None,
		}
	
# endregion


# region Classification Annotation

class ClassificationAnnotation(Annotation):
	"""A classification label for an image.
	
	See Also: :class:`Annotation`.
	
	Args:
		id_: A class ID of the classification data. Default: ``-1`` means unknown.
		label: A label string. Default: ``''``.
		confidence: A confidence value for the data. Default: ``1.0``.
		logits: Logits associated with the labels. Default: ``None``.
	"""
	
	def __init__(
		self,
		id_       : int   = -1,
		label     : str   = "",
		confidence: float = 1.0,
		logits    : np.ndarray | None = None,
		*args, **kwargs
	):
		super().__init__(*args, **kwargs)
		if not 0.0 <= confidence <= 1.0:
			raise ValueError(f":param:`conf` must be between ``0.0`` and ``1.0``, but got {confidence}.")
		if id_ <= 0 and label == "":
			raise ValueError(f"Either :param:`id` or :param:`name` must be defined, but got {id_} and {label}.")
		self._id         = id_
		self._label      = label
		self._confidence = confidence
		self._logits     = np.array(logits) if logits is not None else None
	
	@classmethod
	def from_value(cls, value: ClassificationAnnotation | dict) -> ClassificationAnnotation:
		"""Create a :class:`ClassificationAnnotation` object from an arbitrary
		:param:`value`.
		"""
		if isinstance(value, dict):
			return ClassificationAnnotation(**value)
		elif isinstance(value, ClassificationAnnotation):
			return value
		else:
			raise ValueError(
				f":param:`value` must be a :class:`ClassificationAnnotation` class "
				f"or a :class:`dict`, but got {type(value)}."
			)
	
	@property
	def data(self) -> list | None:
		return [self._id, self._label]


class ClassificationsAnnotation(list[ClassificationAnnotation], Annotation):
	"""A list of classification labels for an image. It is used for multi-labels
	or multi-classes classification tasks.
	
	See Also: :class:`Annotation`.
	
	Args:
		seq: A list of :class:`ClassificationAnnotation` objects.
	"""
	
	def __init__(self, seq: list[ClassificationAnnotation | dict]):
		super().__init__(ClassificationAnnotation.from_value(value=i) for i in seq)
	
	def __setitem__(self, index: int, item: ClassificationAnnotation | dict):
		super().__setitem__(index, ClassificationAnnotation.from_value(item))
	
	def insert(self, index: int, item: ClassificationAnnotation | dict):
		super().insert(index, ClassificationAnnotation.from_value(item))
	
	def append(self, item: ClassificationAnnotation | dict):
		super().append(ClassificationAnnotation.from_value(item))
	
	def extend(self, other: list[ClassificationAnnotation | dict]):
		super().extend([ClassificationAnnotation.from_value(item) for item in other])
	
	@property
	def data(self) -> list | None:
		return [i.data for i in self]
	
	@property
	def ids(self) -> list[int]:
		return [i._id for i in self]
	
	@property
	def labels(self) -> list[str]:
		return [i._label for i in self]
	
# endregion


# region Regression Annotation

class RegressionAnnotation(Annotation):
	"""A single regression value.
	
	See Also: :class:`Annotation`.
	
	Args:
		value: The regression value.
		confidence: A confidence value for the data. Default: ``1.0``.
	"""
	
	def __init__(
		self,
		value     : float,
		confidence: float = 1.0,
		*args, **kwargs
	):
		super().__init__(*args, **kwargs)
		if not 0.0 <= confidence <= 1.0:
			raise ValueError(f":param:`conf` must be between ``0.0`` and ``1.0``, but got {confidence}.")
		self._value      = value
		self._confidence = confidence
	
	@property
	def data(self) -> list | None:
		return [self._value]

# endregion


# region Detection Annotation

class DetectionAnnotation(Annotation):
	"""An object detection data. Usually, it is represented as a list of
	bounding boxes (for an object with multiple parts created by an occlusion),
	and an instance mask.
	
	See Also: :class:`Annotation`.
	
	Args:
		id_: A class ID of the detection data. Default: ``-1`` means unknown.
		index: An index for the object. Default: ``-1``.
		label: Annotation string. Default: ``''``.
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
	def from_mask(cls, mask: np.ndarray, label: str, **kwargs) -> DetectionAnnotation:
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
	def from_value(cls, value: DetectionAnnotation | dict) -> DetectionAnnotation:
		"""Create a :class:`BBoxAnnotation` object from an arbitrary :param:`value`.
		"""
		if isinstance(value, dict):
			return DetectionAnnotation(**value)
		elif isinstance(value, DetectionAnnotation):
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
	
	def to_polyline(self, tolerance: int = 2, filled: bool = True) -> PolylineAnnotation:
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
	) -> SegmentationAnnotation:
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


class DetectionsAnnotation(list[DetectionAnnotation], Annotation):
	"""A list of object detection labels in an image.
	
	See Also: :class:`Annotation`.
	
	Args:
		seq: A list of :class:`BBoxAnnotation` objects.
	"""
	
	def __init__(self, seq: list[DetectionAnnotation | dict]):
		super().__init__(DetectionAnnotation.from_value(value=i) for i in seq)
	
	def __setitem__(self, index: int, item: DetectionAnnotation | dict):
		super().__setitem__(index, DetectionAnnotation.from_value(item))
	
	def insert(self, index: int, item: DetectionAnnotation | dict):
		super().insert(index, DetectionAnnotation.from_value(item))
	
	def append(self, item: DetectionAnnotation | dict):
		super().append(DetectionAnnotation.from_value(item))
	
	def extend(self, other: list[DetectionAnnotation | dict]):
		super().extend([DetectionAnnotation.from_value(item) for item in other])
	
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
	
	def to_polylines(self, tolerance: int = 2, filled: bool = True) -> PolylinesAnnotation:
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
	) -> SegmentationAnnotation:
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


class DetectionsLabelCOCO(DetectionsAnnotation):
	"""A list of object detection labels in COCO format.
	
	See Also: :class:`BBoxesAnnotation`.
	"""
	pass


class DetectionsLabelKITTI(DetectionsAnnotation):
	"""A list of object detection labels in KITTI format.
	
	See Also: :class:`BBoxesAnnotation`.
	"""
	pass


class DetectionsLabelVOC(DetectionsAnnotation):
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
	) -> DetectionsLabelVOC:
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
		
		detections: list[DetectionAnnotation] = []
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
				DetectionAnnotation(
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


class DetectionsLabelYOLO(DetectionsAnnotation):
	"""A list of object detection labels in YOLO format. YOLO label consists of
	several bounding boxes. One YOLO label corresponds to one image and one
	annotation file.
	
	See Also: :class:`BBoxesAnnotation`.
	"""
	
	@classmethod
	def from_file(cls, path: core.Path) -> DetectionsLabelYOLO:
		"""Create a :class:`YOLODetectionsLabel` object from a `.txt` file.
		
		Args:
			path: Path to the annotation `.txt` file.
		
		Return:
			A :class:`YOLODetections` object.
		"""
		path = core.Path(path)
		if not path.is_txt_file():
			raise ValueError(f":param:`path` must be a valid path to an ``.txt`` file, but got {path}.")
		
		detections: list[DetectionAnnotation] = []
		lines = open(path, "r").readlines()
		for l in lines:
			d          = l.split(" ")
			bbox       = [float(b) for b in d[1:5]]
			confidence = float(d[5]) if len(d) >= 6 else 1.0
			detections.append(
				DetectionAnnotation(
					id_        = int(d[0]),
					bbox       = np.array(bbox),
					confidence= confidence
				)
			)
		return cls(detections)


class TemporalDetectionAnnotation(Annotation):
	"""An object detection label in a video whose support is defined by a start
	and end frame. Usually, it is represented as a list of bounding boxes (for
	an object with multiple parts created by an occlusion), and an instance
	mask.
	
	See Also: :class:`Annotation`.
	"""
	
	@property
	def data(self) -> list | None:
		raise NotImplementedError(f"This function has not been implemented!")

# endregion


# region Keypoint Annotation

class KeypointAnnotation(Annotation):
	"""A list keypoints label for a single object in an image.
	
	See Also: :class:`Annotation`.
	
	Args:
		id_: The class ID of the polyline data. Default: ``-1`` means unknown.
		index: An index for the polyline. Default: ``-1``.
		label: The label string. Default: ``''``.
		confidence: A confidence value for the data. Default: ``1.0``.
		points: A list of lists of :math:`(x, y)` points in :math:`[0, 1] x [0, 1]`.
	"""
	
	def __init__(
		self,
		id_       : int   = -1,
		index     : int   = -1,
		label     : str   = "",
		confidence: float = 1.0,
		points    : list  = [],
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
		self._points     = points
	
	@classmethod
	def from_value(cls, value: KeypointAnnotation | dict) -> KeypointAnnotation:
		"""Create a :class:`KeypointAnnotation` object from an arbitrary :param:`value`.
		"""
		if isinstance(value, dict):
			return KeypointAnnotation(**value)
		elif isinstance(value, KeypointAnnotation):
			return value
		else:
			raise ValueError(
				f":param:`value` must be a :class:`KeypointAnnotation` class or a "
				f":class:`dict`, but got {type(value)}."
			)
	
	@property
	def data(self) -> list | None:
		"""The label's data."""
		return [
			self._points,
			self._id,
			self._label,
			self._confidence,
			self._index,
		]


class KeypointsAnnotation(list[KeypointAnnotation], Annotation):
	"""A list of keypoint labels for multiple objects in an image.
	
	See Also: :class:`Annotation`.
	
	Args:
		seq: A list of :class:`KeypointAnnotation` objects.
	"""
	
	def __init__(self, seq: list[KeypointAnnotation | dict]):
		super().__init__(KeypointAnnotation.from_value(value=i) for i in seq)
	
	def __setitem__(self, index: int, item: KeypointAnnotation | dict):
		super().__setitem__(index, KeypointAnnotation.from_value(item))
	
	def insert(self, index: int, item: KeypointAnnotation | dict):
		super().insert(index, KeypointAnnotation.from_value(item))
	
	def append(self, item: KeypointAnnotation | dict):
		super().append(KeypointAnnotation.from_value(item))
	
	def extend(self, other: list[KeypointAnnotation | dict]):
		super().extend([KeypointAnnotation.from_value(item) for item in other])
	
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
	def points(self) -> list:
		return [i._points for i in self]


class KeypointsLabelCOCO(KeypointsAnnotation):
	"""A list of keypoint labels for multiple objects in COCO format.
	
	See Also: :class:`KeypointsAnnotation`.
	"""
	pass

# endregion


# region Polyline Annotation

class PolylineAnnotation(Annotation):
	"""A set of semantically related polylines or polygons for a single object
	in an image.
	
	See Also: :class:`Annotation`.
	
	Args:
		id_: The class ID of the polyline data. Default: ``-1`` means unknown.
		index: An index for the polyline. Default: ``-1``.
		label: The label string. Default: ``''``.
		confidence: A confidence value for the data. Default: ``1.0``.
		points: A list of lists of :math:`(x, y)` points in
			:math:`[0, 1] x [0, 1]` describing the vertices of each shape in the
			polyline.
		closed: Whether the shapes are closed, in other words, and edge should
			be drawn. from the last vertex to the first vertex of each shape.
			Default: ``False``.
		filled: Whether the polyline represents polygons, i.e., shapes that
			should be filled when rendering them. Default: ``False``.
	"""
	
	def __init__(
		self,
		id_       : int   = -1,
		index     : int   = -1,
		label     : str   = "",
		confidence: float = 1.0,
		points    : list  = [],
		closed    : bool  = False,
		filled    : bool  = False,
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
		self._closed     = closed
		self._filled     = filled
		self._confidence = confidence
		self._points     = points
	
	@classmethod
	def from_mask(
		cls,
		mask     : np.ndarray,
		label    : str = "",
		tolerance: int = 2,
		**kwargs
	) -> PolylineAnnotation:
		"""Create a :class:`PolylineAnnotation` instance with its :param:`mask`
		attribute populated from the given full image mask. The instance mask
		for the object is extracted by computing the bounding rectangle of the
		non-zero values in the image mask.
		
		Args:
			mask: An optional 2D integer numpy array to use as an initial mask
				to which to add this object. Default: ``None``.
			label: A label string. Default: ``''``.
			tolerance: A tolerance, in pixels, when generating approximate
				polygons for each region. Typical values are 1-3 pixels.
				Default: ``2``.
			**kwargs: additional attributes for the :class:`PolylineAnnotation`.
		
		Return:
			A :class:`PolylineAnnotation` object.
		"""
		pass
	
	@classmethod
	def from_value(cls, value: PolylineAnnotation | dict) -> PolylineAnnotation:
		"""Create a :class:`PolylineAnnotation` object from an arbitrary :param:`value`.
		"""
		if isinstance(value, dict):
			return PolylineAnnotation(**value)
		elif isinstance(value, PolylineAnnotation):
			return value
		else:
			raise ValueError(
				f":param:`value` must be a :class:`PolylineAnnotation` class or a "
				f":class:`dict`, but got {type(value)}."
			)
	
	@property
	def data(self) -> list | None:
		return [
			self._points,
			self._id,
			self._label,
			self._confidence,
			self._index,
		]
	
	def to_detection(
		self,
		mask_size : _size_2_t | None = None,
		image_size: _size_2_t | None = None,
	) -> "BBoxAnnotation":
		"""Return a :class:`BBoxAnnotation` object of this instance whose
		bounding bbox tightly encloses the polyline. If a :param:`mask_size` is
		provided, an instance mask of the specified size encoding the polyline
		shape is included.
	 
		Alternatively, if a :param:`frame_size` is provided, the required mask
		size is then computed based off the polyline points and
		:param:`frame_size`.
		
		Args:
			mask_size: An optional shape at which to render an instance mask
				for the polyline.
			image_size: Used when no :param:`mask_size` is provided. An optional
				shape of the frame containing this polyline that's used to
				compute the required :param:`mask_size`.
		
		Return:
			A :class:`BBoxAnnotation` object.
		"""
		pass
	
	def to_segmentation(
		self,
		mask      : np.ndarray | None = None,
		image_size: _size_2_t  | None = None,
		target    : int               = 255,
		thickness : int               = 1,
	) -> SegmentationAnnotation:
		"""Return a :class:`SegmentationAnnotation` object of this class. Only
		object with instance masks (i.e., their :param:`mask` attributes
		populated) will be rendered.
		
		Args:
			mask: An optional 2D integer numpy array to use as an initial mask
				to which to add this object. Default: ``None``.
			image_size: The shape of the segmentation mask to render. This
				parameter has no effect if a :param:`mask` is provided.
				Default: ``None``.
			target: The pixel value to use to render the object. If you want
				color mask, just pass in the :param:`id` attribute.
				Default: ``255``.
			thickness: The thickness, in pixels, at which to render (non-filled)
				polylines. Default: ``1``.
				
		Return:
			A :class:`SegmentationAnnotation` object.
		"""
		pass


class PolylinesAnnotation(list[PolylineAnnotation], Annotation):
	"""A list of polylines or polygon labels for multiple objects in an image.
	
	See Also: :class:`Annotation`.
	
	Args:
		seq: A list of :class:`PolylineAnnotation` objects.
	"""
	
	def __init__(self, seq: list[PolylineAnnotation | dict]):
		super().__init__(PolylineAnnotation.from_value(value=i) for i in seq)
	
	def __setitem__(self, index: int, item: PolylineAnnotation | dict):
		super().__setitem__(index, PolylineAnnotation.from_value(item))
	
	def insert(self, index: int, item: PolylineAnnotation | dict):
		super().insert(index, PolylineAnnotation.from_value(item))
	
	def append(self, item: PolylineAnnotation | dict):
		super().append(PolylineAnnotation.from_value(item))
	
	def extend(self, other: list[PolylineAnnotation | dict]):
		super().extend([PolylineAnnotation.from_value(item) for item in other])
	
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
	def points(self) -> list:
		return [i._points for i in self]
	
	def to_detections(
		self,
		mask_size : _size_2_t | None = None,
		image_size: _size_2_t | None = None,
	) -> DetectionsAnnotation:
		"""Return a :class:`BBoxesAnnotation` object of this instance whose
		bounding boxes tightly enclose the polylines. If a :param:`mask_size`
		is provided, an instance mask of the specified size encoding the
		polyline shape is included.
	 
		Alternatively, if a :param:`frame_size` is provided, the required mask
		size is then computed based off the polyline points and
		:param:`frame_size`.
		
		Args:
			mask_size: An optional shape at which to render an instance mask
				for the polyline.
			image_size: Used when no :param:`mask_size` is provided. An optional
				shape of the frame containing this polyline that is used to
				compute the required :param:`mask_size`.
		
		Return:
			A :class:`BBoxesAnnotation` object.
		"""
		pass
	
	def to_segmentation(
		self,
		mask      : np.ndarray | None = None,
		image_size: _size_2_t  | None = None,
		target    : int               = 255,
		thickness : int               = 1,
	) -> SegmentationAnnotation:
		"""Return a :class:`SegmentationAnnotation` object of this instance. Only
		polylines with instance masks (i.e., their :param:`mask` attributes
		populated) will be rendered.
		
		Args:
			mask: An optional 2D integer numpy array to use as an initial mask
				to which to add this object. Default: ``None``.
			image_size: The shape of the segmentation mask to render. This
				parameter has no effect if a :param:`mask` is provided. Defaults
				to None.
			target: The pixel value to use to render the object. If you want
				color mask, just pass in the :param:`id` attribute.
				Default: ``255``.
			thickness: The thickness, in pixels, at which to render (non-filled)
				polylines. Default: ``1``.
				
		Return:
			A :class:`SegmentationAnnotation` object.
		"""
		pass

# endregion


# region Heatmap Annotation

class HeatmapAnnotation(Annotation):
	"""A heatmap label in an image.
	
	See Also: :class:`Annotation`.
	
	Args:
		map: A 2D numpy array.
		range: An optional [min, max] range of the map's values. If None is
			provided, [0, 1] will be assumed if :param:`map` contains floating
			point values, and [0, 255] will be assumed if :param:`map` contains
			integer values.
	"""
	
	@property
	def data(self) -> list | None:
		raise NotImplementedError(f"This function has not been implemented!")

# endregion


# region Segmentation Annotation

class SegmentationAnnotation(Annotation):
	"""A semantic segmentation label in an image.
	
	See Also: :class:`Annotation`.
	
	Args:
		id_: The ID of the image. This can be an integer or a string. This
			attribute is useful for batch processing where you want to keep the
			objects in the correct frame sequence.
		name: The name of the image. Default: ``None``.
		path: The path to the image file. Default: ``None``.
		mask: The image with integer values encoding the semantic labels.
			Default: ``None``.
		load: If ``True``, the image will be loaded into memory when the object
			is created. Default: ``False``.
		cache: If ``True``, the image will be loaded into memory and kept there.
			Default: ``False``.
	"""
	
	def __init__(
		self,
		id_  : int               = uuid.uuid4().int,
		name : str        | None = None,
		path : core.Path  | None = None,
		mask : np.ndarray | None = None,
		load : bool              = False,
		cache: bool              = False,
		*args, **kwargs
	):
		super().__init__(*args, **kwargs)
		self._id             = id_
		self._image          = None
		self._keep_in_memory = cache
		
		self._path = core.Path(path) if path is not None else None
		if self.path is None or not self.path.is_image_file():
			raise ValueError(f":param:`path` must be a valid path to an image file, but got {path}.")
		
		if name is None:
			name = str(core.Path(path).name) if path.is_image_file() else f"{id_}"
		self._name = name
		self._stem = core.Path(self._name).stem
		
		if load and mask is None:
			mask = self.load()
		
		self._shape = core.get_image_shape(input=mask) if mask is not None else None
		
		if self._keep_in_memory:
			self._mask = mask
	
	def load(
		self,
		path : core.Path | None = None,
		cache: bool             = False,
	) -> np.ndarray:
		"""Load segmentation mask image into memory.
		
		Args:
			path: The path to the segmentation mask file. Default: ``None``.
			cache: If ``True``, the image will be loaded into memory and kept
				there. Default: ``False``.
			
		Return:
			Return image of shape :math:`[H, W, C]`.
		"""
		self._keep_in_memory = cache
		
		if path is not None:
			path = core.Path(path)
			if path.is_image_file():
				self._path = path
		
		self._path = core.Path(path) if path is not None else None
		if self.path is None or not self.path.is_image_file():
			raise ValueError(f":param:`path` must be a valid path to an image file, but got {path}.")
		
		mask = core.read_image(path=self.path, to_rgb=True, to_tensor=False, normalize=False)
		self._shape = core.get_image_shape(input=mask) if (mask is not None) else self._shape
		
		if self._keep_in_memory:
			self._mask = mask
		return mask
	
	@property
	def path(self) -> core.Path | None:
		return self._path
	
	@property
	def data(self) -> np.ndarray | None:
		if self._mask is None:
			return self.load()
		else:
			return self._mask
	
	@property
	def meta(self) -> dict:
		return {
			"id"   : self._id,
			"name" : self._name,
			"stem" : self._stem,
			"path" : self.path,
			"shape": self._shape,
			"hash" : self.path.stat().st_size if isinstance(self.path, core.Path) else None,
		}

# endregion

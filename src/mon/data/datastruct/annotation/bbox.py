#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements annotations that take the form of a bounding box."""

from __future__ import annotations

__all__ = [
	"BBoxAnnotation",
	"BBoxesAnnotation",
]

from abc import ABC

import numpy as np

from mon import core
from mon.data.datastruct.annotation import base

console = core.console


# region BBox Annotation

class BBoxAnnotation(base.Annotation):
	"""A bounding box annotation in an image. Usually, it has a bounding box and
	an instance segmentation mask.
	
	Args:
		class_id: A class ID of the bounding box. ``-1`` means unknown.
		bbox: A bounding box's coordinates of shape :math:`[4]`.
		confidence: A confidence in :math:`[0, 1]` for the detection.
			Default: ``1.0``.
		mask: Instance segmentation mask for the object within its bounding
			bbox, which should be a binary (0/1) 2D sequence or a binary integer
			tensor. Default: ``None``.
	
	See Also: :class:`mon.data.datastruct.annotation.base.Annotation`.
	"""
	
	def __init__(
		self,
		class_id  : int,
		bbox      : np.ndarray | list | tuple,
		mask      : np.ndarray | list | tuple | None = None,
		confidence: float = 1.0,
		*args, **kwargs
	):
		super().__init__(*args, **kwargs)
		self.class_id   = class_id
		self.bbox       = bbox
		self.mask       = mask
		self.confidence = confidence
	
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
	def bbox(self) -> np.ndarray:
		"""Return the bounding box of shape :math:`[4]`."""
		return self._bbox
	
	@bbox.setter
	def bbox(self, bbox: np.ndarray | list | tuple):
		bbox = np.ndarray(bbox) if not isinstance(bbox, np.ndarray) else bbox
		if bbox.ndim == 1 and bbox.size == 4:
			self._bbox = bbox
		else:
			raise ValueError(f":param:`bbox` must be a 1D array of size 4, but got {bbox.ndim} and {bbox.size}.")
	
	@property
	def confidence(self) -> float:
		"""The confidence of the bounding box."""
		return self._confidence
	
	@confidence.setter
	def confidence(self, confidence: float):
		if not 0.0 <= confidence <= 1.0:
			raise ValueError(f":param:`confidence` must be between ``0.0`` and ``1.0``, but got {confidence}.")
		self._confidence = confidence
	
	@property
	def data(self) -> list | None:
		"""The label's data."""
		return [
			self.bbox[0],
			self.bbox[1],
			self.bbox[2],
			self.bbox[3],
			self.confidence,
			self.class_id,
		]
	

class BBoxesAnnotation(base.Annotation, ABC):
	"""A list of all bounding box annotations in an image.
	
	Args:
	
	See Also: :class:`mon.data.datastruct.annotation.base.Annotation`.
	"""
	
	def __init__(self):
		super().__init__()
		self.annotations: list[BBoxAnnotation] = []
	
	@property
	def data(self) -> list | None:
		return [i.data for i in self.annotations]
	
	@property
	def class_ids(self) -> list[int]:
		return [i.class_id for i in self.annotations]
	
	@property
	def bboxes(self) -> list:
		return [i.bbox for i in self.annotations]

# endregion

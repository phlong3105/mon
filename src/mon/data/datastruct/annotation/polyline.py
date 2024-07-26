#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements annotations that take the form of a polyline."""

from __future__ import annotations

__all__ = [
	"PolylineAnnotation",
	"PolylinesAnnotation",
]

import numpy as np

from mon import core
from mon.core import _size_2_t
from mon.data.datastruct.annotation import base

console = core.console


# region Polyline Annotation

class PolylineAnnotation(base.Annotation):
	"""A set of semantically related polylines or polygons for a single object
	in an image.
	
	See Also: :class:`mon.data.datastruct.annotation.base.Annotation`.
	
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
	) -> "SegmentationAnnotation":
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


class PolylinesAnnotation(list[PolylineAnnotation], base.Annotation):
	"""A list of polylines or polygon labels for multiple objects in an image.
	
	See Also: :class:`mon.data.datastruct.annotation.base.Annotation`.
	
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
	) -> "DetectionsAnnotation":
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
	) -> "SegmentationAnnotation":
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

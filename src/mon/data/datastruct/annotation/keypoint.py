#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements annotations that take the form of a keypoint."""

from __future__ import annotations

__all__ = [
	"KeypointAnnotation",
	"KeypointsAnnotation",
	"KeypointsLabelCOCO",
]

from mon import core
from mon.data.datastruct.annotation import base

console = core.console


# region Keypoint Annotation

class KeypointAnnotation(base.Annotation):
	"""A list of keypoints annotation for a single object in an image.
	
	See Also: :class:`mon.data.datastruct.annotation.base.Annotation`.
	
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


class KeypointsAnnotation(list[KeypointAnnotation], base.Annotation):
	"""A list of keypoint labels for multiple objects in an image.
	
	See Also: :class:`mon.data.datastruct.annotation.base.Annotation`.
	
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

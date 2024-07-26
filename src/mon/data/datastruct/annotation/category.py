#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements annotations that take the form of a category or class."""

from __future__ import annotations

__all__ = [
	"ClassificationAnnotation",
	"ClassificationsAnnotation",
]

import numpy as np

from mon import core
from mon.data.datastruct.annotation import base

console = core.console


# region Classification Annotation

class ClassificationAnnotation(base.Annotation):
	"""A classification annotation for an image.
	
	See Also: :class:`mon.data.datastruct.annotation.base.Annotation`.
	
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


class ClassificationsAnnotation(list[ClassificationAnnotation], base.Annotation):
	"""A list of classification labels for an image. It is used for multi-labels
	or multi-classes classification tasks.
	
	See Also: :class:`mon.data.datastruct.annotation.base.Annotation`.
	
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

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements annotations that take the form of a category or class."""

from __future__ import annotations

__all__ = [
	"ClassLabel",
	"ClassLabels",
	"ClassificationAnnotation",
	"ClassificationsAnnotation",
]

from typing import Any

import cv2
import numpy as np
import torch

from mon import core
from mon.data.datastruct.annotation import base

console = core.console


# region ClassLabel

class ClassLabel(dict, base.Annotation):
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

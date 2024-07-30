#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements classlabels in a dataset."""

from __future__ import annotations

__all__ = [
	"ClassLabel",
	"ClassLabels",
]

from typing import Any

from mon import core
from mon.core.file import json

console = core.console


# region ClassLabel

ClassLabel = dict


class ClassLabels(dict[str, ClassLabel]):
	"""A :class:`dict` of all the class-labels defined in a dataset.
	
	Notes:
		We inherit the standard Python :class:`dict` to take advantage of the
		built-in functions.
	"""
	
	@classmethod
	def from_file(cls, path: core.Path) -> ClassLabels:
		"""Create a :class:`ClassLabels` object from the content of a ``.json``
		file specified by the :param:`path`.
		"""
		path = core.Path(path)
		if not path.is_json_file():
			raise ValueError(f":param:`path` must be a ``.json`` file, but got {path}.")
		with open(path, "r") as file:
			return json.load(file)
	
	@classmethod
	def from_value(cls, value: Any) -> ClassLabels | None:
		"""Create a :class:`ClassLabels` object from a value."""
		if (
			isinstance(value, dict)
			and all(isinstance(k, str) and isinstance(v, dict) for k, v in value.items())
		):
			return value
		elif isinstance(value, list):
			return {v["name"]: v for i, v in enumerate(value)}
		elif isinstance(value, str | core.Path):
			return cls.from_file(value)
		else:
			return None
	
	@property
	def id2label(self) -> dict[int, ClassLabel]:
		"""A :class:`dict` mapping items' IDs (keys) to items (values)."""
		return {int(label["id"]): label for key, label in self.items()}
	
	@property
	def id2name(self) -> dict[int, str]:
		"""A :class:`dict` mapping items' IDs (keys) to items (values)."""
		return {int(label["id"]): label["name"] for key, label in self.items()}
	
	@property
	def num_classes(self) -> int:
		"""Return the number of classes in the dataset."""
		return len(self)
	
	@property
	def num_trainable_classes(self) -> int:
		"""Return the number of trainable classes in the dataset."""
		count = 0
		for k, v in self.items():
			id_ = v.get("id", None)
			if (id_ is not None) and (id_ >= 0):
				count += 1
		return count
	
	def print(self):
		"""Print all items (class-labels) in a rich format."""
		if len(self) <= 0:
			console.log("[yellow]No class is available.")
			return
		console.log("Classlabels:")
		core.print_table(list(self.values()))

# endregion

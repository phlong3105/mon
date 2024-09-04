#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""ClassLabel Annotation.

This module implements classlabels in a dataset.
"""

from __future__ import annotations

__all__ = [
	"ClassLabels",
]

from mon.core.rich import console, print_table


# region ClassLabel


class ClassLabels(list[dict]):
	"""A :obj:`list` of all the class-labels defined in a dataset.
	
	Notes:
		We inherit the standard Python :obj:`list` to take advantage of the
		built-in functions.
	"""
	
	@property
	def trainable_classes(self) -> ClassLabels:
		"""Return all the trainable classes."""
		return [i for i in self if 0 <= i["train_id"] < 255]
	
	@property
	def keys(self) -> list[str]:
		"""Return all the keys in the class-labels."""
		return list(self[0].keys())
	
	@property
	def names(self) -> list[str]:
		"""Return all the names in the class-labels."""
		return [item["name"] for item in self]
	
	@property
	def ids(self) -> list[int]:
		"""Return all the IDs in the class-labels."""
		return [item["id"] for item in self]
	
	@property
	def id2class(self) -> dict[int, dict]:
		"""A :obj:`dict` mapping items' IDs (keys) to items (values)."""
		return {int(item["id"]): item for item in self}
	
	@property
	def id2name(self) -> dict[int, str]:
		"""A :obj:`dict` mapping items' IDs (keys) to items (values)."""
		return {int(item["id"]): item["name"] for item in self}
	
	@property
	def id2color(self) -> dict[int, list[int] | tuple[int, int, int]]:
		"""A :obj:`dict` mapping items' IDs (keys) to items (values)."""
		return {int(item["id"]): item["color"] for item in self}
	
	@property
	def num_classes(self) -> int:
		"""Return the number of classes in the dataset."""
		return len(self)
	
	def print(self):
		"""Print all items (class-labels) in a rich format."""
		if len(self) <= 0:
			console.log("[yellow]No class is available.")
			return
		console.log("Classlabels:")
		print_table(self)

# endregion

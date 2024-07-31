#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements annotations that take the form of a value (number,
boolean, etc.).
"""

from __future__ import annotations

__all__ = [
	"RegressionAnnotation",
]

from mon import core
from mon.data.datastruct.annotation import base

console = core.console


# region Regression

class RegressionAnnotation(base.Annotation):
	"""A single regression value.
	
	Args:
		value: The regression value.
		confidence: A confidence in :math:`[0, 1]` for the regression.
			Default: ``1.0``.
	
	See Also: :class:`mon.data.datastruct.annotation.base.Annotation`.
	"""
	
	def __init__(
		self,
		value     : float,
		confidence: float = 1.0,
		*args, **kwargs
	):
		super().__init__(*args, **kwargs)
		self.value      = value
		self.confidence = confidence
	
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
		return [self.value]

# endregion

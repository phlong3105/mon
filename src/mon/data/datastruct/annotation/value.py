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


# region Regression Annotation

class RegressionAnnotation(base.Annotation):
	"""A single regression value.
	
	See Also: :class:`mon.data.datastruct.annotation.base.Annotation`.
	
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

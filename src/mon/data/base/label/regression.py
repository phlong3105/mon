#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements regression labels."""

from __future__ import annotations

__all__ = [
    "RegressionLabel",
]

from mon import core
from mon.data.base.label import base

console = core.console


# region Regression

class RegressionLabel(base.Label):
    """A single regression value.
    
    See Also: :class:`mon.data.base.label.base.Label`.
    
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
            raise ValueError(
                f":param:`conf` must be between ``0.0`` and ``1.0``, "
                f"but got {confidence}."
            )
        self.value      = value
        self.confidence = confidence
    
    @property
    def data(self) -> list | None:
        """The label's data."""
        return [self.value]
    
# endregion

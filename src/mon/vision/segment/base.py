#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the base class for segmentation models."""

from __future__ import annotations

__all__ = [
    "SegmentationModel",
]

from abc import ABC

from mon import core, nn
from mon.globals import ZOO_DIR

console = core.console


# region Model

class SegmentationModel(nn.Model, ABC):
    """The base class for all segmentation models.
    
    See Also: :class:`nn.Model`.
    """
    
    @property
    def zoo_dir(self) -> core.Path:
        return ZOO_DIR / "vision" / "segment" / self.name
    
# endregion

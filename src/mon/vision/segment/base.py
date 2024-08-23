#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Base Segmentation Model.

This module implements the base class for segmentation models.
"""

from __future__ import annotations

__all__ = [
    "SegmentationModel",
]

from abc import ABC

from mon import core, nn

console = core.console


# region Model

class SegmentationModel(nn.Model, ABC):
    """The base class for all segmentation models."""

    pass

# endregion

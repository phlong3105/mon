#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements heatmap labels."""

from __future__ import annotations

__all__ = [
    "HeatmapLabel",
]

from mon import core
from mon.data.base.label import base

console = core.console


# region Heatmap

class HeatmapLabel(base.Label):
    """A heatmap label in an image.
    
    See Also: :class:`Label`.
    
    Args:
        map: A 2D numpy array.
        range: An optional [min, max] range of the map's values. If None is
            provided, [0, 1] will be assumed if :param:`map` contains floating
            point values, and [0, 255] will be assumed if :param:`map` contains
            integer values.
    """

    @property
    def data(self) -> list | None:
        """The label's data."""
        raise NotImplementedError(f"This function has not been implemented!")

# endregion

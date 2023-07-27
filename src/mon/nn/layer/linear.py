#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements linearity layers."""

from __future__ import annotations

__all__ = [
    "Bilinear", "Identity", "LazyLinear", "Linear",
]

from torch import nn

from mon.globals import LAYERS
from mon.nn.layer import base


# region Linearity

@LAYERS.register()
class Bilinear(base.PassThroughLayerParsingMixin, nn.Bilinear):
    pass


@LAYERS.register()
class Identity(base.PassThroughLayerParsingMixin, nn.Identity):
    pass


@LAYERS.register()
class LazyLinear(base.PassThroughLayerParsingMixin, nn.LazyLinear):
    pass


@LAYERS.register()
class Linear(base.PassThroughLayerParsingMixin, nn.Linear):
    pass

# endregion

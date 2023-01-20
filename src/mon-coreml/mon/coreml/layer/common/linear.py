#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements linearity layers."""

from __future__ import annotations

__all__ = [
    "Bilinear", "Identity", "LazyLinear", "Linear",
]

from torch import nn

from mon.coreml import constant
from mon.coreml.layer import base


# region Linearity

@constant.LAYER.register()
class Bilinear(base.PassThroughLayerParsingMixin, nn.Bilinear):
    pass


@constant.LAYER.register()
class Identity(base.PassThroughLayerParsingMixin, nn.Identity):
    pass


@constant.LAYER.register()
class LazyLinear(base.PassThroughLayerParsingMixin, nn.LazyLinear):
    pass


@constant.LAYER.register()
class Linear(base.PassThroughLayerParsingMixin, nn.Linear):
    pass

# endregion

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements bottleneck layers."""

from __future__ import annotations

__all__ = [
    "Bottleneck",
]

import torchvision

from mon.globals import LAYERS
from mon.nn.layer import base


# region Bottleneck

@LAYERS.register()
class Bottleneck(base.PassThroughLayerParsingMixin, torchvision.models.resnet.Bottleneck):
    pass

# endregion

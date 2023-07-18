#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements bottleneck layers."""

from __future__ import annotations

__all__ = [
    "Bottleneck",
]

from typing import Any, Callable

import torch
import torchvision
from torch import nn

from mon.coreml.layer.base import activation, base, conv, normalization, pooling
from mon.coreml.layer.typing import _size_2_t
from mon.globals import LAYERS


# region Bottleneck

@LAYERS.register()
class Bottleneck(base.PassThroughLayerParsingMixin, torchvision.models.resnet.Bottleneck):
    pass

# endregion

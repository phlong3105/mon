#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements fold/unfold layers."""

from __future__ import annotations

__all__ = [
	"Fold",
	"Unfold",
]

from torch import nn

from mon.globals import LAYERS
from mon.nn.layer import base


# region Fold

@LAYERS.register()
class Fold(base.PassThroughLayerParsingMixin, nn.Fold):
	pass

# endregion


# region Unfold

@LAYERS.register()
class Unfold(base.PassThroughLayerParsingMixin, nn.Unfold):
	pass

# endregion

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements shuffling layers."""

from __future__ import annotations

__all__ = [
	"ChannelShuffle",
	"PixelShuffle",
	"PixelUnshuffle",
]

from torch import nn

from mon.globals import LAYERS
from mon.nn.layer import base


# region Channel Shuffle

@LAYERS.register()
class ChannelShuffle(base.PassThroughLayerParsingMixin, nn.ChannelShuffle):
	pass

# endregion


# region Pixel Shuffle

@LAYERS.register()
class PixelShuffle(base.PassThroughLayerParsingMixin, nn.PixelShuffle):
	pass


@LAYERS.register()
class PixelUnshuffle(base.PassThroughLayerParsingMixin, nn.PixelUnshuffle):
	pass

# endregion

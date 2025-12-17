#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A package for miscellaneous neural network (NN) components.

This package provides various miscellaneous modules and classes for building and
training neural networks (NNs) in deep learning applications.
"""

__all__ = [
    "AFF",
    "AlphaDropout",
    "ChannelShuffle",
    "DAF",
    "Dropout",
    "Dropout1d",
    "Dropout2d",
    "Dropout3d",
    "Embedding",
    "EmbeddingBag",
    "FeatureAlphaDropout",
    "Flatten",
    "Fold",
    "MS_CAM",
    "PixelShuffle",
    "PixelUnshuffle",
    "Unflatten",
    "Unfold",
    "Upsample",
    "UpsamplingBilinear2d",
    "UpsamplingNearest2d",
    "iAFF",
]

from .dropout import (
    AlphaDropout,
    Dropout,
    Dropout1d,
    Dropout2d,
    Dropout3d,
    FeatureAlphaDropout,
)
from .flatten import Flatten, Unflatten
from .fold import Fold, Unfold
from .fusion import AFF, DAF, iAFF, MS_CAM
from .shuffle import ChannelShuffle, PixelShuffle, PixelUnshuffle
from .sparse import Embedding, EmbeddingBag
from .upsampling import Upsample, UpsamplingBilinear2d, UpsamplingNearest2d

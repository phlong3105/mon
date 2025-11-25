#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements prior functions."""

__all__ = [
    "BoundaryAwarePrior",
    "BrightnessAttentionMap",
    "ImageLocalMean",
    "ImageLocalStdDev",
    "ImageLocalVariance",
    "atmospheric_point_spread_function",
    "atmospheric_prior",
    "blur_spot_prior",
    "boundary_aware_prior",
    "bright_channel_prior",
    "bright_spot_prior",
    "brightness_attention_map",
    "dark_channel_prior",
    "dark_channel_prior_paper",
    "image_local_mean",
    "image_local_stddev",
    "image_local_variance",
]

from .attention import brightness_attention_map, BrightnessAttentionMap
from .descriptive import (
    image_local_mean,
    image_local_stddev,
    image_local_variance,
    ImageLocalMean,
    ImageLocalStdDev,
    ImageLocalVariance,
)
from .feature import boundary_aware_prior, BoundaryAwarePrior
from .physical import atmospheric_point_spread_function, atmospheric_prior
from .statistical import (
    blur_spot_prior,
    bright_channel_prior,
    bright_spot_prior,
    dark_channel_prior,
    dark_channel_prior_paper,
)

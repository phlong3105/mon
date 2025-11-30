#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A package for image priors.

This package implements various image priors used in computer vision and image
processing tasks. These priors help in enhancing image quality, dehazing,
denoising, and other applications by leveraging statistical, physical, and
feature-based properties of images.
"""

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

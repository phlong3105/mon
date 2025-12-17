#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A package for image data type.

This package provides various classes and functions for handling image data,
including priors, filters, color transformations, and utility functions.
"""

__all__ = [
    "BoundaryAwarePrior",
    "BoxFilter",
    "BrightnessAttentionMap",
    "ConvGuidedFilter",
    "FastGuidedFilter",
    "GuidedFilter",
    "Image",
    "ImageLocalMean",
    "ImageLocalStdDev",
    "ImageLocalVariance",
    "RGBToHVI",
    "atmospheric_point_spread_function",
    "atmospheric_prior",
    "blur_spot_prior",
    "boundary_aware_prior",
    "box_filter",
    "bright_channel_prior",
    "bright_spot_prior",
    "brightness_attention_map",
    "center",
    "color_transfer",
    "dark_channel_prior",
    "dark_channel_prior_paper",
    "guided_filter",
    "image_local_mean",
    "image_local_stddev",
    "image_local_variance",
    "imgsz",
    "is_channel_first",
    "is_channel_last",
    "is_color",
    "is_grayscale",
    "is_image",
    "is_normalized",
    "load",
    "num_channels",
    "pad_square",
    "pair_downsample",
    "read_shape",
    "read_size",
    "save",
    "shape",
    "sobel_filter",
    "split",
    "to_array",
    "to_channel_first",
    "to_channel_last",
    "to_tensor",
]

from .color import color_transfer, RGBToHVI
from .core import Image
from .filtering import (
    box_filter,
    BoxFilter,
    ConvGuidedFilter,
    FastGuidedFilter,
    guided_filter,
    GuidedFilter,
    sobel_filter,
)
from .io import load, read_shape, read_size, save
from .priors import (
    atmospheric_point_spread_function,
    atmospheric_prior,
    blur_spot_prior,
    boundary_aware_prior,
    BoundaryAwarePrior,
    bright_channel_prior,
    bright_spot_prior,
    brightness_attention_map,
    BrightnessAttentionMap,
    dark_channel_prior,
    dark_channel_prior_paper,
    image_local_mean,
    image_local_stddev,
    image_local_variance,
    ImageLocalMean,
    ImageLocalStdDev,
    ImageLocalVariance,
)
from .processing import (
    pad_square,
    pair_downsample,
    split,
    to_array,
    to_channel_first,
    to_channel_last,
    to_tensor,
)
from .utils import (
    center,
    imgsz,
    is_channel_first,
    is_channel_last,
    is_color,
    is_grayscale,
    is_image,
    is_normalized,
    num_channels,
    shape,
)

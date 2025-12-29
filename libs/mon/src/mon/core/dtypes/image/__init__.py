#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Image data type.

This package contains a "full-stack" toolkit for image data, including data
structure, ingestion, analysis, atomic transformations, complex workflows, and
rendering utilities.

Notes:
    - Design Pattern: Toolkit Pattern.
    - Goal: Encapsulate related functionalities for a specific data type or
      domain within a single package.
    - Structure:
        ::

            toolkit/           # A "Toolkit" for a specific data type
            ├── __init__.py    # Exposes all
            ├── core.py        # Base classes and mixins
            ├── io.py          # Resource management
            ├── meta.py        # Discovery and lookup
            ├── ops.py         # Utility and algorithm
            ├── proc.py        # Workflow orchestration
            └── vis.py         # UI/UX rendering
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
    "apsf",
    "center",
    "color_transfer",
    "imgsz",
    "is_channel_first",
    "is_channel_last",
    "is_color",
    "is_grayscale",
    "is_image",
    "is_normalized",
    "num_channels",
    "pad_square",
    "pair_downsample",
    "read",
    "read_shape",
    "read_size",
    "shape",
    "sobel_filter",
    "split",
    "to_array",
    "to_channel_first",
    "to_channel_last",
    "to_tensor",
    "write",
]

from .color import color_transfer, RGBToHVI
from .core import Image
from .filtering import (
    BoxFilter,
    ConvGuidedFilter,
    FastGuidedFilter,
    GuidedFilter,
    sobel_filter,
)
from .io import read, read_shape, read_size, write
from .meta import *
from .ops import (
    center,
    imgsz,
    is_channel_first,
    is_channel_last,
    is_color,
    is_grayscale,
    is_image,
    is_normalized,
    num_channels,
    pad_square,
    pair_downsample,
    shape,
    split,
    to_array,
    to_channel_first,
    to_channel_last,
    to_tensor,
)
from .priors import (
    BoundaryAwarePrior,
    BrightnessAttentionMap,
    ImageLocalMean,
    ImageLocalStdDev,
    ImageLocalVariance,
    apsf,
)
from .proc import *
from .vis import *

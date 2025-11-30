#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A package for image filtering data type.

This package implements various image filtering techniques, including box filter,
guided filter, and Sobel filter. These filters are commonly used in image
processing tasks such as smoothing, edge detection, and detail enhancement.
"""

__all__ = [
    "BoxFilter",
    "ConvGuidedFilter",
    "FastGuidedFilter",
    "GuidedFilter",
    "box_filter",
    "guided_filter",
    "sobel_filter",
]

from .box_filter import box_filter, BoxFilter
from .guided_filter import (
    ConvGuidedFilter,
    FastGuidedFilter,
    guided_filter,
    GuidedFilter,
)
from .sobel_filter import sobel_filter

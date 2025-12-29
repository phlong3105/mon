#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Concrete data container implementations.

This package contains concrete implementations of data containers. Each sub-package
is a template method pattern for a specific type of data container (e.g., datasets,
dataloaders, etc.).
"""

__all__ = [
    "ImageDataset",
    "ImageEvalDataset",
    "ImageLoader",
    "VideoLoader",
    "is_video_dataset",
]

# from .datapools import *
from .datasets import (
    ImageDataset,
    ImageEvalDataset,
    ImageLoader,
    is_video_dataset,
    VideoLoader,
)

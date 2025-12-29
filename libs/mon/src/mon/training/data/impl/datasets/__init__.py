#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Training datasets.

This package provides various dataset classes and utilities for loading and
managing datasets, including image and video datasets. It includes base classes
for datasets, as well as specific implementations for image datasets with
evaluation capabilities and video datasets using OpenCV.
"""

__all__ = [
    "ImageDataset",
    "ImageEvalDataset",
    "ImageLoader",
    "VideoLoader",
    "is_video_dataset",
]

from .eval import ImageEvalDataset
from .image import ImageDataset, ImageLoader
from .video import is_video_dataset, VideoLoader

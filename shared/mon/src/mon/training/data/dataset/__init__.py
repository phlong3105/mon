#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A package for dataset handling.

This package provides various dataset classes and utilities for loading and
managing datasets, including image and video datasets. It includes base classes
for datasets, as well as specific implementations for image datasets with
evaluation capabilities and video datasets using OpenCV.
"""

__all__ = [
    "DataLoaderMixin",
    "Dataset",
    "DatasetLoadingMixin",
    "DatasetMetadataMixin",
    "DatasetMultimodalLoadingMixin",
    "ImageDataset",
    "ImageEvalDataset",
    "ImageLoader",
    "Modalities",
    "Modality",
    "VideoLoaderCV",
    "is_video_dataset",
]

from .base import (
    DataLoaderMixin,
    Dataset,
    DatasetLoadingMixin,
    DatasetMetadataMixin,
    DatasetMultimodalLoadingMixin,
    Modalities,
    Modality,
)
from .eval import ImageEvalDataset
from .image import ImageDataset, ImageLoader
from .video import is_video_dataset, VideoLoaderCV

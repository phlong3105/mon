#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A package for training data handling.

This package provides classes and functions for managing datasets, data loaders,
and data pools for training machine learning models. It supports various data
modalities including images and videos, and includes utilities for building
data loaders and datasets.
"""

__all__ = [
    "Classes",
    "DataLoader",
    "DataLoaderMixin",
    "DataPool",
    "Dataset",
    "DatasetLoadingMixin",
    "DatasetMetadataMixin",
    "DatasetMixin",
    "DatasetMultimodalLoadingMixin",
    "ImageDataPool",
    "ImageDataset",
    "ImageEvalDataset",
    "ImageLoader",
    "Modalities",
    "Modality",
    "SAMInstanceMixin",
    "VideoLoaderCV",
    "build_dataloader",
    "build_dataset",
    "is_video_dataset",
    "parse_data_dir",
]

from .builder import build_dataloader, build_dataset, parse_data_dir
from .classes import Classes
from .dataloader import DataLoader
from .datapool import DataPool, ImageDataPool
from .dataset import (
    DataLoaderMixin,
    Dataset,
    DatasetLoadingMixin,
    DatasetMetadataMixin,
    DatasetMultimodalLoadingMixin,
    ImageDataset,
    ImageEvalDataset,
    ImageLoader,
    is_video_dataset,
    Modalities,
    Modality,
    VideoLoaderCV,
)
from .mixin import DatasetMixin, SAMInstanceMixin

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This package provides various dataset classes and utilities for handling
different types of data.
"""

__all__ = [
    "BaseDataset",
    "BaseDualDomainDataset",
    "BaseEvalDataset",
    "ImageEvalDataset",
    "ImageLoader",
    "Modalities",
    "Modality",
    "VideoLoader",
    "VideoLoaderCV",
    "ImageDataset",
    "ImageDualDomainDataset",
    "is_video_dataset",
]

from .base import (
    BaseDataset,
    BaseDualDomainDataset,
    BaseEvalDataset,
    Modalities,
    Modality,
)
from .image import (
    ImageDataset,
    ImageDualDomainDataset,
    ImageEvalDataset,
    ImageLoader,
)
from .video import is_video_dataset, VideoLoader, VideoLoaderCV

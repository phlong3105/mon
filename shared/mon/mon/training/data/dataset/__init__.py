#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This package provides various dataset classes and utilities for handling
different types of data.
"""

__all__ = [
    "ALBUMENTATIONS_TARGETS",
    "BaseDataset",
    "DualDomainDataset",
    "EvalDataset",
    "ImageEvalDataset",
    "ImageLoader",
    "Modalities",
    "Modality",
    "VideoLoader",
    "VideoLoaderCV",
    "VideoWriter",
    "VideoWriterCV",
    "VideoWriterFFmpeg",
    "VisionDataset",
    "VisionDualDomainDataset",
    "is_video_dataset",
]

from .base import (
    BaseDataset,
    DualDomainDataset,
    EvalDataset,
    Modalities,
    Modality,
)
from .image import (
    ImageEvalDataset,
    ImageLoader,
)
from .video import (
    is_video_dataset,
    VideoLoader,
    VideoLoaderCV,
    VideoWriter,
    VideoWriterCV,
    VideoWriterFFmpeg,
)
from .vision import (
    ALBUMENTATIONS_TARGETS,
    VisionDataset,
    VisionDualDomainDataset,
)

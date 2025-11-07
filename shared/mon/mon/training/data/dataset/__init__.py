#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This package provides various dataset classes and utilities for handling
different types of data.
"""

__all__ = [
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

from .base import *
from .image import *
from .video import *
from .vision import *

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Data.

This module implements base and template classes for defining datasets, data loaders.
It also includes utilities functions for building datasets and data loaders.
"""

__all__ = [
    "ALBUMENTATIONS_TARGETS",
    "BaseDataset",
    "Classes",
    "DataLoader",
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
    "build_dataloader",
    "build_dataset",
    "is_video_dataset",
    "parse_data_dir",
]

from .builder import build_dataloader, build_dataset, parse_data_dir
from .classes import Classes
from .dataloader import DataLoader
from .dataset import (
    ALBUMENTATIONS_TARGETS,
    BaseDataset,
    DualDomainDataset,
    EvalDataset,
    ImageEvalDataset,
    ImageLoader,
    is_video_dataset,
    Modalities,
    Modality,
    VideoLoader,
    VideoLoaderCV,
    VideoWriter,
    VideoWriterCV,
    VideoWriterFFmpeg,
    VisionDataset,
    VisionDualDomainDataset,
)

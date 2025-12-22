#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""AI data management.

Provide data management functionalities for training machine learning models.

Package structure:
    data/
    ├── __init__.py           # Unified API (exposes Dataset, DataLoader)
    ├── base.py               # Abstract base classes (The "Contract")
    ├── constants.py          # Enums, standard paths, default values
    ├── mixins/               # Capability modules (Registrable, DualPath)
    │   ├── __init__.py
    │   ├── ...
    ├── datasets/             # Concrete implementations (Image, Video)
    │   ├── __init__.py
    │   ├── ...
    ├── loading/              # Data movement logic
    │   ├── __init__.py
    │   ├── dataloader.py     # The DataLoader class
    └── registry.py           # Logic to list and switch between methods
"""

__all__ = [
    "DataLoader",
    "Dataset",
    "ImageDataset",
    "ImageEvalDataset",
    "ImageLoader",
    "SAMInstanceMixin",
    "VideoLoaderCV",
    "build_dataloader",
    "build_dataset",
    "is_video_dataset",
    "parse_data_dir",
]

from .base import Dataset
from .constants import Modalities, Modality
from .datasets import (
    DataLoaderMixin,
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
from .loading import DataLoader
from .mixins import DatasetMixin, SAMInstanceMixin
from .registry import build_dataloader, build_dataset, parse_data_dir

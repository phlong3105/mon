#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module for core dataset classes and types.

This module provides core dataset classes and types for handling various data
modalities, including images, depth maps, and infrared maps. It includes base
dataset classes, data loaders, and utilities for managing different data sources
and tasks.
"""

__all__ = [
    "BBoxList",
    "TensorOrArray",
    "Classes",
    "DATASETS",
    "DEPTH_SOURCE",
    "DataLoader",
    "Dataset",
    "DefaultDepthMap",
    "DefaultInfraredMap",
    "DepthMap",
    "DepthName",
    "DepthSource",
    "Frame",
    "INFRARED_SOURCE",
    "Image",
    "ImageDataset",
    "ImageEvalDataset",
    "ImageLoader",
    "InfraredMap",
    "InfraredName",
    "InfraredSource",
    "Instance",
    "Modalities",
    "Modality",
    "Probs",
    "SemanticMask",
    "Split",
    "Task",
    "VideoLoaderCV",
    "VideoWriter",
    "VideoWriterCV",
    "VideoWriterFFmpeg",
]

from functools import partial

from mon.core import (
    DATASETS,
    DEPTH_SOURCE,
    DepthSource,
    INFRARED_SOURCE,
    InfraredSource,
    Split,
    Task,
)
from mon.core.dtypes import (
    TensorOrArray,
    BBoxList,
    DepthMap,
    Frame,
    Image,
    InfraredMap,
    Instance,
    Probs,
    SemanticMask,
)
from mon.core.dtypes.video import (
    VideoWriter,
    VideoWriterCV,
    VideoWriterFFmpeg,
)
from mon.training.data import (
    Classes,
    DataLoader,
    Dataset,
    ImageDataset,
    ImageEvalDataset,
    ImageLoader,
    Modalities,
    Modality,
    VideoLoaderCV,
)

# Constants for convenience
DepthName          = f"{DEPTH_SOURCE.value}"
InfraredName       = f"{INFRARED_SOURCE.value}"
DefaultDepthMap    = partial(DepthMap,    source=DEPTH_SOURCE)
DefaultInfraredMap = partial(InfraredMap, source=INFRARED_SOURCE)

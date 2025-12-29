#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Datasets metadata operations.

This module provides metadata operations for building datasets.
"""

__all__ = [
    "BBox",
    "BBoxList",
    "Class",
    "ClassList",
    "DATASETS",
    "DEPTH_SOURCE",
    "Data",
    "DataLoadMixin",
    "DataLoader",
    "Dataset",
    "DefaultDepthMap",
    "DefaultInfraredMap",
    "DepthMap",
    "DepthName",
    "DepthSource",
    "DeviceManagementMixin",
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
    "Probabilities",
    "SemanticMask",
    "Split",
    "Task",
    "TensorOrArray",
    "VideoLoader",
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
    BBox,
    BBoxList,
    Class,
    ClassList,
    Data,
    DataLoadMixin,
    DepthMap,
    DeviceManagementMixin,
    Frame,
    Image,
    InfraredMap,
    Instance,
    Probabilities,
    SemanticMask,
    TensorOrArray,
    VideoWriter,
    VideoWriterCV,
    VideoWriterFFmpeg,
)
from mon.training.data import (
    DataLoader,
    Dataset,
    ImageDataset,
    ImageEvalDataset,
    ImageLoader,
    Modalities,
    Modality,
    VideoLoader,
)

# Constants for convenience
DepthName          = f"{DEPTH_SOURCE.value}"
InfraredName       = f"{INFRARED_SOURCE.value}"
DefaultDepthMap    = partial(DepthMap,    source=DEPTH_SOURCE)
DefaultInfraredMap = partial(InfraredMap, source=INFRARED_SOURCE)

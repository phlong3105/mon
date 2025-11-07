#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements core components for datasets."""

__all__ = [
    "BaseDataset",
    "BaseTensorOrArray",
    "Classes",
    "DATASETS",
    "DEPTH_SOURCE",
    "DataLoader",
    "DefaultDepthMap",
    "DefaultInfraredMap",
    "DepthMap",
    "DepthName",
    "DepthSource",
    "Frame",
    "HBBs",
    "INFRARED_SOURCE",
    "Image",
    "ImageLoader",
    "InfraredMap",
    "InfraredName",
    "InfraredSource",
    "Modalities",
    "Modality",
    "Probs",
    "SemanticMask",
    "Split",
    "Task",
    "VideoLoader",
    "VideoLoaderCV",
    "VideoWriter",
    "VideoWriterCV",
    "VideoWriterFFmpeg",
    "VisionDataset",
]

from functools import partial

from mon.constants import DATASETS, DEPTH_SOURCE, INFRARED_SOURCE
from mon.core.dtypes import (
    BaseTensorOrArray,
    DepthMap,
    Frame,
    HBBs,
    Image,
    InfraredMap,
    Probs,
    SemanticMask,

)
from mon.core.enum import DepthSource, InfraredSource, Split, Task
from mon.training.data import (
    BaseDataset,
    Classes,
    DataLoader,
    ImageLoader,
    Modalities,
    Modality,
    VideoLoader,
    VideoLoaderCV,
    VideoWriter,
    VideoWriterCV,
    VideoWriterFFmpeg,
    VisionDataset,
)

# Constants for convenience
DepthName          = f"{DEPTH_SOURCE.value}"
InfraredName       = f"{INFRARED_SOURCE.value}"
DefaultDepthMap    = partial(DepthMap,    source=DEPTH_SOURCE)
DefaultInfraredMap = partial(InfraredMap, source=INFRARED_SOURCE)

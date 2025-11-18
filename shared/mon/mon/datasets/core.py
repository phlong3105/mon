#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements core components for datasets."""

__all__ = [
    "ALBUMENTATIONS_TARGETS",
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
    "DualDomainDataset",
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
    "VisionDualDomainDataset",
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
    BaseTensorOrArray,
    DepthMap,
    Frame,
    HBBs,
    Image,
    InfraredMap,
    Probs,
    SemanticMask,
)
from mon.training.data import (
    ALBUMENTATIONS_TARGETS,
    BaseDataset,
    Classes,
    DataLoader,
    DualDomainDataset,
    ImageLoader,
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

# Constants for convenience
DepthName          = f"{DEPTH_SOURCE.value}"
InfraredName       = f"{INFRARED_SOURCE.value}"
DefaultDepthMap    = partial(DepthMap,    source=DEPTH_SOURCE)
DefaultInfraredMap = partial(InfraredMap, source=INFRARED_SOURCE)

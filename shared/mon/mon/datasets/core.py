#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements core components for datasets."""

__all__ = [
    "BaseDataset",
    "BaseDualDomainDataset",
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
    "ImageDataset",
    "ImageDualDomainDataset",
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
from mon.core.dtypes.video import (
    VideoWriter,
    VideoWriterCV,
    VideoWriterFFmpeg,
)
from mon.training.data import (
    BaseDataset,
    BaseDualDomainDataset,
    Classes,
    DataLoader,
    ImageDataset,
    ImageDualDomainDataset,
    ImageLoader,
    Modalities,
    Modality,
    VideoLoader,
    VideoLoaderCV,
)

# Constants for convenience
DepthName          = f"{DEPTH_SOURCE.value}"
InfraredName       = f"{INFRARED_SOURCE.value}"
DefaultDepthMap    = partial(DepthMap,    source=DEPTH_SOURCE)
DefaultInfraredMap = partial(InfraredMap, source=INFRARED_SOURCE)

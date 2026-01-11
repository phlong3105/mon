#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Datasets API.

This module collects all classes and functions that are commonly used in
datasets. It is intended to be imported by other dataset modules for convenience.
"""

__all__ = [
    "BBox",
    "BBoxList",
    "Class",
    "ClassList",
    "DATASETS",
    "DEPTH_SOURCE",
    "Data",
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
    "PersistentData",
    "Probabilities",
    "RegistrableMixin",
    "SemanticMask",
    "Split",
    "Task",
    "TensorOrArray",
    "VideoLoader",
    "VideoWriter",
    "VideoWriterCV",
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
    DepthMap,
    DeviceManagementMixin,
    Frame,
    Image,
    InfraredMap,
    Instance,
    PersistentData,
    Probabilities,
    SemanticMask,
    TensorOrArray,
    VideoWriter,
    VideoWriterCV,
)
from mon.training.data import (
    DataLoader,
    Dataset,
    ImageDataset,
    ImageEvalDataset,
    ImageLoader,
    Modalities,
    Modality,
    RegistrableMixin,
    VideoLoader,
)

# Constants for convenience
DepthName          = f"{DEPTH_SOURCE.value}"
InfraredName       = f"{INFRARED_SOURCE.value}"
DefaultDepthMap    = partial(DepthMap,    source=DEPTH_SOURCE)
DefaultInfraredMap = partial(InfraredMap, source=INFRARED_SOURCE)

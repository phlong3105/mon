#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Datasets API.

This module collects all classes and functions that are commonly used in
datasets. It is intended to be imported by other dataset modules for
convenience.
"""

from __future__ import annotations

__all__ = [
    "BBox",
    "BBoxList",
    "Class",
    "ClassList",
    "DATASETS",
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
    "SOURCE",
    "SemanticMask",
    "Split",
    "Task",
    "TensorOrArray",
    "VideoLoader",
    "VideoWriter",
    "VideoWriterCV",
    "create_download_bar",
    "create_progress_bar",
]

from functools import partial

from mon.core import (
    create_download_bar,
    create_progress_bar,
    DATASETS,
    DepthSource,
    InfraredSource,
    SOURCE,
    Split,
    Task,
)
from mon.core.types import (
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


# ==============================================================================
# region CONSTANTS
# ==============================================================================

# Constants for convenience
DepthName = f"{SOURCE.DEPTH.value}"
InfraredName = f"{SOURCE.INFRARED.value}"
DefaultDepthMap = partial(DepthMap, source=SOURCE.DEPTH)
DefaultInfraredMap = partial(InfraredMap, source=SOURCE.INFRARED)

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""DarkFace Datasets.

This module provides the DarkFace dataset for nighttime face enhancement and
detection.
"""

from __future__ import annotations

__all__ = [
    "DarkFace",
]

from mon.core import Class, ClassList, DATASETS, Split, Task
from mon.dataset.base import (
    DatasetRegisterMixin,
    DepthModality,
    ImageDataset,
    ImageModality,
    ModalityList,
)


# ==============================================================================
# region DATASETS
# ==============================================================================

@DATASETS.register(name="darkface")
class DarkFace(ImageDataset, DatasetRegisterMixin):
    """DarkFace dataset."""

    name: str = "darkface"
    tasks: list[Task] = [Task.LLE, Task.DETECT]
    dirname: str = "darkface"
    subdir: str = "darkface"
    splits: list[Split] = [Split.TEST]
    modalities: ModalityList = ModalityList([
        ImageModality(name="image", dirname="images"),
        DepthModality(name="depth", dirname="depth"),
    ])
    classes: ClassList = ClassList([
        Class(name="face", id=0, color=(81, 120, 228)),
    ])

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion

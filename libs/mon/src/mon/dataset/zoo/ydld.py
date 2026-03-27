#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""YDLD Datasets.

This module provides the YDLD (YouTube Driving Light Detection) dataset for
nighttime light detection and enhancement.
"""

from __future__ import annotations

__all__ = [
    "YDLD",
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

@DATASETS.register(name="ydld")
class YDLD(ImageDataset, DatasetRegisterMixin):
    """YDLD dataset."""

    name: str = "ydld"
    tasks: list[Task] = [Task.LLIE, Task.DETECT]
    dirname: str = "ydld"
    subdir: str = "ydld"
    splits: list[Split] = [Split.TRAIN, Split.TEST]
    modalities: ModalityList = ModalityList([
        ImageModality(name="image", dirname="image"),
        DepthModality(name="depth", dirname="depth"),
    ])
    classes: ClassList = ClassList([
        Class(name="car_light",            id=0, color=(255, 0, 0)),
        Class(name="traffic_signal_light", id=1, color=(0, 128, 0)),
        Class(name="street_light",         id=2, color=(0, 0, 255)),
    ])

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion

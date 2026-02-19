#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Speed Datasets.

This module provides dataset classes for the Speed10 and Speed1K datasets,
which are designed for benchmarking performance.
"""

from __future__ import annotations

__all__ = [
    "Speed10",
    "Speed1K",
]

from mon.core import ClassList, DATASETS, Split, Task
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

@DATASETS.register(name="speed10")
class Speed10(ImageDataset, DatasetRegisterMixin):
    """Speed10 dataset."""

    name: str = "speed10"
    tasks: list[Task] = [Task.BENCHMARK]
    dirname: str = "speed10"
    subdir: str = ""
    splits: list[Split] = [Split.TEST]
    modalities: ModalityList = ModalityList([
        ImageModality(name="image", dirname="image"),
        DepthModality(name="depth", dirname="depth"),
    ])
    classes: ClassList = ClassList()


@DATASETS.register(name="speed1k")
class Speed1K(ImageDataset, DatasetRegisterMixin):
    """Speed1K dataset."""

    name: str = "speed1k"
    tasks: list[Task] = [Task.BENCHMARK]
    dirname: str = "speed1k"
    subdir: str = ""
    splits: list[Split] = [Split.TEST]
    modalities: ModalityList = ModalityList([
        ImageModality(name="image", dirname="image"),
        DepthModality(name="depth", dirname="depth"),
    ])
    classes: ClassList = ClassList()

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion

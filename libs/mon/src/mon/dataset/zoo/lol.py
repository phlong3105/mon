#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""LOL Datasets.

This module provides the LOL-v1 and LOL-v2 datasets for low-light image
enhancement tasks.

The LOL-v1 dataset consists of paired low-light and normal-light images, while
the LOL-v2 dataset includes additional unpaired images for more challenging
scenarios. Both datasets are widely used benchmarks for evaluating low-light
image enhancement algorithms.
"""

from __future__ import annotations

__all__ = [
    "LOLv1",
    "LOLv2Real",
    "LOLv2Syn",
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

@DATASETS.register(name="lol_v1")
class LOLv1(ImageDataset, DatasetRegisterMixin):
    """LOL-v1 dataset."""

    name: str = "lol_v1"
    tasks: list[Task] = [Task.LLE]
    dirname: str = "lol_v1"
    subdir: str = ""
    splits: list[Split] = [Split.TRAIN, Split.TEST]
    modalities: ModalityList = ModalityList([
        ImageModality(name="image", dirname="images"),
        DepthModality(name="depth", dirname="depth"),
        ImageModality(name="target", dirname="targets"),
    ])
    classes: ClassList = ClassList()


@DATASETS.register(name="lol_v2_real")
class LOLv2Real(ImageDataset, DatasetRegisterMixin):
    """LOL-v2-Real dataset."""

    name: str = "lol_v2_real"
    tasks: list[Task] = [Task.LLE]
    dirname: str = "lol_v2"
    subdir: str = "real"
    splits: list[Split] = [Split.TRAIN, Split.TEST]
    modalities: ModalityList = ModalityList([
        ImageModality(name="image", dirname="images"),
        DepthModality(name="depth", dirname="depth"),
        ImageModality(name="target", dirname="targets"),
    ])
    classes: ClassList = ClassList()


@DATASETS.register(name="lol_v2_syn")
class LOLv2Syn(ImageDataset, DatasetRegisterMixin):
    """LOL-v2-Synthetic dataset."""

    name: str = "lol_v2_real"
    tasks: list[Task] = [Task.LLE]
    dirname: str = "lol_v2"
    subdir: str = "syn"
    splits: list[Split] = [Split.TRAIN, Split.TEST]
    modalities: ModalityList = ModalityList([
        ImageModality(name="image", dirname="images"),
        DepthModality(name="depth", dirname="depth"),
        ImageModality(name="target", dirname="targets"),
    ])
    classes: ClassList = ClassList()

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Unpaired LOL Datasets.

This module provides several sub-datasets usually gathered under the umbrella
of the "Unpaired" dataset, which are commonly used for low-light image
enhancement tasks.
"""

from __future__ import annotations

__all__ = [
    "DICM",
    "Fusion",
    "LIME",
    "MEF",
    "NPE",
    "VV",
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

@DATASETS.register(name="dicm")
class DICM(ImageDataset, DatasetRegisterMixin):
    """LOL-v1 dataset."""

    name: str = "dicm"
    tasks: list[Task] = [Task.LLE]
    dirname: str = "unpaired_lol"
    subdir: str = "dicm"
    splits: list[Split] = [Split.TEST]
    modalities: ModalityList = ModalityList([
        ImageModality(name="image", dirname="image"),
        DepthModality(name="depth", dirname="depth"),
    ])
    classes: ClassList = ClassList()


@DATASETS.register(name="fusion")
class Fusion(ImageDataset, DatasetRegisterMixin):
    """Fusion dataset."""

    name: str = "fusion"
    tasks: list[Task] = [Task.LLE]
    dirname: str = "unpaired_lol"
    subdir: str = "fusion"
    splits: list[Split] = [Split.TEST]
    modalities: ModalityList = ModalityList([
        ImageModality(name="image", dirname="image"),
        DepthModality(name="depth", dirname="depth"),
    ])
    classes: ClassList = ClassList()


@DATASETS.register(name="lime")
class LIME(ImageDataset, DatasetRegisterMixin):
    """LIME dataset."""

    name: str = "lime"
    tasks: list[Task] = [Task.LLE]
    dirname: str = "unpaired_lol"
    subdir: str = "lime"
    splits: list[Split] = [Split.TEST]
    modalities: ModalityList = ModalityList([
        ImageModality(name="image", dirname="image"),
        DepthModality(name="depth", dirname="depth"),
    ])
    classes: ClassList = ClassList()


@DATASETS.register(name="mef")
class MEF(ImageDataset, DatasetRegisterMixin):
    """MEF dataset."""

    name: str = "mef"
    tasks: list[Task] = [Task.LLE]
    dirname: str = "unpaired_lol"
    subdir: str = "mef"
    splits: list[Split] = [Split.TEST]
    modalities: ModalityList = ModalityList([
        ImageModality(name="image", dirname="image"),
        DepthModality(name="depth", dirname="depth"),
    ])
    classes: ClassList = ClassList()


@DATASETS.register(name="npe")
class NPE(ImageDataset, DatasetRegisterMixin):
    """NPE dataset."""

    name: str = "npe"
    tasks: list[Task] = [Task.LLE]
    dirname: str = "unpaired_lol"
    subdir: str = "npe"
    splits: list[Split] = [Split.TEST]
    modalities: ModalityList = ModalityList([
        ImageModality(name="image", dirname="image"),
        DepthModality(name="depth", dirname="depth"),
    ])
    classes: ClassList = ClassList()


@DATASETS.register(name="vv")
class VV(ImageDataset, DatasetRegisterMixin):
    """VV dataset."""

    name: str = "vv"
    tasks: list[Task] = [Task.LLE]
    dirname: str = "unpaired_lol"
    subdir: str = "vv"
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

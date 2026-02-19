#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""FiveK Datasets.

This module provides the FiveK dataset and its variants for image enhancement.
"""

from __future__ import annotations

__all__ = [
    "FiveK",
    "FiveK_A",
    "FiveK_B",
    "FiveK_C",
    "FiveK_D",
    "FiveK_E",
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

@DATASETS.register(name="fivek")
class FiveK(ImageDataset, DatasetRegisterMixin):
    """FiveK dataset."""

    name: str = "fivek"
    tasks: list[Task] = [Task.ENHANCE]
    dirname: str = "fivek"
    subdir: str = ""
    splits: list[Split] = [Split.TRAIN, Split.VAL, Split.TEST]
    modalities: ModalityList = ModalityList([
        ImageModality(name="image", dirname="image"),
        DepthModality(name="depth", dirname="depth"),
        ImageModality(name="target", dirname="target_c"),
        ImageModality(name="target_a", dirname="target_a"),
        ImageModality(name="target_b", dirname="target_b"),
        ImageModality(name="target_c", dirname="target_c"),
        ImageModality(name="target_d", dirname="target_d"),
        ImageModality(name="target_e", dirname="target_e"),
    ])
    classes: ClassList = ClassList()


@DATASETS.register(name="fivek_a")
class FiveK_A(ImageDataset, DatasetRegisterMixin):
    """FiveK-A dataset."""

    name: str = "fivek_a"
    tasks: list[Task] = [Task.ENHANCE]
    dirname: str = "fivek"
    subdir: str = ""
    splits: list[Split] = [Split.TRAIN, Split.VAL, Split.TEST]
    modalities: ModalityList = ModalityList([
        ImageModality(name="image", dirname="image"),
        DepthModality(name="depth", dirname="depth"),
        ImageModality(name="target", dirname="target_a"),
    ])
    classes: ClassList = ClassList()


@DATASETS.register(name="fivek_b")
class FiveK_B(ImageDataset, DatasetRegisterMixin):
    """FiveK-B dataset."""

    name: str = "fivek_b"
    tasks: list[Task] = [Task.ENHANCE]
    dirname: str = "fivek"
    subdir: str = ""
    splits: list[Split] = [Split.TRAIN, Split.VAL, Split.TEST]
    modalities: ModalityList = ModalityList([
        ImageModality(name="image", dirname="image"),
        DepthModality(name="depth", dirname="depth"),
        ImageModality(name="target", dirname="target_b"),
    ])
    classes: ClassList = ClassList()


@DATASETS.register(name="fivek_c")
class FiveK_C(ImageDataset, DatasetRegisterMixin):
    """FiveK-C dataset."""

    name: str = "fivek_c"
    tasks: list[Task] = [Task.ENHANCE]
    dirname: str = "fivek"
    subdir: str = ""
    splits: list[Split] = [Split.TRAIN, Split.VAL, Split.TEST]
    modalities: ModalityList = ModalityList([
        ImageModality(name="image", dirname="image"),
        DepthModality(name="depth", dirname="depth"),
        ImageModality(name="target", dirname="target_c"),
    ])
    classes: ClassList = ClassList()


@DATASETS.register(name="fivek_d")
class FiveK_D(ImageDataset, DatasetRegisterMixin):
    """FiveK-D dataset."""

    name: str = "fivek_d"
    tasks: list[Task] = [Task.ENHANCE]
    dirname: str = "fivek"
    subdir: str = ""
    splits: list[Split] = [Split.TRAIN, Split.VAL, Split.TEST]
    modalities: ModalityList = ModalityList([
        ImageModality(name="image", dirname="image"),
        DepthModality(name="depth", dirname="depth"),
        ImageModality(name="target", dirname="target_d"),
    ])
    classes: ClassList = ClassList()


@DATASETS.register(name="fivek_E")
class FiveK_E(ImageDataset, DatasetRegisterMixin):
    """FiveK-E dataset."""

    name: str = "fivek_e"
    tasks: list[Task] = [Task.ENHANCE]
    dirname: str = "fivek"
    subdir: str = ""
    splits: list[Split] = [Split.TRAIN, Split.VAL, Split.TEST]
    modalities: ModalityList = ModalityList([
        ImageModality(name="image", dirname="image"),
        DepthModality(name="depth", dirname="depth"),
        ImageModality(name="target", dirname="target_e"),
    ])
    classes: ClassList = ClassList()

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion

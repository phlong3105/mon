#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""UHD-LOL Datasets.

This module provides the UHD-LL dataset for low-light image enhancement.

References:
    - Data: https://github.com/TaoWangzj/LLFormerV2
"""

from __future__ import annotations

__all__ = [
    "UHD_LOL_4K",
    "UHD_LOL_4x4K",
    "UHD_LOL_8K",
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

@DATASETS.register(name="uhd_lol_4k")
class UHD_LOL_4K(ImageDataset, DatasetRegisterMixin):
    """UHD-LOL-4K dataset."""

    name: str = "uhd_lol_4k"
    tasks: list[Task] = [Task.LLE]
    dirname: str = "uhd_lol"
    subdir: str = "4k"
    splits: list[Split] = [Split.TRAIN, Split.TEST]
    modalities: ModalityList = ModalityList([
        ImageModality(name="image", dirname="images"),
        DepthModality(name="depth", dirname="depth"),
        ImageModality(name="target", dirname="targets"),
    ])
    classes: ClassList = ClassList()


@DATASETS.register(name="uhd_lol_8k")
class UHD_LOL_8K(ImageDataset, DatasetRegisterMixin):
    """UHD-LOL-8K dataset."""

    name: str = "uhd_lol_8k"
    tasks: list[Task] = [Task.LLE]
    dirname: str = "uhd_lol"
    subdir: str = "8k"
    splits: list[Split] = [Split.TRAIN, Split.TEST]
    modalities: ModalityList = ModalityList([
        ImageModality(name="image", dirname="images"),
        DepthModality(name="depth", dirname="depth"),
        ImageModality(name="target", dirname="targets"),
    ])
    classes: ClassList = ClassList()


@DATASETS.register(name="uhd_lol_4x4k")
class UHD_LOL_4x4K(ImageDataset, DatasetRegisterMixin):
    """UHD-LOL-8K Cropped dataset."""

    name: str = "uhd_lol_4x4k"
    tasks: list[Task] = [Task.LLE]
    dirname: str = "uhd_lol"
    subdir: str = "4x4k"
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

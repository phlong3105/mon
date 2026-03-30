#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""UHD-LL Datasets.

This module provides the UHD-LL dataset for low-light image enhancement.
"""

from __future__ import annotations

__all__ = [
    "UHD_LL",
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

@DATASETS.register(name="uhd_ll")
class UHD_LL(ImageDataset, DatasetRegisterMixin):
    """UHD-LL dataset."""

    name: str = "uhd_ll"
    tasks: list[Task] = [Task.LLE]
    dirname: str = "uhd_ll"
    subdir: str = ""
    splits: list[Split] = [Split.TRAIN, Split.TEST]
    modalities: ModalityList = ModalityList([
        ImageModality(name="image", dirname="image"),
        DepthModality(name="depth", dirname="depth"),
        ImageModality(name="target", dirname="target"),
    ])
    classes: ClassList = ClassList()

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""LSRW Datasets.

This module provides the LSRW dataset for low-light image enhancement and
denoising.
"""

from __future__ import annotations

__all__ = [
    "LSRW",
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

@DATASETS.register(name="lsrw")
class LSRW(ImageDataset, DatasetRegisterMixin):
    """LSRW dataset."""

    name: str = "lsrw"
    tasks: list[Task] = [Task.LLE]
    dirname: str = "lsrw"
    subdir: str = ""
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

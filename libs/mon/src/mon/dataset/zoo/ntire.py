#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""NTIRE Challenge Datasets.

This module provides the datasets used in the NTIRE Challenges.
"""

from __future__ import annotations

__all__ = [
    "NTIRE2025LLIE",
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
# region NTIRE 2025
# ==============================================================================

@DATASETS.register(name="ntire_2025_llie")
class NTIRE2025LLIE(ImageDataset, DatasetRegisterMixin):
    """NTIRE 2025 LLIE dataset."""

    name: str = "ntire_2025_llie"
    tasks: list[Task] = [Task.LLE]
    dirname: str = "ntire_2025_llie"
    subdir: str = ""
    splits: list[Split] = [Split.TRAIN, Split.VAL, Split.TEST]
    modalities: ModalityList = ModalityList([
        ImageModality(name="image", dirname="image"),
        DepthModality(name="depth", dirname="depth"),
        ImageModality(name="target", dirname="target", val=False, test=False),
    ])
    classes: ClassList = ClassList()

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion

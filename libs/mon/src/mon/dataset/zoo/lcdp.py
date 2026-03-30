#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""LCDP Datasets.

This module provides the Local Color Distributions Prior (LCDP) dataset for
image enhancement.

References:
    - Paper: "Local Color Distributions Prior for Image Enhancement," ECCV 2022.
    - Code: https://github.com/onpix/LCDPNet
"""

from __future__ import annotations

__all__ = [
    "LCDP",
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

@DATASETS.register(name="lcdp")
class LCDP(ImageDataset, DatasetRegisterMixin):
    """LCDP dataset."""

    name: str = "lcdp"
    tasks: list[Task] = [Task.LLE]
    dirname: str = "lcdp"
    subdir: str = "lcdp"
    splits: list[Split] = [Split.TRAIN, Split.VAL, Split.TEST]
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

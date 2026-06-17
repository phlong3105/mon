#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""MSEC Datasets.

This module provides the Multi-Scale Exposure Correction (MSEC) dataset for
exposure correction.

References:
    - Paper: "Learning Multi-Scale Photo Exposure Correction," CVPR 2021.
    - Code: https://github.com/mahmoudnafifi/Exposure_Correction
"""

from __future__ import annotations

__all__ = [
    "MSEC",
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

@DATASETS.register(name="msec")
class MSEC(ImageDataset, DatasetRegisterMixin):
    """MSEC dataset."""

    name: str = "msec"
    tasks: list[Task] = [Task.LLE]
    dirname: str = "msec"
    subdir: str = "msec"
    splits: list[Split] = [Split.TRAIN, Split.VAL, Split.TEST]
    modalities: ModalityList = ModalityList([
        ImageModality(name="image", dirname="images"),
        DepthModality(name="depth", dirname="depth"),
        ImageModality(name="target", dirname="targets", train=True, val=True, test=False),
        ImageModality(name="target_a", dirname="targets_a", train=False, val=False, test=True),
        ImageModality(name="target_b", dirname="targets_b", train=False, val=False, test=True),
        ImageModality(name="target_c", dirname="targets_c", train=False, val=False, test=True),
        ImageModality(name="target_d", dirname="targets_d", train=False, val=False, test=True),
        ImageModality(name="target_e", dirname="targets_e", train=False, val=False, test=True),
    ])
    classes: ClassList = ClassList()

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion

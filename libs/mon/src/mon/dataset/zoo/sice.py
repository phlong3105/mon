#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""SICE Datasets.

This module provides the SICE dataset and its variants (SICE-LR, SICE-ME) for
low-light image enhancement and exposure enhancement.

References:
    - Paper: "Learning a Deep Single Image Contrast Enhancer from Multi-Exposure
      Images," TIP 2018.
    - Code: https://github.com/csjcai/SICE

Notices:
    The testing index in Dataset_part1:
        - 4-23
        - 28
        - 31
        - 33-34
        - 37-39
        - 46-52
        - 55-69
        - 75-79
        - 100-103
    For the under-exposure testing, we choose the -1ev as the low-light input image:
        - If there are 7 images, then it is number 3.
        - If there are 9 images, then it is number 4.
    For the over-exposure testing, we choose the +1ev as the over-exposure input image:
        - If there are 7 images, then it is number 5.
        - If there are 9 images, then it is number 6. (My assumption)
"""

from __future__ import annotations

__all__ = [
    "SICE",
    "SICE_LR",
    "SICE_ME",
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

@DATASETS.register(name="sice")
class SICE(ImageDataset, DatasetRegisterMixin):
    """SICE dataset."""

    name: str = "sice"
    tasks: list[Task] = [Task.LLIE]
    dirname: str = "sice"
    subdir: str = "sice"
    splits: list[Split] = [Split.TRAIN, Split.TEST]
    modalities: ModalityList = ModalityList([
        ImageModality(name="image", dirname="image_under"),
        ImageModality(name="image_under", dirname="image_under"),
        ImageModality(name="image_over", dirname="image_over"),
        DepthModality(name="depth", dirname="depth"),
        ImageModality(name="target", dirname="target"),
    ])
    classes: ClassList = ClassList()


@DATASETS.register(name="sice_lr")
class SICE_LR(ImageDataset, DatasetRegisterMixin):
    """SICE-LR dataset."""

    name: str = "sice_lr"
    tasks: list[Task] = [Task.LLIE]
    dirname: str = "sice"
    subdir: str = "sice_lr"
    splits: list[Split] = [Split.TRAIN, Split.TEST]
    modalities: ModalityList = ModalityList([
        ImageModality(name="image", dirname="image_under"),
        ImageModality(name="image_under", dirname="image_under"),
        ImageModality(name="image_over", dirname="image_over"),
        DepthModality(name="depth", dirname="depth"),
        ImageModality(name="target", dirname="target"),
    ])
    classes: ClassList = ClassList()


@DATASETS.register(name="sice_me")
class SICE_ME(ImageDataset, DatasetRegisterMixin):
    """SICE-ME dataset.

    Include multi-exposure training images. This dataset is used in unsupervised
    curve-estimation methods for low-light enhancement (e.g., Zero-DCE).
    """

    name: str = "sice_me"
    tasks: list[Task] = [Task.LLIE]
    dirname: str = "sice"
    subdir: str = "me"
    splits: list[Split] = [Split.TRAIN, Split.VAL]
    modalities: ModalityList = ModalityList([
        ImageModality(name="image", dirname="image"),
        DepthModality(name="depth", dirname="depth"),
        ImageModality(name="target", dirname="target", train=False, val=True, test=False),
    ])
    classes: ClassList = ClassList()

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion

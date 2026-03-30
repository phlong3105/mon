#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""ExDark Datasets.

This module provides the ExDark dataset for nighttime object enhancement and
detection.

References:
    - Data: https://github.com/cs-chan/Exclusively-Dark-Image-Dataset
"""

from __future__ import annotations

__all__ = [
    "ExDark",
]

from mon.core import Class, ClassList, DATASETS, Split, Task
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

@DATASETS.register(name="exdark")
class ExDark(ImageDataset, DatasetRegisterMixin):
    """ExDark dataset."""

    name: str = "exdark"
    tasks: list[Task] = [Task.LLE, Task.DETECT]
    dirname: str = "exdark"
    subdir: str = "exdark"
    splits: list[Split] = [Split.TEST]
    modalities: ModalityList = ModalityList([
        ImageModality(name="image", dirname="image"),
        DepthModality(name="depth", dirname="depth"),
    ])
    classes: ClassList = ClassList([
        Class(name="Bicycle"   , id=1 , color=(138, 183, 33)),
        Class(name="Boat"      , id=2 , color=(19 , 64 , 83)),
        Class(name="Bottle"    , id=3 , color=(139, 160, 1)),
        Class(name="Bus"       , id=4 , color=(140, 24 , 143)),
        Class(name="Car"       , id=5 , color=(49 , 3  , 150)),
        Class(name="Cat"       , id=6 , color=(41 , 174, 251)),
        Class(name="Chair"     , id=7 , color=(94 , 173, 36)),
        Class(name="Cup"       , id=8 , color=(28 , 47 , 55)),
        Class(name="Dog"       , id=9 , color=(21 , 8  , 251)),
        Class(name="Motorcycle", id=10, color=(122, 35 , 2)),
        Class(name="People"    , id=11, color=(81 , 120, 228)),
        Class(name="Table"     , id=12, color=(216, 147, 179)),
    ])

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion

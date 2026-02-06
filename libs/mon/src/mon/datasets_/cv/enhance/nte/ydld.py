#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""YDLD dataset.

This module provides the YDLD (YouTube Driving Light Detection) dataset for
nighttime light detection and enhancement.
"""

from __future__ import annotations

__all__ = [
    "YDLD",
]

from ....api import *


# ==============================================================================
# region DATASETS
# ==============================================================================

@DATASETS.register(name="ydld")
class YDLD(ImageDataset, RegistrableMixin):
    """YDLD dataset."""

    name: str = "ydld"
    tasks: list[Task] = [Task.NTE, Task.LLE, Task.DETECT]
    subroot: str = None
    splits: list[Split] = [Split.TRAIN, Split.TEST]
    modalities: Modalities = {
        "image": Modality(
            name="image",
            type="image",
            module=Image,
            train=True,
            test=True,
            primary=True,
        ),
    }
    classlist: ClassList = ClassList(
        [
            {"name": "car_light", "id": 0, "color": (255, 0, 0)},
            {"name": "traffic_signal_light", "id": 1, "color": (0, 128, 0)},
            {"name": "street_light", "id": 2, "color": (0, 0, 255)},
        ],
    )


# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion

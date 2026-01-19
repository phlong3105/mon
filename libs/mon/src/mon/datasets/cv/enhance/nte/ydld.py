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


@DATASETS.register()
class YDLD(ImageDataset, RegistrableMixin):
    """YDLD dataset."""

    _name      : str         = "ydld"
    _tasks     : list[Task]  = [Task.NTE, Task.LLE, Task.DETECT]
    _subset    : str         = None
    _splits    : list[Split] = [Split.TRAIN, Split.TEST]
    _modalities: Modalities  = {
        "image": Modality(
            name    = "image",
            type    = "image",
            module  = Image,
            train   = True,
            test    = True,
            primary = True,
        ),
    }
    _classlist : ClassList   = ClassList([
        {"name": "car_light",            "id": 0, "color": (255,   0,   0)},
        {"name": "traffic_signal_light", "id": 1, "color": (0  , 128,   0)},
        {"name": "street_light",         "id": 2, "color": (0  ,   0, 255)},
    ])


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion

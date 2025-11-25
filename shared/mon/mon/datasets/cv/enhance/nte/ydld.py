#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements YDLD (YouTube Driving Light Detection) dataset for
nighttime light detection and enhancement.
"""

__all__ = [
    "YDLD",
]

from ....core import *


@DATASETS.register(name="ydld")
class YDLD(ImageDataset):
    """YDLD dataset."""

    root_name : str         = "ydld"
    tasks     : list[Task]  = [Task.NTE, Task.LLE, Task.DETECT]
    splits    : list[Split] = [Split.TRAIN, Split.TEST]
    modalities: Modalities  = {
        "image": Modality(name="image",   type="image", module=Image,           train=True, test=True, primary=True),
        # "depth": Modality(name=DepthName, type="image", module=DefaultDepthMap, train=True, test=True),
    }
    classes   : Classes     = Classes([
        {"name": "car_light",            "id": 0, "color": (255,   0,   0)},
        {"name": "traffic_signal_light", "id": 1, "color": (0  , 128,   0)},
        {"name": "street_light",         "id": 2, "color": (0  ,   0, 255)},
    ])

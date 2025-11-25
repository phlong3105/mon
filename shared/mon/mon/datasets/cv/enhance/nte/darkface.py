#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements DarkFace dataset for nighttime face enhancement and detection."""

__all__ = [
    "DarkFace",
]

from ....core import *


@DATASETS.register(name="darkface")
class DarkFace(ImageDataset):
    """DarkFace dataset."""

    root_name : str         = "darkface"
    tasks     : list[Task]  = [Task.NTE, Task.LLE, Task.DETECT]
    splits    : list[Split] = [Split.TEST]
    modalities: Modalities  = {
        "image": Modality(name="image",   type="image", module=Image,           train=True, test=True, primary=True),
        "depth": Modality(name=DepthName, type="image", module=DefaultDepthMap, train=True, test=True),
    }
    classes   : Classes     = Classes([
        {"name": "face", "id": 0, "color": [ 81, 120, 228]},
    ])

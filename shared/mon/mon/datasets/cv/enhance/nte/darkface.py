#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module for DarkFace dataset.

This module implements DarkFace dataset for nighttime face enhancement and
detection.
"""

__all__ = [
    "DarkFace",
]

from ....core import *


@DATASETS.register(name="darkface")
class DarkFace(ImageDataset):
    """DarkFace dataset."""

    _root_name : str         = "darkface"
    _tasks     : list[Task]  = [Task.NTE, Task.LLE, Task.DETECT]
    _splits    : list[Split] = [Split.TEST]
    _modalities: Modalities  = {
        "image": Modality(name="image",   type="image", module=Image,           train=True, test=True, primary=True),
        "depth": Modality(name=DepthName, type="image", module=DefaultDepthMap, train=True, test=True),
    }
    _classes   : Classes     = Classes([
        {"name": "face", "id": 0, "color": [ 81, 120, 228]},
    ])

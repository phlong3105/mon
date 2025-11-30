#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module for RealNightHaze dataset.

This module implements the RealNightHaze dataset for nighttime image dehazing.
"""

__all__ = [
    "RealNightHaze",
]

from ....core import *


@DATASETS.register(name="realnighthaze")
class RealNightHaze(ImageDataset):
    """RealNightHaze dataset."""

    _root_name : str         = "realnighthaze"
    _tasks     : list[Task]  = [Task.NTE, Task.LLE, Task.DEHAZE]
    _splits    : list[Split] = [Split.TEST]
    _modalities: Modalities  = {
        "image": Modality(name="image",   type="image", module=Image,           train=True, test=True, primary=True),
        "depth": Modality(name=DepthName, type="image", module=DefaultDepthMap, train=True, test=True),
    }
    _classes   : Classes     = None

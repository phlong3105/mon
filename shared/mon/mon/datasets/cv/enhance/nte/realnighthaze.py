#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the RealNightHaze dataset for nighttime image dehazing."""

__all__ = [
    "RealNightHaze",
]

from ....core import *


@DATASETS.register(name="realnighthaze")
class RealNightHaze(ImageDataset):
    """RealNightHaze dataset."""

    root_name : str         = "realnighthaze"
    tasks     : list[Task]  = [Task.NTE, Task.LLE, Task.DEHAZE]
    splits    : list[Split] = [Split.TEST]
    modalities: Modalities  = {
        "image": Modality(name="image",   type="image", module=Image,           train=True, test=True, primary=True),
        "depth": Modality(name=DepthName, type="image", module=DefaultDepthMap, train=True, test=True),
    }
    classes   : Classes     = None

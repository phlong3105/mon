#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module for Speed datasets.

This module implements the Speed10 and Speed1K datasets for efficiency
benchmarking in low-light image enhancement tasks.
"""

__all__ = [
    "Speed1K",
]

from ..core import *


@DATASETS.register(name="speed10")
class Speed10(ImageDataset):
    """Speed10 dataset."""

    _subset    : str         = "speed10"
    _tasks     : list[Task]  = [Task.LLE]
    _splits    : list[Split] = [Split.TEST]
    _modalities: Modalities  = {
        "image": Modality(name="image",   type="image", module=Image,           train=True, test=True, primary=True),
        "depth": Modality(name=DepthName, type="image", module=DefaultDepthMap, train=True, test=True),
    }
    _classes   : ClassList   = None
    

@DATASETS.register(name="speed1k")
class Speed1K(ImageDataset):
    """Speed1K dataset."""

    _subset    : str         = "speed1k"
    _tasks     : list[Task]  = [Task.LLE]
    _splits    : list[Split] = [Split.TEST]
    _modalities: Modalities  = {
        "image": Modality(name="image",   type="image", module=Image,           train=True, test=True, primary=True),
        "depth": Modality(name=DepthName, type="image", module=DefaultDepthMap, train=True, test=True),
    }
    _classes   : ClassList   = None

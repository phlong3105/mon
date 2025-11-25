#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements Speed datasets for efficiency benchmarking."""

__all__ = [
    "Speed1K",
]

from ..core import *


@DATASETS.register(name="speed10")
class Speed10(ImageDataset):
    """Speed10 dataset."""

    root_name : str         = "speed10"
    tasks     : list[Task]  = [Task.LLE]
    splits    : list[Split] = [Split.TEST]
    modalities: Modalities  = {
        "image": Modality(name="image",   type="image", module=Image,           train=True, test=True, primary=True),
        "depth": Modality(name=DepthName, type="image", module=DefaultDepthMap, train=True, test=True),
    }
    classes   : Classes     = None
    

@DATASETS.register(name="speed1k")
class Speed1K(ImageDataset):
    """Speed1K dataset."""

    root_name : str         = "speed1k"
    tasks     : list[Task]  = [Task.LLE]
    splits    : list[Split] = [Split.TEST]
    modalities: Modalities  = {
        "image": Modality(name="image",   type="image", module=Image,           train=True, test=True, primary=True),
        "depth": Modality(name=DepthName, type="image", module=DefaultDepthMap, train=True, test=True),
    }
    classes   : Classes     = None

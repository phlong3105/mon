#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the DICM dataset for low-light image enhancement tasks."""

__all__ = [
    "DICM",
]

from ....core import *


@DATASETS.register(name="dicm")
class DICM(ImageDataset):
    """DICM dataset."""
    
    root_name : str         = "dicm"
    tasks     : list[Task]  = [Task.LLE]
    splits    : list[Split] = [Split.TEST]
    modalities: Modalities  = {
        "image": Modality(name="image",   type="image", module=Image,           train=True, test=True, primary=True),
        "depth": Modality(name=DepthName, type="image", module=DefaultDepthMap, train=True, test=True),
    }
    classes   : Classes     = None
    num_classes: int        = 0

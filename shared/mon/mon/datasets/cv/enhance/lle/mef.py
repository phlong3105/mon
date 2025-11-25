#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the MEF dataset for low-light image enhancement tasks."""

__all__ = [
    "MEF",
]

from ....core import *


@DATASETS.register(name="mef")
class MEF(ImageDataset):
    """MEF dataset."""
    
    root_name : str         = "mef"
    tasks     : list[Task]  = [Task.LLE]
    splits    : list[Split] = [Split.TEST]
    modalities: Modalities  = {
        "image": Modality(name="image",   type="image", module=Image,           train=True, test=True, primary=True),
        "depth": Modality(name=DepthName, type="image", module=DefaultDepthMap, train=True, test=True),
    }
    classes   : Classes     = None

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the Fusion dataset for low-light enhancement tasks."""

__all__ = [
    "Fusion",
]

from ....core import *


@DATASETS.register(name="fusion")
class Fusion(ImageDataset):
    """Fusion dataset."""
    
    root_name : str         = "fusion"
    tasks     : list[Task]  = [Task.LLE]
    splits    : list[Split] = [Split.TEST]
    modalities: Modalities  = {
        "image": Modality(name="image",   type="image", module=Image,           train=True, test=True, primary=True),
        "depth": Modality(name=DepthName, type="image", module=DefaultDepthMap, train=True, test=True),
    }
    classes   : Classes     = None

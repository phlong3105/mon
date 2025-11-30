#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module for Fusion dataset.

This module implements the Fusion dataset for low-light enhancement tasks.
"""

__all__ = [
    "Fusion",
]

from ....core import *


@DATASETS.register(name="fusion")
class Fusion(ImageDataset):
    """Fusion dataset."""
    
    _root_name : str         = "fusion"
    _tasks     : list[Task]  = [Task.LLE]
    _splits    : list[Split] = [Split.TEST]
    _modalities: Modalities  = {
        "image": Modality(name="image",   type="image", module=Image,           train=True, test=True, primary=True),
        "depth": Modality(name=DepthName, type="image", module=DefaultDepthMap, train=True, test=True),
    }
    _classes   : Classes     = None

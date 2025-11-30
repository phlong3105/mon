#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module for DICM dataset.

This module implements the DICM dataset for low-light image enhancement tasks.
"""

__all__ = [
    "DICM",
]

from ....core import *


@DATASETS.register(name="dicm")
class DICM(ImageDataset):
    """DICM dataset."""
    
    _root_name : str         = "dicm"
    _tasks     : list[Task]  = [Task.LLE]
    _splits    : list[Split] = [Split.TEST]
    _modalities: Modalities  = {
        "image": Modality(name="image",   type="image", module=Image,           train=True, test=True, primary=True),
        "depth": Modality(name=DepthName, type="image", module=DefaultDepthMap, train=True, test=True),
    }
    _classes   : Classes     = None

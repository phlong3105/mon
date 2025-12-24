#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module for LIME dataset.

This module implements the LIME dataset for low-light image enhancement tasks.
"""

__all__ = [
    "LIME",
]

from ....core import *


@DATASETS.register(name="lime")
class LIME(ImageDataset):
    """LIME dataset."""
    
    _subset    : str         = "lime"
    _tasks     : list[Task]  = [Task.LLE]
    _splits    : list[Split] = [Split.TEST]
    _modalities: Modalities  = {
        "image": Modality(name="image",   type="image", module=Image,           train=True, test=True, primary=True),
        "depth": Modality(name=DepthName, type="image", module=DefaultDepthMap, train=True, test=True),
    }
    _classes   : ClassList   = None

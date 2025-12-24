#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module for LOL-v1 dataset.

This module implements the LOL-v1 dataset for low-light image enhancement tasks.
"""

__all__ = [
    "LOLv1",
]

from ....core import *


@DATASETS.register(name="lolv1")
class LOLv1(ImageDataset):
    """LOL-v1 dataset."""
    
    _subset    : str         = "lolv1"
    _tasks     : list[Task]  = [Task.LLE]
    _splits    : list[Split] = [Split.TRAIN, Split.TEST]
    _modalities: Modalities  = {
        "image": Modality(name="image",   type="image", module=Image,           train=True, test=True, primary=True),
        "depth": Modality(name=DepthName, type="image", module=DefaultDepthMap, train=True, test=True),
        "ref"  : Modality(name="ref",     type="image", module=Image,           train=True, test=True),
    }
    _classes   : ClassList   = None

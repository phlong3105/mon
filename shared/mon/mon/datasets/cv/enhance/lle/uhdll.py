#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module for the UHD-LL dataset.

This module implements the UHD-LL dataset for low-light image enhancement tasks.
"""

__all__ = [
    "UHDLL",
]

from ....core import *


@DATASETS.register(name="uhdll")
class UHDLL(ImageDataset):
    """UHD-LL dataset."""
    
    _root_name : str         = "uhdll"
    _tasks     : list[Task]  = [Task.LLE]
    _splits    : list[Split] = [Split.TRAIN, Split.TEST]
    _modalities: Modalities  = {
        "image": Modality(name="image",   type="image", module=Image,           train=True, test=True, primary=True),
        "depth": Modality(name=DepthName, type="image", module=DefaultDepthMap, train=True, test=True),
        "ref"  : Modality(name="ref",     type="image", module=Image,           train=True, test=True),
    }
    _classes   : Classes     = None

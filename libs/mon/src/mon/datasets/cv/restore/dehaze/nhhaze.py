#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module for the NH-Haze dataset.

This module implements NH-Haze dataset for image dehazing.
"""

__all__ = [
    "NHHaze",
]

from ....core import *


@DATASETS.register(name="nhhaze")
class NHHaze(ImageDataset):
    """NH-Haze dataset."""
    
    _subset    : str         = "nhhaze"
    _tasks     : list[Task]  = [Task.DEHAZE]
    _splits    : list[Split] = [Split.TRAIN, Split.VAL, Split.TEST]
    _modalities: Modalities  = {
        "image": Modality(name="image",   type="image", module=Image,           train=True, test=True, primary=True),
        "depth": Modality(name=DepthName, type="image", module=DefaultDepthMap, train=True, test=True),
        "ref"  : Modality(name="ref",     type="image", module=Image,           train=True, test=True),
    }
    _classes   : ClassList   = None

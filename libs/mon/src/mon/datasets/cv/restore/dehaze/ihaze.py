#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module for the I-Haze dataset.

This module implements I-Haze dataset for image dehazing.
"""

__all__ = [
    "IHaze",
]

from mon.core import rich
from ....core import *


@DATASETS.register(name="ihaze")
class IHaze(ImageDataset):
    """IHaze dataset."""
    
    _subset    : str         = "ihaze"
    _tasks     : list[Task]  = [Task.DEHAZE]
    _splits    : list[Split] = [Split.TRAIN, Split.VAL, Split.TEST]
    _modalities: Modalities  = {
        "image": Modality(name="image",   type="image", module=Image,           train=True, test=True, primary=True),
        "depth": Modality(name=DepthName, type="image", module=DefaultDepthMap, train=True, test=True),
        "ref"  : Modality(name="ref",     type="image", module=Image,           train=True, test=True),
    }
    _classes   : Classes     = None

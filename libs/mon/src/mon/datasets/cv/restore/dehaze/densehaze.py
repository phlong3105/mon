#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module for the Dense-Haze dataset.

This module implements the Dense-Haze dataset for image dehazing.
"""

__all__ = [
    "DenseHaze",
]

from mon.core import rich
from ....core import *


@DATASETS.register(name="densehaze")
class DenseHaze(ImageDataset):
    """Dense-Haze dataset."""
    
    _subset    : str         = "densehaze"
    _tasks     : list[Task]  = [Task.DEHAZE]
    _splits    : list[Split] = [Split.TRAIN, Split.VAL, Split.TEST]
    _modalities: Modalities  = {
        "image": Modality(name="image",   type="image", module=Image,           train=True, test=True, primary=True),
        "depth": Modality(name=DepthName, type="image", module=DefaultDepthMap, train=True, test=True),
        "ref"  : Modality(name="ref",     type="image", module=Image,           train=True, test=True),
    }
    _classes   : ClassList   = None

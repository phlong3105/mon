#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module for VV dataset.

This module implements the VV dataset for low-light enhancement tasks.
"""

__all__ = [
    "VV",
]

from ....core import *


@DATASETS.register(name="vv")
class VV(ImageDataset):
    """VV dataset."""

    _root_name : str         = "vv"
    _tasks     : list[Task]  = [Task.LLE]
    _splits    : list[Split] = [Split.TEST]
    _modalities: Modalities  = {
        "image": Modality(name="image",   type="image", module=Image,           train=True, test=True, primary=True),
        "depth": Modality(name=DepthName, type="image", module=DefaultDepthMap, train=True, test=True),
    }
    _classes   : Classes     = None

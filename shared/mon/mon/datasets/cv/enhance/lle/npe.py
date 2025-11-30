#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module for NPE dataset.

This module implements the NPE dataset for low-light image enhancement tasks.
"""

__all__ = [
    "NPE",
]

from ....core import *


@DATASETS.register(name="npe")
class NPE(ImageDataset):
    """NPE dataset."""

    _root_name : str         = "npe"
    _tasks     : list[Task]  = [Task.LLE]
    _splits    : list[Split] = [Split.TEST]
    _modalities: Modalities  = {
        "image": Modality(name="image",   type="image", module=Image,           train=True, test=True, primary=True),
        "depth": Modality(name=DepthName, type="image", module=DefaultDepthMap, train=True, test=True),
    }
    _classes   : Classes     = None

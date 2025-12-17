#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module for the Real-LOL-Blur dataset.

This module implements the Real-LOL-Blur dataset for deblurring and low-light
enhancement.
"""

__all__ = [
    "RealLOLBlur",
]

from ....core import *


@DATASETS.register(name="reallolblur")
class RealLOLBlur(ImageDataset):
    """Real-LOL-Blur dataset."""
    
    _subset    : str         = "reallolblur"
    _tasks     : list[Task]  = [Task.DEBLUR, Task.LLE]
    _splits    : list[Split] = [Split.TEST]
    _modalities: Modalities  = {
        "image": Modality(name="image", type="image", module=Image, train=True, test=True, primary=True),
    }
    _classes   : Classes     = None

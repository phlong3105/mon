#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the Real-LOL-Blur dataset for deblurring and
low-light enhancement.
"""

__all__ = [
    "RealLOLBlur",
]

from ....core import *


@DATASETS.register(name="reallolblur")
class RealLOLBlur(ImageDataset):
    """Real-LOL-Blur dataset."""
    
    root_name : str         = "reallolblur"
    tasks     : list[Task]  = [Task.DEBLUR, Task.LLE]
    splits    : list[Split] = [Split.TEST]
    modalities: Modalities  = {
        "image": Modality(name="image", type="image", module=Image, train=True, test=True, primary=True),
    }
    classes   : Classes     = None

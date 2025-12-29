#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Real-LOL-Blur dataset.

This module implements the Real-LOL-Blur dataset for image de-blurring and
low-light enhancement.
"""

__all__ = [
    "RealLOLBlur",
]

from ....meta import *


@DATASETS.register(name="reallolblur")
class RealLOLBlur(ImageDataset):
    """Real-LOL-Blur dataset."""
    
    _subset    : str         = "reallolblur"
    _tasks     : list[Task]  = [Task.DEBLUR, Task.LLE]
    _splits    : list[Split] = [Split.TEST]
    _modalities: Modalities  = {
        "image": Modality(name="image", type="image", module=Image, train=True, test=True, primary=True),
    }
    _classlist : ClassList   = None

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""MEF dataset.

This module implements the MEF dataset for low-light image enhancement.
"""

__all__ = [
    "MEF",
]

from ....meta import *


@DATASETS.register(name="mef")
class MEF(ImageDataset):
    """MEF dataset."""
    
    _subset    : str         = "mef"
    _tasks     : list[Task]  = [Task.LLE]
    _splits    : list[Split] = [Split.TEST]
    _modalities: Modalities  = {
        "image": Modality(name="image",   type="image", module=Image,           train=True, test=True, primary=True),
        "depth": Modality(name=DepthName, type="image", module=DefaultDepthMap, train=True, test=True),
    }
    _classlist : ClassList   = None

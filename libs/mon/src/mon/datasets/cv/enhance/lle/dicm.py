#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""DICM dataset.

This module implements the DICM dataset for low-light image enhancement.
"""

__all__ = [
    "DICM",
]

from ....api import *


@DATASETS.register(name="dicm")
class DICM(ImageDataset):
    """DICM dataset."""
    
    _subset    : str         = "dicm"
    _tasks     : list[Task]  = [Task.LLE]
    _splits    : list[Split] = [Split.TEST]
    _modalities: Modalities  = {
        "image": Modality(name="image",   type="image", module=Image,           train=True, test=True, primary=True),
        "depth": Modality(name=DepthName, type="image", module=DefaultDepthMap, train=True, test=True),
    }
    _classlist : ClassList   = None

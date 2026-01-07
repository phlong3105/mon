#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Fusion dataset.

This module implements the Fusion dataset for low-light enhancement.
"""

__all__ = [
    "Fusion",
]

from ....api import *


@DATASETS.register(name="fusion")
class Fusion(ImageDataset):
    """Fusion dataset."""
    
    _subset    : str         = "fusion"
    _tasks     : list[Task]  = [Task.LLE]
    _splits    : list[Split] = [Split.TEST]
    _modalities: Modalities  = {
        "image": Modality(name="image",   type="image", module=Image,           train=True, test=True, primary=True),
        "depth": Modality(name=DepthName, type="image", module=DefaultDepthMap, train=True, test=True),
    }
    _classlist : ClassList   = None

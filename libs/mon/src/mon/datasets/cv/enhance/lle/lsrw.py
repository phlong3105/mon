#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""LSRW dataset.

This module implements the LSRW dataset for low-light image enhancement.
"""

__all__ = [
    "LSRW",
]

from ....api import *


@DATASETS.register(name="lsrw")
class LSRW(ImageDataset):
    """LSRW dataset."""
    
    _subset    : str         = "lsrw"
    _tasks     : list[Task]  = [Task.LLE]
    _splits    : list[Split] = [Split.TRAIN, Split.TEST]
    _modalities: Modalities  = {
        "image": Modality(name="image",   type="image", module=Image,           train=True, test=True, primary=True),
        "depth": Modality(name=DepthName, type="image", module=DefaultDepthMap, train=True, test=True),
        "ref"  : Modality(name="ref",     type="image", module=Image,           train=True, test=True),
    }
    _classlist : ClassList   = None

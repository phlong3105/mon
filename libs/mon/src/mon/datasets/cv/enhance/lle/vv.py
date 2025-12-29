#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""VV dataset.

This module implements the VV dataset for low-light enhancement tasks.
"""

__all__ = [
    "VV",
]

from ....meta import *


@DATASETS.register(name="vv")
class VV(ImageDataset):
    """VV dataset."""

    _subset    : str         = "vv"
    _tasks     : list[Task]  = [Task.LLE]
    _splits    : list[Split] = [Split.TEST]
    _modalities: Modalities  = {
        "image": Modality(name="image",   type="image", module=Image,           train=True, test=True, primary=True),
        "depth": Modality(name=DepthName, type="image", module=DefaultDepthMap, train=True, test=True),
    }
    _classlist : ClassList   = None

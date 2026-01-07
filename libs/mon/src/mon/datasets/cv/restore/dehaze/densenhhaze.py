#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Dense-NH-Haze dataset.

This module implements the Dense-NH-Haze dataset for image de-hazing.
"""

__all__ = [
    "DenseNHHaze",
]

from ....api import *


@DATASETS.register(name="densenhhaze")
class DenseNHHaze(ImageDataset):
    """Dense-NH-Haze dataset."""
    
    _subset    : str         = "densenhhaze"
    _tasks     : list[Task]  = [Task.DEHAZE]
    _splits    : list[Split] = [Split.TRAIN, Split.VAL, Split.TEST]
    _modalities: Modalities  = {
        "image": Modality(name="image",   type="image", module=Image,           train=True, test=True, primary=True),
        "depth": Modality(name=DepthName, type="image", module=DefaultDepthMap, train=True, test=True),
        "ref"  : Modality(name="ref",     type="image", module=Image,           train=True, test=True),
    }
    _classlist : ClassList   = None

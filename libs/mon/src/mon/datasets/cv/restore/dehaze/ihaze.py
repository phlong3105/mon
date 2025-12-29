#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""I-Haze dataset.

This module implements the I-Haze dataset for image de-hazing.
"""

__all__ = [
    "IHaze",
]

from ....meta import *


@DATASETS.register(name="ihaze")
class IHaze(ImageDataset):
    """IHaze dataset."""
    
    _subset    : str         = "ihaze"
    _tasks     : list[Task]  = [Task.DEHAZE]
    _splits    : list[Split] = [Split.TRAIN, Split.VAL, Split.TEST]
    _modalities: Modalities  = {
        "image": Modality(name="image",   type="image", module=Image,           train=True, test=True, primary=True),
        "depth": Modality(name=DepthName, type="image", module=DefaultDepthMap, train=True, test=True),
        "ref"  : Modality(name="ref",     type="image", module=Image,           train=True, test=True),
    }
    _classlist : ClassList   = None

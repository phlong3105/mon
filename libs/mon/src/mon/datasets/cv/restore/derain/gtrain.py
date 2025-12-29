#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""GT-Rain dataset.

This module implements the GT-Rain dataset for image de-raining.
"""

__all__ = [
    "GTRain",
]

from mon.core import rich
from ....meta import *


@DATASETS.register(name="gtrain")
class GTRain(ImageDataset):
    """GTRain dataset."""
    
    _subset    : str         = "gtrain"
    _tasks     : list[Task]  = [Task.DERAIN]
    _splits    : list[Split] = [Split.TRAIN, Split.VAL, Split.TEST]
    _modalities: Modalities  = {
        "image": Modality(name="image", type="image", module=Image, train=True, test=True, primary=True),
        "ref"  : Modality(name="ref",   type="image", module=Image, train=True, test=True),
    }
    _classlist : ClassList   = None

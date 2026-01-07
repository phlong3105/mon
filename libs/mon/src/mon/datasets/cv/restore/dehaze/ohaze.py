#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""O-Haze dataset.

This module implements the O-Haze dataset for image de-hazing.
"""

__all__ = [
    "OHaze",
]

from ....api import *


@DATASETS.register(name="ohaze")
class OHaze(ImageDataset):
    """O-Haze dataset."""

    _subset    : str         = "ohaze"
    _tasks     : list[Task]  = [Task.DEHAZE]
    _splits    : list[Split] = [Split.TRAIN, Split.VAL, Split.TEST]
    _modalities: Modalities  = {
        "image": Modality(name="image",   type="image", module=Image,           train=True, test=True, primary=True),
        "depth": Modality(name=DepthName, type="image", module=DefaultDepthMap, train=True, test=True),
        "ref"  : Modality(name="ref",     type="image", module=Image,           train=True, test=True),
    }
    _classlist : ClassList   = None

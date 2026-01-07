#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""GT-Snow dataset.

This module implements GT-Snow dataset for image de-snowing.
"""

__all__ = [
    "GTSnow",
]

from ....api import *


@DATASETS.register(name="gtsnow")
class GTSnow(ImageDataset):
    """GTSnow dataset."""
    
    _subset    : str         = "gtsnow"
    _tasks     : list[Task]  = [Task.DESNOW]
    _splits    : list[Split] = [Split.TRAIN]
    _modalities: Modalities  = {
        "image": Modality(name="image", type="image", module=Image, train=True, test=True, primary=True),
        "ref"  : Modality(name="ref",   type="image", module=Image, train=True, test=False),
    }
    _classlist : ClassList   = None

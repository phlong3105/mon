#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""FlareReal800 dataset.

This module implements the FlareReal800 dataset for image de-flaring.
"""

__all__ = [
    "FlareReal800",
]

from ....meta import *


@DATASETS.register(name="flarereal800")
class FlareReal800(ImageDataset):
    """FlareReal800 dataset."""
    
    _subset    : str         = "flarereal800"
    _tasks     : list[Task]  = [Task.DEFLARE]
    _splits    : list[Split] = [Split.TRAIN, Split.VAL]
    _modalities: Modalities  = {
        "image": Modality(name="image", type="image", module=Image, train=True, test=True, primary=True),
        "ref"  : Modality(name="ref",   type="image", module=Image, train=True, test=False),
    }
    _classlist : ClassList   = None

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module for the FlareReal800 dataset.

This module implements the FlareReal800 dataset for deflaring tasks.
"""

__all__ = [
    "FlareReal800",
]

from ....core import *


@DATASETS.register(name="flarereal800")
class FlareReal800(ImageDataset):
    """FlareReal800 dataset."""
    
    _root_name : str         = "flarereal800"
    _tasks     : list[Task]  = [Task.DEFLARE]
    _splits    : list[Split] = [Split.TRAIN, Split.VAL]
    _modalities: Modalities  = {
        "image": Modality(name="image", type="image", module=Image, train=True, test=True, primary=True),
        "ref"  : Modality(name="ref",   type="image", module=Image, train=True, test=False),
    }
    _classes   : Classes     = None

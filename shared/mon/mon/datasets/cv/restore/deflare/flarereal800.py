#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the FlareReal800 dataset for deflaring tasks."""

__all__ = [
    "FlareReal800",
]

from ....core import *


@DATASETS.register(name="flarereal800")
class FlareReal800(ImageDataset):
    """FlareReal800 dataset."""
    
    root_name : str         = "flarereal800"
    tasks     : list[Task]  = [Task.DEFLARE]
    splits    : list[Split] = [Split.TRAIN, Split.VAL]
    modalities: Modalities  = {
        "image": Modality(name="image", type="image", module=Image, train=True, test=True, primary=True),
        "ref"  : Modality(name="ref",   type="image", module=Image, train=True, test=False),
    }
    classes   : Classes     = None

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module for the LightEffect dataset.

This module implements the LightEffect dataset for deflaring tasks.
"""

__all__ = [
    "LightEffect",
]

from ....core import *


@DATASETS.register(name="lighteffect")
class LightEffect(ImageDataset):
    """LightEffect dataset."""
    
    _subset    : str         = "lighteffect"
    _tasks     : list[Task]  = [Task.DEFLARE]
    _splits    : list[Split] = [Split.TRAIN]
    _modalities: Modalities  = {
        "image": Modality(name="image", type="image", module=Image, train=True, test=True, primary=True),
    }
    _classes   : ClassList   = None

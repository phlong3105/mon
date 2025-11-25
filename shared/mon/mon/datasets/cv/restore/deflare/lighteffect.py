#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the LightEffect dataset for deflaring tasks."""

__all__ = [
    "LightEffect",
]

from ....core import *


@DATASETS.register(name="lighteffect")
class LightEffect(ImageDataset):
    """LightEffect dataset."""
    
    root_name : str         = "lighteffect"
    tasks     : list[Task]  = [Task.DEFLARE]
    splits    : list[Split] = [Split.TRAIN]
    modalities: Modalities  = {
        "image": Modality(name="image", type="image", module=Image, train=True, test=True, primary=True),
    }
    classes   : Classes     = None

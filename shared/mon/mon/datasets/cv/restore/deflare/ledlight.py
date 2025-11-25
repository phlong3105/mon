#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements LEDLight dataset for deflaring tasks."""

__all__ = [
    "LEDLight",
]

from ....core import *


@DATASETS.register(name="ledlight")
class LEDLight(ImageDataset):
    """LEDLight dataset."""
    
    root_name : str         = "ledlight"
    tasks     : list[Task]  = [Task.DEFLARE]
    splits    : list[Split] = [Split.TEST]
    modalities: Modalities  = {
        "image": Modality(name="image", type="image", module=Image, train=True, test=True, primary=True),
        "ref"  : Modality(name="ref",   type="image", module=Image, train=True, test=True),
    }
    classes   : Classes     = None

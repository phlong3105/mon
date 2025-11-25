#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the GT-Rain dataset for image deraining tasks."""

__all__ = [
    "GTRain",
]

from mon.core import rich
from ....core import *


@DATASETS.register(name="gtrain")
class GTRain(ImageDataset):
    """GTRain dataset."""
    
    root_name : str         = "gtrain"
    tasks     : list[Task]  = [Task.DERAIN]
    splits    : list[Split] = [Split.TRAIN, Split.VAL, Split.TEST]
    modalities: Modalities  = {
        "image": Modality(name="image", type="image", module=Image, train=True, test=True, primary=True),
        "ref"  : Modality(name="ref",   type="image", module=Image, train=True, test=True),
    }
    classes   : Classes     = None

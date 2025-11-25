#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements O-Haze dataset for image dehazing."""

__all__ = [
    "OHaze",
]

from ....core import *


@DATASETS.register(name="ohaze")
class OHaze(ImageDataset):
    """O-Haze dataset."""

    root_name : str         = "ohaze"
    tasks     : list[Task]  = [Task.DEHAZE]
    splits    : list[Split] = [Split.TRAIN, Split.VAL, Split.TEST]
    modalities: Modalities  = {
        "image": Modality(name="image",   type="image", module=Image,           train=True, test=True, primary=True),
        "depth": Modality(name=DepthName, type="image", module=DefaultDepthMap, train=True, test=True),
        "ref"  : Modality(name="ref",     type="image", module=Image,           train=True, test=True),
    }
    classes   : Classes     = None

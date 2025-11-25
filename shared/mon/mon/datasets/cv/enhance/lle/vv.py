#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the VV dataset for low-light enhancement tasks."""

__all__ = [
    "VV",
]

from ....core import *


@DATASETS.register(name="vv")
class VV(ImageDataset):
    """VV dataset."""

    root_name : str         = "vv"
    tasks     : list[Task]  = [Task.LLE]
    splits    : list[Split] = [Split.TEST]
    modalities: Modalities  = {
        "image": Modality(name="image",   type="image", module=Image,           train=True, test=True, primary=True),
        "depth": Modality(name=DepthName, type="image", module=DefaultDepthMap, train=True, test=True),
    }
    classes   : Classes     = None

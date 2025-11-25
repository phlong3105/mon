#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the UHD-LL dataset for low-light image enhancement tasks."""

__all__ = [
    "UHDLL",
]

from ....core import *


@DATASETS.register(name="uhdll")
class UHDLL(ImageDataset):
    """UHD-LL dataset."""
    
    root_name : str         = "uhdll"
    tasks     : list[Task]  = [Task.LLE]
    splits    : list[Split] = [Split.TRAIN, Split.TEST]
    modalities: Modalities  = {
        "image": Modality(name="image",   type="image", module=Image,           train=True, test=True, primary=True),
        "depth": Modality(name=DepthName, type="image", module=DefaultDepthMap, train=True, test=True),
        "ref"  : Modality(name="ref",     type="image", module=Image,           train=True, test=True),
    }
    classes   : Classes     = None

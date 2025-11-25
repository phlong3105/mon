#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements LSRW dataset for low-light image enhancement tasks."""

__all__ = [
    "LSRW",
]

from ....core import *


@DATASETS.register(name="lsrw")
class LSRW(ImageDataset):
    """LSRW dataset."""
    
    root_name : str         = "lsrw"
    tasks     : list[Task]  = [Task.LLE]
    splits    : list[Split] = [Split.TRAIN, Split.TEST]
    modalities: Modalities  = {
        "image": Modality(name="image",   type="image", module=Image,           train=True, test=True, primary=True),
        "depth": Modality(name=DepthName, type="image", module=DefaultDepthMap, train=True, test=True),
        "ref"  : Modality(name="ref",     type="image", module=Image,           train=True, test=True),
    }
    classes   : Classes     = None

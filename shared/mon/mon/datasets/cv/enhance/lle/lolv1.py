#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the LOL-v1 dataset for low-light image enhancement tasks."""

__all__ = [
    "LOLv1",
]

from ....core import *


@DATASETS.register(name="lolv1")
class LOLv1(ImageDataset):
    """LOL-v1 dataset."""
    
    root_name : str         = "lolv1"
    tasks     : list[Task]  = [Task.LLE]
    splits    : list[Split] = [Split.TRAIN, Split.TEST]
    modalities: Modalities  = {
        "image": Modality(name="image",   type="image", module=Image,           train=True, test=True, primary=True),
        "depth": Modality(name=DepthName, type="image", module=DefaultDepthMap, train=True, test=True),
        "ref"  : Modality(name="ref",     type="image", module=Image,           train=True, test=True),
    }
    classes   : Classes     = None

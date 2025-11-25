#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the NPE dataset for low-light image enhancement tasks."""

__all__ = [
    "NPE",
]

from ....core import *


@DATASETS.register(name="npe")
class NPE(ImageDataset):
    """NPE dataset."""

    root_name : str         = "npe"
    tasks     : list[Task]  = [Task.LLE]
    splits    : list[Split] = [Split.TEST]
    modalities: Modalities  = {
        "image": Modality(name="image",   type="image", module=Image,           train=True, test=True, primary=True),
        "depth": Modality(name=DepthName, type="image", module=DefaultDepthMap, train=True, test=True),
    }
    classes   : Classes     = None

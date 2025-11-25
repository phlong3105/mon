#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the Rain800 dataset for image deraining tasks."""

__all__ = [
    "Rain800",
]

from mon.core import rich
from ....core import *


@DATASETS.register(name="rain800")
class Rain800(ImageDataset):
    """Rain800 dataset."""

    root_name : str         = "rain800"
    tasks     : list[Task]  = [Task.DERAIN]
    splits    : list[Split] = [Split.TRAIN, Split.VAL, Split.TEST]
    modalities: Modalities  = {
        "image": Modality(name="image", type="image", module=Image, train=True, test=True, primary=True),
        "ref"  : Modality(name="ref",   type="image", module=Image, train=True, test=True),
    }
    classes   : Classes     = None

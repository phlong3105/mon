#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the Rain1400 dataset for image deraining tasks."""

__all__ = [
    "Rain1400",
]

from ....core import *


@DATASETS.register(name="rain1400")
class Rain1400(ImageDataset):
    """Rain1400 dataset."""

    root_name : str         = "rain1400"
    tasks     : list[Task]  = [Task.DERAIN]
    splits    : list[Split] = [Split.TRAIN, Split.TEST]
    modalities: Modalities  = {
        "image": Modality(name="image", type="image", module=Image, train=True, test=True, primary=True),
        "ref"  : Modality(name="ref",   type="image", module=Image, train=True, test=True),
    }
    classes   : Classes     = None

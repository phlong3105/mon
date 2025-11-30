#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module for the Rain1400 dataset.

This module implements the Rain1400 dataset for image deraining tasks.
"""

__all__ = [
    "Rain1400",
]

from ....core import *


@DATASETS.register(name="rain1400")
class Rain1400(ImageDataset):
    """Rain1400 dataset."""

    _root_name : str         = "rain1400"
    _tasks     : list[Task]  = [Task.DERAIN]
    _splits    : list[Split] = [Split.TRAIN, Split.TEST]
    _modalities: Modalities  = {
        "image": Modality(name="image", type="image", module=Image, train=True, test=True, primary=True),
        "ref"  : Modality(name="ref",   type="image", module=Image, train=True, test=True),
    }
    _classes   : Classes     = None

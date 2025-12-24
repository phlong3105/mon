#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module for the Rain800 dataset.

This module implements the Rain800 dataset for image deraining tasks.
"""

__all__ = [
    "Rain800",
]

from mon.core import rich
from ....core import *


@DATASETS.register(name="rain800")
class Rain800(ImageDataset):
    """Rain800 dataset."""

    _subset    : str         = "rain800"
    _tasks     : list[Task]  = [Task.DERAIN]
    _splits    : list[Split] = [Split.TRAIN, Split.VAL, Split.TEST]
    _modalities: Modalities  = {
        "image": Modality(name="image", type="image", module=Image, train=True, test=True, primary=True),
        "ref"  : Modality(name="ref",   type="image", module=Image, train=True, test=True),
    }
    _classes   : ClassList   = None

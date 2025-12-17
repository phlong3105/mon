#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module for the Rain100 dataset.

This module implements the Rain100 dataset for image deraining tasks.
"""

__all__ = [
    "Rain100",
    "Rain100H",
    "Rain100L",
]

from ....core import *


@DATASETS.register(name="rain100")
class Rain100(ImageDataset):
    """Rain100 dataset."""

    _subset    : str         = "rain100"
    _tasks     : list[Task]  = [Task.DERAIN]
    _splits    : list[Split] = [Split.TEST]
    _modalities: Modalities  = {
        "image": Modality(name="image", type="image", module=Image, train=True, test=True, primary=True),
        "ref"  : Modality(name="ref",   type="image", module=Image, train=True, test=True),
    }
    _classes   : Classes     = None


@DATASETS.register(name="rain100h")
class Rain100H(ImageDataset):
    """Rain100H dataset."""

    _subset    : str         = "rain100h"
    _tasks     : list[Task]  = [Task.DERAIN]
    _splits    : list[Split] = [Split.TRAIN, Split.TEST]
    _modalities: Modalities  = {
        "image": Modality(name="image", type="image", module=Image, train=True, test=True, primary=True),
        "ref"  : Modality(name="ref",   type="image", module=Image, train=True, test=True),
    }
    _classes   : Classes     = None

    

@DATASETS.register(name="rain100l")
class Rain100L(ImageDataset):
    """Rain100L dataset."""

    _subset    : str         = "rain100l"
    _tasks     : list[Task]  = [Task.DERAIN]
    _splits    : list[Split] = [Split.TRAIN, Split.TEST]
    _modalities: Modalities  = {
        "image": Modality(name="image", type="image", module=Image, train=True, test=True, primary=True),
        "ref"  : Modality(name="ref",   type="image", module=Image, train=True, test=True),
    }
    _classes   : Classes     = None

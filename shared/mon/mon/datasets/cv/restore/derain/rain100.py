#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the Rain100 dataset for image deraining tasks."""

__all__ = [
    "Rain100",
    "Rain100H",
    "Rain100L",
]

from ....core import *


@DATASETS.register(name="rain100")
class Rain100(ImageDataset):
    """Rain100 dataset."""

    root_name : str         = "rain100"
    tasks     : list[Task]  = [Task.DERAIN]
    splits    : list[Split] = [Split.TEST]
    modalities: Modalities  = {
        "image": Modality(name="image", type="image", module=Image, train=True, test=True, primary=True),
        "ref"  : Modality(name="ref",   type="image", module=Image, train=True, test=True),
    }
    classes   : Classes     = None


@DATASETS.register(name="rain100h")
class Rain100H(ImageDataset):
    """Rain100H dataset."""

    root_name : str         = "rain100h"
    tasks     : list[Task]  = [Task.DERAIN]
    splits    : list[Split] = [Split.TRAIN, Split.TEST]
    modalities: Modalities  = {
        "image": Modality(name="image", type="image", module=Image, train=True, test=True, primary=True),
        "ref"  : Modality(name="ref",   type="image", module=Image, train=True, test=True),
    }
    classes   : Classes     = None

    

@DATASETS.register(name="rain100l")
class Rain100L(ImageDataset):
    """Rain100L dataset."""

    root_name : str         = "rain100l"
    tasks     : list[Task]  = [Task.DERAIN]
    splits    : list[Split] = [Split.TRAIN, Split.TEST]
    modalities: Modalities  = {
        "image": Modality(name="image", type="image", module=Image, train=True, test=True, primary=True),
        "ref"  : Modality(name="ref",   type="image", module=Image, train=True, test=True),
    }
    classes   : Classes     = None

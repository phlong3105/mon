#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the Rain12 dataset for image deraining tasks."""

__all__ = [
    "Rain12",
]

from mon.core import rich
from ....core import *


@DATASETS.register(name="rain12")
class Rain12(ImageDataset):
    """Rain12 dataset."""

    root_name : str         = "rain12"
    tasks     : list[Task]  = [Task.DERAIN]
    splits    : list[Split] = [Split.TRAIN]
    modalities: Modalities  = {
        "image": Modality(name="image", type="image", module=Image, train=True, test=True, primary=True),
        "ref"  : Modality(name="ref",   type="image", module=Image, train=True, test=False),
    }
    classes   : Classes     = None

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the Rain2800 dataset for image deraining tasks."""

__all__ = [
    "Rain2800",
]

from ....core import *


@DATASETS.register(name="rain2800")
class Rain2800(ImageDataset):
    """Rain2800 dataset."""

    root_name : str         = "rain2800"
    tasks     : list[Task]  = [Task.DERAIN]
    splits    : list[Split] = [Split.TEST]
    modalities: Modalities  = {
        "image": Modality(name="image", type="image", module=Image, train=True, test=True, primary=True),
        "ref"  : Modality(name="ref",   type="image", module=Image, train=True, test=True),
    }
    classes   : Classes     = None

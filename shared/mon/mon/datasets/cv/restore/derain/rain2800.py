#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module for the Rain2800 dataset.

This module implements the Rain2800 dataset for image deraining tasks.
"""

__all__ = [
    "Rain2800",
]

from ....core import *


@DATASETS.register(name="rain2800")
class Rain2800(ImageDataset):
    """Rain2800 dataset."""

    _root_name : str         = "rain2800"
    _tasks     : list[Task]  = [Task.DERAIN]
    _splits    : list[Split] = [Split.TEST]
    _modalities: Modalities  = {
        "image": Modality(name="image", type="image", module=Image, train=True, test=True, primary=True),
        "ref"  : Modality(name="ref",   type="image", module=Image, train=True, test=True),
    }
    _classes   : Classes     = None

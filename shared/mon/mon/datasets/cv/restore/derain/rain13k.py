#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module for the Rain13K dataset.

This module implements the Rain13K dataset for image deraining tasks.
"""

__all__ = [
    "Rain13K",
]

from mon.core import rich
from ....core import *


@DATASETS.register(name="rain13k")
class Rain13K(ImageDataset):
    """Rain13K dataset."""

    _root_name : str         = "rain13k"
    _tasks     : list[Task]  = [Task.DERAIN]
    _splits    : list[Split] = [Split.TRAIN]
    _modalities: Modalities  = {
        "image": Modality(name="image", type="image", module=Image, train=True, test=True, primary=True),
        "ref"  : Modality(name="ref",   type="image", module=Image, train=True, test=False),
    }
    _classes   : Classes     = None

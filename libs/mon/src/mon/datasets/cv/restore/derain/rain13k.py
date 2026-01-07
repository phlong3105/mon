#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Rain13K dataset.

This module implements the Rain13K dataset for image de-raining.
"""

__all__ = [
    "Rain13K",
]

from mon.core import rich
from ....api import *


@DATASETS.register(name="rain13k")
class Rain13K(ImageDataset):
    """Rain13K dataset."""

    _subset    : str         = "rain13k"
    _tasks     : list[Task]  = [Task.DERAIN]
    _splits    : list[Split] = [Split.TRAIN]
    _modalities: Modalities  = {
        "image": Modality(name="image", type="image", module=Image, train=True, test=True, primary=True),
        "ref"  : Modality(name="ref",   type="image", module=Image, train=True, test=False),
    }
    _classlist : ClassList   = None

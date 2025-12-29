#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Rain12 dataset.

This module implements the Rain12 dataset for image de-raining.
"""

__all__ = [
    "Rain12",
]

from mon.core import rich
from ....meta import *


@DATASETS.register(name="rain12")
class Rain12(ImageDataset):
    """Rain12 dataset."""

    _subset    : str         = "rain12"
    _tasks     : list[Task]  = [Task.DERAIN]
    _splits    : list[Split] = [Split.TRAIN]
    _modalities: Modalities  = {
        "image": Modality(name="image", type="image", module=Image, train=True, test=True, primary=True),
        "ref"  : Modality(name="ref",   type="image", module=Image, train=True, test=False),
    }
    _classlist : ClassList   = None

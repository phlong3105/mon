#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Rain1400 dataset.

This module implements the Rain1400 dataset for image de-raining.
"""

__all__ = [
    "Rain1400",
]

from ....meta import *


@DATASETS.register(name="rain1400")
class Rain1400(ImageDataset):
    """Rain1400 dataset."""

    _subset    : str         = "rain1400"
    _tasks     : list[Task]  = [Task.DERAIN]
    _splits    : list[Split] = [Split.TRAIN, Split.TEST]
    _modalities: Modalities  = {
        "image": Modality(name="image", type="image", module=Image, train=True, test=True, primary=True),
        "ref"  : Modality(name="ref",   type="image", module=Image, train=True, test=True),
    }
    _classlist : ClassList   = None

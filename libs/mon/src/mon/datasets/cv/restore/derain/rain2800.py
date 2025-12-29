#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Rain2800 dataset.

This module implements the Rain2800 dataset for image de-raining.
"""

__all__ = [
    "Rain2800",
]

from ....meta import *


@DATASETS.register(name="rain2800")
class Rain2800(ImageDataset):
    """Rain2800 dataset."""

    _subset    : str         = "rain2800"
    _tasks     : list[Task]  = [Task.DERAIN]
    _splits    : list[Split] = [Split.TEST]
    _modalities: Modalities  = {
        "image": Modality(name="image", type="image", module=Image, train=True, test=True, primary=True),
        "ref"  : Modality(name="ref",   type="image", module=Image, train=True, test=True),
    }
    _classlist : ClassList   = None

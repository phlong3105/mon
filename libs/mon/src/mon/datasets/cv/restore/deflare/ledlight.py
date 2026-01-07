#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""LEDLight dataset.

This module implements the LEDLight dataset for image de-flaring.
"""

__all__ = [
    "LEDLight",
]

from ....api import *


@DATASETS.register(name="ledlight")
class LEDLight(ImageDataset):
    """LEDLight dataset."""
    
    _subset    : str         = "ledlight"
    _tasks     : list[Task]  = [Task.DEFLARE]
    _splits    : list[Split] = [Split.TEST]
    _modalities: Modalities  = {
        "image": Modality(name="image", type="image", module=Image, train=True, test=True, primary=True),
        "ref"  : Modality(name="ref",   type="image", module=Image, train=True, test=True),
    }
    _classlist : ClassList   = None

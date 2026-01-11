#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""GT-Rain dataset.

This module provides the GT-Rain dataset for image de-raining.
"""

from __future__ import annotations

__all__ = [
    "GTRain",
]

from ....api import *


@DATASETS.register()
class GTRain(ImageDataset, RegistrableMixin):
    """GTRain dataset."""
    
    _name      : str         = "gtrain"
    _tasks     : list[Task]  = [Task.DERAIN]
    _subset    : str         = None
    _splits    : list[Split] = [Split.TRAIN, Split.VAL, Split.TEST]
    _modalities: Modalities  = {
        "image": Modality(
            name    = "image",
            type    = "image",
            module  = Image,
            train   = True,
            test    = True,
            primary = True,
        ),
        "ref"  : Modality(
            name    = "ref",
            type    = "image",
            module  = Image,
            train   = True,
            test    = True,
        ),
    }
    _classlist : ClassList   = None

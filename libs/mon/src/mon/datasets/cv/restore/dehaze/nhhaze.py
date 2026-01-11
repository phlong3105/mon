#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""NH-Haze dataset.

This module provides the NH-Haze dataset for image de-hazing.
"""

from __future__ import annotations

__all__ = [
    "NHHaze",
]

from ....api import *


@DATASETS.register()
class NHHaze(ImageDataset, RegistrableMixin):
    """NH-Haze dataset."""
    
    _name      : str         = "nhhaze"
    _tasks     : list[Task]  = [Task.DEHAZE]
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
        "depth": Modality(
            name    = DepthName,
            type    = "image",
            module  = DefaultDepthMap,
            train   = True,
            test    = True,
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

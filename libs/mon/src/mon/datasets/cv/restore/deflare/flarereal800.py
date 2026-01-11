#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""FlareReal800 dataset.

This module provides the FlareReal800 dataset for image de-flaring.
"""

from __future__ import annotations

__all__ = [
    "FlareReal800",
]

from ....api import *


@DATASETS.register()
class FlareReal800(ImageDataset, RegistrableMixin):
    """FlareReal800 dataset."""
    
    _name      : str         = "flarereal800"
    _tasks     : list[Task]  = [Task.DEFLARE]
    _subset    : str         = None
    _splits    : list[Split] = [Split.TRAIN, Split.VAL]
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
            test    = False,
        ),
    }
    _classlist : ClassList   = None

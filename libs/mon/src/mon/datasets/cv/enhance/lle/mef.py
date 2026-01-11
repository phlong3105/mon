#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""MEF dataset.

This module provides the MEF dataset for low-light image enhancement.
"""

from __future__ import annotations

__all__ = [
    "MEF",
]

from ....api import *


@DATASETS.register()
class MEF(ImageDataset, RegistrableMixin):
    """MEF dataset."""
    
    _name      : str         = "mef"
    _tasks     : list[Task]  = [Task.LLE]
    _subset    : str         = None
    _splits    : list[Split] = [Split.TEST]
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
    }
    _classlist : ClassList   = None

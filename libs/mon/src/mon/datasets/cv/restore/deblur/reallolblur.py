#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Real-LOL-Blur dataset.

This module provides the Real-LOL-Blur dataset for image de-blurring and
low-light enhancement.
"""

from __future__ import annotations

__all__ = [
    "RealLOLBlur",
]

from ....api import *


@DATASETS.register()
class RealLOLBlur(ImageDataset, RegistrableMixin):
    """Real-LOL-Blur dataset."""
    
    _name      : str         = "reallolblur"
    _tasks     : list[Task]  = [Task.DEBLUR, Task.LLE]
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
    }
    _classlist : ClassList   = None

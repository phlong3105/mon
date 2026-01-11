#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""DarkFace dataset.

This module provides the DarkFace dataset for nighttime face enhancement and
detection.
"""

from __future__ import annotations

__all__ = [
    "DarkFace",
]

from ....api import *


@DATASETS.register()
class DarkFace(ImageDataset, RegistrableMixin):
    """DarkFace dataset."""

    _name      : str         = "darkface"
    _tasks     : list[Task]  = [Task.NTE, Task.LLE, Task.DETECT]
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
    _classlist : ClassList   = ClassList([
        {"name": "face", "id": 0, "color": [81, 120, 228]},
    ])

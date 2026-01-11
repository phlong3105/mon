#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""GTA5NighttimeFog dataset.

This module provides the GTA5NighttimeFog dataset for nighttime image dehazing.

References:
    - Data: https://github.com/jinyeying/nighttime_dehaze
"""

from __future__ import annotations

__all__ = [
    "GTA5NighttimeFog",
]

from ....api import *


@DATASETS.register()
class GTA5NighttimeFog(ImageDataset, RegistrableMixin):
    """GTA5NighttimeFog dataset."""
    
    _name      : str         = "gta5nighttimefog"
    _tasks     : list[Task]  = [Task.NTE, Task.LLE, Task.DEHAZE]
    _subset    : str         = None
    _splits    : list[Split] = [Split.TRAIN, Split.TEST]
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

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""RealNightHaze dataset.

This module provides the RealNightHaze dataset for nighttime image dehazing.
"""

from __future__ import annotations

__all__ = [
    "RealNightHaze",
]

from ....api import *


# ==============================================================================
# region DATASETS
# ==============================================================================

@DATASETS.register()
class RealNightHaze(ImageDataset, RegistrableMixin):
    """RealNightHaze dataset."""

    _name      : str         = "realnighthaze"
    _tasks     : list[Task]  = [Task.NTE, Task.LLE, Task.DEHAZE]
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

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion

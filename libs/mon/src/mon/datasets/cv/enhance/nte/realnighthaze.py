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

    name      : str         = "realnighthaze"
    tasks     : list[Task]  = [Task.NTE, Task.LLE, Task.DEHAZE]
    subset    : str         = None
    splits    : list[Split] = [Split.TEST]
    modalities: Modalities  = {
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
    classlist : ClassList   = None

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion

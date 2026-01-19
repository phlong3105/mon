#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""VV dataset.

This module provides the VV dataset for low-light enhancement tasks.
"""

from __future__ import annotations

__all__ = [
    "VV",
]

from ....api import *


# ==============================================================================
# region DATASETS
# ==============================================================================

@DATASETS.register()
class VV(ImageDataset, RegistrableMixin):
    """VV dataset."""

    _name      : str         = "vv"
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

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""UHD-LL dataset.

This module provides the UHD-LL dataset for low-light image enhancement.
"""

from __future__ import annotations

__all__ = [
    "UHDLL",
]

from ....api import *


# ==============================================================================
# region DATASETS
# ==============================================================================

@DATASETS.register()
class UHDLL(ImageDataset, RegistrableMixin):
    """UHD-LL dataset."""

    _name      : str         = "uhdll"
    _tasks     : list[Task]  = [Task.LLE]
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

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion

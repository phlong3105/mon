#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""UHD dataset.

This module provides the UHD dataset for low-light enhancement.
"""

from __future__ import annotations

__all__ = [
    "UHD4K",
    "UHD8K",
]

from ....api import *


# ==============================================================================
# region DATASETS
# ==============================================================================

# @DATASETS.register()
class UHD4K(ImageDataset, RegistrableMixin):
    """UHD-4K dataset."""

    _name      : str         = "uhd4k"
    _tasks     : list[Task]  = [Task.LLE]
    _subset    : str         = "4k"
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


# @DATASETS.register()
class UHD8K(ImageDataset, RegistrableMixin):
    """UHD-8K dataset."""

    _name      : str         = "uhd8k"
    _tasks     : list[Task]  = [Task.LLE]
    _subset    : str         = "8k"
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

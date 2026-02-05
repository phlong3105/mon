#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""DICM dataset.

This module provides the DICM dataset for low-light image enhancement.
"""

from __future__ import annotations

__all__ = [
    "DICM",
]

from ....api import *


# ==============================================================================
# region DATASETS
# ==============================================================================

@DATASETS.register(name="dicm")
class DICM(ImageDataset, RegistrableMixin):
    """DICM dataset."""

    name      : str         = "dicm"
    tasks     : list[Task]  = [Task.LLE]
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

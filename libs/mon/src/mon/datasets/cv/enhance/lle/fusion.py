#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Fusion dataset.

This module provides the Fusion dataset for low-light enhancement.
"""

from __future__ import annotations

__all__ = [
    "Fusion",
]

from ....api import *


# ==============================================================================
# region DATASETS
# ==============================================================================

@DATASETS.register(name="fusion")
class Fusion(ImageDataset, RegistrableMixin):
    """Fusion dataset."""

    name      : str         = "fusion"
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

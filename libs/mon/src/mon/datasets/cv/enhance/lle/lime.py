#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""LIME dataset.

This module provides the LIME dataset for low-light image enhancement.
"""

from __future__ import annotations

__all__ = [
    "LIME",
]

from ....api import *


# ==============================================================================
# region DATASETS
# ==============================================================================

@DATASETS.register(name="lime")
class LIME(ImageDataset, RegistrableMixin):
    """LIME dataset."""

    name      : str         = "lime"
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

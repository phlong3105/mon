#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""NPE dataset.

This module provides the NPE dataset for low-light image enhancement.
"""

from __future__ import annotations

__all__ = [
    "NPE",
]

from ....api import *


# ==============================================================================
# region DATASETS
# ==============================================================================

@DATASETS.register()
class NPE(ImageDataset, RegistrableMixin):
    """NPE dataset."""

    name      : str         = "npe"
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

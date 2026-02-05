#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""FlareReal800 dataset.

This module provides the FlareReal800 dataset for image de-flaring.
"""

from __future__ import annotations

__all__ = [
    "FlareReal800",
]

from ....api import *


# ==============================================================================
# region DATASETS
# ==============================================================================

@DATASETS.register()
class FlareReal800(ImageDataset, RegistrableMixin):
    """FlareReal800 dataset."""

    name      : str         = "flarereal800"
    tasks     : list[Task]  = [Task.DEFLARE]
    subset    : str         = None
    splits    : list[Split] = [Split.TRAIN, Split.VAL]
    modalities: Modalities  = {
        "image": Modality(
            name    = "image",
            type    = "image",
            module  = Image,
            train   = True,
            test    = True,
            primary = True,
        ),
        "ref"  : Modality(
            name    = "ref",
            type    = "image",
            module  = Image,
            train   = True,
            test    = False,
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

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""GT-Rain dataset.

This module provides the GT-Rain dataset for image de-raining.
"""

from __future__ import annotations

__all__ = [
    "GTRain",
]

from ....api import *


# ==============================================================================
# region DATASETS
# ==============================================================================

@DATASETS.register()
class GTRain(ImageDataset, RegistrableMixin):
    """GTRain dataset."""

    name      : str         = "gtrain"
    tasks     : list[Task]  = [Task.DERAIN]
    subset    : str         = None
    splits    : list[Split] = [Split.TRAIN, Split.VAL, Split.TEST]
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

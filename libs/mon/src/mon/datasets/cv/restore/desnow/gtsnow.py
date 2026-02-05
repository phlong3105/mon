#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""GT-Snow dataset.

This module provides GT-Snow dataset for image de-snowing.
"""

from __future__ import annotations

__all__ = [
    "GTSnow",
]

from ....api import *


# ==============================================================================
# region DATASETS
# ==============================================================================

@DATASETS.register()
class GTSnow(ImageDataset, RegistrableMixin):
    """GTSnow dataset."""

    name      : str         = "gtsnow"
    tasks     : list[Task]  = [Task.DESNOW]
    subset    : str         = None
    splits    : list[Split] = [Split.TRAIN]
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

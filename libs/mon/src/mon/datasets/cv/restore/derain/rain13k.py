#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Rain13K dataset.

This module provides the Rain13K dataset for image de-raining.
"""

from __future__ import annotations

__all__ = [
    "Rain13K",
]

from ....api import *


# ==============================================================================
# region DATASETS
# ==============================================================================

@DATASETS.register()
class Rain13K(ImageDataset, RegistrableMixin):
    """Rain13K dataset."""

    name      : str         = "rain13k"
    tasks     : list[Task]  = [Task.DERAIN]
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

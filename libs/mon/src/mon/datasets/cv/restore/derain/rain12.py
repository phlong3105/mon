#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Rain12 dataset.

This module provides the Rain12 dataset for image de-raining.
"""

from __future__ import annotations

__all__ = [
    "Rain12",
]

from ....api import *


# ==============================================================================
# region DATASETS
# ==============================================================================

@DATASETS.register()
class Rain12(ImageDataset, RegistrableMixin):
    """Rain12 dataset."""

    name      : str         = "rain12"
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

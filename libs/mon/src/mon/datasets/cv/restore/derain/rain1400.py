#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Rain1400 dataset.

This module provides the Rain1400 dataset for image de-raining.
"""

from __future__ import annotations

__all__ = [
    "Rain1400",
]

from ....api import *


# ==============================================================================
# region DATASETS
# ==============================================================================

@DATASETS.register()
class Rain1400(ImageDataset, RegistrableMixin):
    """Rain1400 dataset."""

    name      : str         = "rain1400"
    tasks     : list[Task]  = [Task.DERAIN]
    subset    : str         = None
    splits    : list[Split] = [Split.TRAIN, Split.TEST]
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

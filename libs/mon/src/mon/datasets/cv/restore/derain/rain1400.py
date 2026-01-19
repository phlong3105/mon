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

    _name      : str         = "rain1400"
    _tasks     : list[Task]  = [Task.DERAIN]
    _subset    : str         = None
    _splits    : list[Split] = [Split.TRAIN, Split.TEST]
    _modalities: Modalities  = {
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
    _classlist : ClassList   = None

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion

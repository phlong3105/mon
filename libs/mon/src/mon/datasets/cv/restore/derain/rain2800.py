#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Rain2800 dataset.

This module provides the Rain2800 dataset for image de-raining.
"""

from __future__ import annotations

__all__ = [
    "Rain2800",
]

from ....api import *


# ==============================================================================
# region DATASETS
# ==============================================================================

@DATASETS.register()
class Rain2800(ImageDataset, RegistrableMixin):
    """Rain2800 dataset."""

    _name      : str         = "rain2800"
    _tasks     : list[Task]  = [Task.DERAIN]
    _subset    : str         = None
    _splits    : list[Split] = [Split.TEST]
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

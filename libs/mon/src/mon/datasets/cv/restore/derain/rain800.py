#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Rain800 dataset.

This module provides the Rain800 dataset for image de-raining.
"""

from __future__ import annotations

__all__ = [
    "Rain800",
]

from ....api import *


@DATASETS.register()
class Rain800(ImageDataset, RegistrableMixin):
    """Rain800 dataset."""

    _name      : str         = "rain800"
    _tasks     : list[Task]  = [Task.DERAIN]
    _subset    : str         = "rain800"
    _splits    : list[Split] = [Split.TRAIN, Split.VAL, Split.TEST]
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


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion

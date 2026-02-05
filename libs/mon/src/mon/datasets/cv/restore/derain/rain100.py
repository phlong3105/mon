#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Rain100 dataset.

This module provides the Rain100 dataset for image de-raining.
"""

from __future__ import annotations

__all__ = [
    "Rain100",
    "Rain100H",
    "Rain100L",
]

from ....api import *


# ==============================================================================
# region DATASETS
# ==============================================================================

@DATASETS.register()
class Rain100(ImageDataset, RegistrableMixin):
    """Rain100 dataset."""

    name      : str         = "rain100"
    tasks     : list[Task]  = [Task.DERAIN]
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
        "ref"  : Modality(
            name    = "ref",
            type    = "image",
            module  = Image,
            train   = True,
            test    = True,
        ),
    }
    classlist : ClassList   = None


@DATASETS.register()
class Rain100H(ImageDataset, RegistrableMixin):
    """Rain100H dataset."""

    name      : str         = "rain100h"
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


@DATASETS.register()
class Rain100L(ImageDataset, RegistrableMixin):
    """Rain100L dataset."""

    name      : str         = "rain100l"
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

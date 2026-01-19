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


@DATASETS.register()
class Rain100(ImageDataset, RegistrableMixin):
    """Rain100 dataset."""

    _name      : str         = "rain100"
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


@DATASETS.register()
class Rain100H(ImageDataset, RegistrableMixin):
    """Rain100H dataset."""

    _name      : str         = "rain100h"
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


@DATASETS.register()
class Rain100L(ImageDataset, RegistrableMixin):
    """Rain100L dataset."""

    _name      : str         = "rain100l"
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


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion

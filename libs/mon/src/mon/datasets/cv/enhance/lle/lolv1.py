#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""LOL-v1 dataset.

This module provides the LOL-v1 dataset for low-light image enhancement.
"""

from __future__ import annotations

__all__ = [
    "LOLv1",
]

from ....api import *


@DATASETS.register(name="lolv1")
class LOLv1(ImageDataset, RegistrableMixin):
    """LOL-v1 dataset."""

    _name      : str         = "lolv1"
    _tasks     : list[Task]  = [Task.LLE]
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
        "depth": Modality(
            name    = DepthName,
            type    = "image",
            module  = DefaultDepthMap,
            train   = True,
            test    = True,
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

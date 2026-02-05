#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Dense-NH-Haze dataset.

This module provides the Dense-NH-Haze dataset for image de-hazing.
"""

from __future__ import annotations

__all__ = [
    "DenseNHHaze",
]

from ....api import *


# ==============================================================================
# region DATASETS
# ==============================================================================

@DATASETS.register()
class DenseNHHaze(ImageDataset, RegistrableMixin):
    """Dense-NH-Haze dataset."""

    name      : str         = "densenhhaze"
    tasks     : list[Task]  = [Task.DEHAZE]
    subset    : str         = None
    splits    : list[Split] = [Split.TRAIN, Split.VAL, Split.TEST]
    modalities: Modalities  = {
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
    classlist : ClassList   = None

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""O-Haze dataset.

This module provides the O-Haze dataset for image de-hazing.
"""

from __future__ import annotations

__all__ = [
    "OHaze",
]

from ....api import *


@DATASETS.register()
class OHaze(ImageDataset, RegistrableMixin):
    """O-Haze dataset."""

    _name      : str         = "ohaze"
    _tasks     : list[Task]  = [Task.DEHAZE]
    _subset    : str         = None
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

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""LightEffect dataset.

This module provides the LightEffect dataset for image de-flaring.
"""

from __future__ import annotations

__all__ = [
    "LightEffect",
]

from ....api import *


# ==============================================================================
# region DATASETS
# ==============================================================================

@DATASETS.register()
class LightEffect(ImageDataset, RegistrableMixin):
    """LightEffect dataset."""

    name      : str         = "lighteffect"
    tasks     : list[Task]  = [Task.DEFLARE]
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
    }
    classlist : ClassList   = None

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion

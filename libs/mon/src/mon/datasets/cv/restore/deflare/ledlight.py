#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""LEDLight dataset.

This module provides the LEDLight dataset for image de-flaring.
"""

from __future__ import annotations

__all__ = [
    "LEDLight",
]

from ....api import *


# ==============================================================================
# region DATASETS
# ==============================================================================

@DATASETS.register()
class LEDLight(ImageDataset, RegistrableMixin):
    """LEDLight dataset."""

    name      : str         = "ledlight"
    tasks     : list[Task]  = [Task.DEFLARE]
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

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion

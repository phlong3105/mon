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


@DATASETS.register()
class LEDLight(ImageDataset, RegistrableMixin):
    """LEDLight dataset."""

    _name      : str         = "ledlight"
    _tasks     : list[Task]  = [Task.DEFLARE]
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


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion

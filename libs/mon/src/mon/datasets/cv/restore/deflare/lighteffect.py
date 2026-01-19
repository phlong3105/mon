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


@DATASETS.register()
class LightEffect(ImageDataset, RegistrableMixin):
    """LightEffect dataset."""

    _name      : str         = "lighteffect"
    _tasks     : list[Task]  = [Task.DEFLARE]
    _subset    : str         = None
    _splits    : list[Split] = [Split.TRAIN]
    _modalities: Modalities  = {
        "image": Modality(
            name    = "image",
            type    = "image",
            module  = Image,
            train   = True,
            test    = True,
            primary = True,
        ),
    }
    _classlist : ClassList   = None


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion

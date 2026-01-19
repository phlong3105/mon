#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Fusion dataset.

This module provides the Fusion dataset for low-light enhancement.
"""

from __future__ import annotations

__all__ = [
    "Fusion",
]

from ....api import *


@DATASETS.register(name="fusion")
class Fusion(ImageDataset, RegistrableMixin):
    """Fusion dataset."""

    _name      : str         = "fusion"
    _tasks     : list[Task]  = [Task.LLE]
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
        "depth": Modality(
            name    = DepthName,
            type    = "image",
            module  = DefaultDepthMap,
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

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Speed benchmarking datasets.

This module provides dataset classes for speed benchmarking.
"""

from __future__ import annotations

__all__ = [
    "Speed10",
    "Speed1K",
]

from ..api import *


# ==============================================================================
# region DATASETS
# ==============================================================================

@DATASETS.register()
class Speed10(ImageDataset, RegistrableMixin):
    """Speed10 dataset."""

    name      : str         = "speed10"
    tasks     : list[Task]  = [Task.LLE]
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
        "depth": Modality(
            name    = DepthName,
            type    = "image",
            module  = DefaultDepthMap,
            train   = True,
            test    = True,
        ),
    }
    classlist : ClassList   = None


@DATASETS.register()
class Speed1K(ImageDataset, RegistrableMixin):
    """Speed1K dataset."""

    name      : str         = "speed1k"
    tasks     : list[Task]  = [Task.LLE]
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
        "depth": Modality(
            name    = DepthName,
            type    = "image",
            module  = DefaultDepthMap,
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

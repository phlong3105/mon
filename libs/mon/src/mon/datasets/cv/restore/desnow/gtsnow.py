#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""GT-Snow dataset.

This module provides GT-Snow dataset for image de-snowing.
"""

from __future__ import annotations

__all__ = [
    "GTSnow",
]

from ....api import *


@DATASETS.register()
class GTSnow(ImageDataset, RegistrableMixin):
    """GTSnow dataset."""

    _name      : str         = "gtsnow"
    _tasks     : list[Task]  = [Task.DESNOW]
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
        "ref"  : Modality(
            name    = "ref",
            type    = "image",
            module  = Image,
            train   = True,
            test    = False,
        ),
    }
    _classlist : ClassList   = None


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion

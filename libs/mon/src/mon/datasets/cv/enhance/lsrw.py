#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""LSRW dataset.

This module provides the LSRW dataset for low-light image enhancement.
"""

from __future__ import annotations

__all__ = [
    "LSRW",
]

from ...api import *


# ==============================================================================
# region DATASETS
# ==============================================================================

@DATASETS.register()
class LSRW(ImageDataset, RegistrableMixin):
    """LSRW dataset."""

    name: str = "lsrw"
    tasks: list[Task] = [Task.LLE]
    subroot: str = None
    splits: list[Split] = [Split.TRAIN, Split.TEST]
    modalities: Modalities = {
        "image": Modality(
            name="image",
            type="image",
            module=Image,
            train=True,
            test=True,
            primary=True,
        ),
        "depth": Modality(
            name=DepthName,
            type="image",
            module=DefaultDepthMap,
            train=True,
            test=True,
        ),
        "ref": Modality(
            name="ref",
            type="image",
            module=Image,
            train=True,
            test=True,
        ),
    }
    classlist: ClassList = None


# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion

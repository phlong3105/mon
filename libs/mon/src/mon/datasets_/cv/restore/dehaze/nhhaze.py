#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""NH-Haze dataset.

This module provides the NH-Haze dataset for image de-hazing.
"""

from __future__ import annotations

__all__ = [
    "NHHaze",
]

from ....api import *


# ==============================================================================
# region DATASETS
# ==============================================================================

@DATASETS.register(name="nhhaze")
class NHHaze(ImageDataset, RegistrableMixin):
    """NH-Haze dataset."""

    name: str = "nhhaze"
    tasks: list[Task] = [Task.DEHAZE]
    subroot: str = None
    splits: list[Split] = [Split.TRAIN, Split.VAL, Split.TEST]
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

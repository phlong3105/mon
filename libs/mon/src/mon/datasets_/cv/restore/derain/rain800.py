#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Rain800 dataset.

This module provides the Rain800 dataset for image de-raining.
"""

from __future__ import annotations

__all__ = [
    "Rain800",
]

from ....api import *


# ==============================================================================
# region DATASETS
# ==============================================================================

@DATASETS.register(name="rain800")
class Rain800(ImageDataset, RegistrableMixin):
    """Rain800 dataset."""

    name: str = "rain800"
    tasks: list[Task] = [Task.DERAIN]
    subroot: str = "rain800"
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

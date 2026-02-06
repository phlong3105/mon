#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Real-LOL-Blur dataset.

This module provides the Real-LOL-Blur dataset for image de-blurring and
low-light enhancement.
"""

from __future__ import annotations

__all__ = [
    "RealLOLBlur",
]

from ....api import *


# ==============================================================================
# region DATASETS
# ==============================================================================

@DATASETS.register(name="reallolblur")
class RealLOLBlur(ImageDataset, RegistrableMixin):
    """Real-LOL-Blur dataset."""

    name: str = "reallolblur"
    tasks: list[Task] = [Task.DEBLUR, Task.LLE]
    subroot: str = None
    splits: list[Split] = [Split.TEST]
    modalities: Modalities = {
        "image": Modality(
            name="image",
            type="image",
            module=Image,
            train=True,
            test=True,
            primary=True,
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

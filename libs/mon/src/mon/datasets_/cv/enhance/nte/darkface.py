#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""DarkFace dataset.

This module provides the DarkFace dataset for nighttime face enhancement and
detection.
"""

from __future__ import annotations

__all__ = [
    "DarkFace",
]

from ....api import *


# ==============================================================================
# region DATASETS
# ==============================================================================

@DATASETS.register(name="darkface")
class DarkFace(ImageDataset, RegistrableMixin):
    """DarkFace dataset."""

    name: str = "darkface"
    tasks: list[Task] = [Task.NTE, Task.LLE, Task.DETECT]
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
        "depth": Modality(
            name=DepthName,
            type="image",
            module=DefaultDepthMap,
            train=True,
            test=True,
        ),
    }
    classlist: ClassList = ClassList(
        [
            {"name": "face", "id": 0, "color": [81, 120, 228]},
        ],
    )


# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion

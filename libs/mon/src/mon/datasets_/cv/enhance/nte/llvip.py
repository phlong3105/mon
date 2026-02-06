#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""LLVIP dataset.

This module provides the LLVIP dataset for nighttime object detection.

References:
    - Paper: "LLVIP: A Visible-infrared Paired Dataset for Low-light Vision,"
      ICCV 2021.
    - Data: https://github.com/bupt-ai-cz/LLVIP
"""

from __future__ import annotations

__all__ = [
    "LLVIP",
]

from ....api import *


# ==============================================================================
# region DATASETS
# ==============================================================================

@DATASETS.register(name="llvip")
class LLVIP(ImageDataset, RegistrableMixin):
    """LLVIP dataset."""

    name: str = "llvip"
    tasks: list[Task] = [Task.NTE, Task.DETECT]
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
        "infrared": Modality(
            name=InfraredName,
            type="mask",
            module=DefaultInfraredMap,
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

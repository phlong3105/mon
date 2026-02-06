#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""UHD-LL dataset.

This module provides the UHD-LL dataset for low-light image enhancement.

References
    - Paper: "Embedding Fourier for Ultra-High-Definition Low-Light Image
      Enhancement," ICLR 2023.
    - Code: https://github.com/Li-Chongyi/UHDFour_code
"""

from __future__ import annotations

__all__ = [
    "UHD_LL",
]

from ...api import *


# ==============================================================================
# region DATASETS
# ==============================================================================

@DATASETS.register(name="uhdll")
class UHD_LL(ImageDataset, RegistrableMixin):
    """UHD-LL dataset.

    Include a total of 2,000 pairs for training and 150 pairs for testing.

    References
        - Paper: "Embedding Fourier for Ultra-High-Definition Low-Light Image
          Enhancement," ICLR 2023.
        - Code: https://github.com/Li-Chongyi/UHDFour_code
    """

    name: str = "uhdll"
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

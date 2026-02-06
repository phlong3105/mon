#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""LOL-Blur dataset.

This module provides LOL-Blur dataset for image deblurring, denoising, and
low-light enhancement.
"""

from __future__ import annotations

__all__ = [
    "LOLBlurB",
    "LOLBlurBN",
    "LOLBlurL",
    "LOLBlurLB",
    "LOLBlurLBN",
    "LOLBlurN",
]

import abc

from ....api import *


# ==============================================================================
# region DATASETS
# ==============================================================================

class LOLBlur(ImageDataset, abc.ABC):
    """LOL-Blur dataset."""

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


@DATASETS.register(name="lolblurb")
class LOLBlurB(LOLBlur, RegistrableMixin):
    """LOL-Blur-B (Blur) dataset."""

    name: str = "lolblurb"
    tasks: list[Task] = [Task.DEBLUR]
    subroot: str = "b"


@DATASETS.register(name="lolblurbn")
class LOLBlurBN(LOLBlur, RegistrableMixin):
    """LOL-Blur-BN (Blur + Noise) dataset."""

    name: str = "lolblurbn"
    tasks: list[Task] = [Task.DEBLUR, Task.DENOISE]
    subroot: str = "bn"


@DATASETS.register(name="lolblurl")
class LOLBlurL(LOLBlur, RegistrableMixin):
    """LOL-Blur-L (Low-Light) dataset."""

    name: str = "lolblurl"
    tasks: list[Task] = [Task.LLE]
    subroot: str = "l"


@DATASETS.register(name="lolblurlb")
class LOLBlurLB(LOLBlur, RegistrableMixin):
    """LOL-Blur-LB (Low-Light + Blur) dataset."""

    name: str = "lolblurlb"
    tasks: list[Task] = [Task.DEBLUR, Task.LLE]
    subroot: str = "lb"


@DATASETS.register(name="lolblurlbn")
class LOLBlurLBN(LOLBlur, RegistrableMixin):
    """LOL-Blur-LBN (Low-Light + Blur + Noise) dataset."""

    name: str = "lolblurlbn"
    tasks: list[Task] = [Task.DEBLUR, Task.DENOISE, Task.LLE]
    subroot: str = "lbn"


@DATASETS.register(name="lolblurn")
class LOLBlurN(LOLBlur, RegistrableMixin):
    """LOL-Blur-N (Noise) dataset."""

    name: str = "lolblurn"
    tasks: list[Task] = [Task.DENOISE]
    subroot: str = "n"


# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""FiveK dataset.

This module provides the FiveK dataset and its variants for image retouching.
"""

__all__ = [
    "FiveK",
    "FiveKA",
    "FiveKB",
    "FiveKC",
    "FiveKD",
    "FiveKE",
]

from ....api import *


# ==============================================================================
# region DATASETS
# ==============================================================================

@DATASETS.register(name="fivek")
class FiveK(ImageDataset):
    """FiveK dataset."""

    subset    : str         = "fivek"
    tasks     : list[Task]  = [Task.RETOUCH, Task.EXPOSURE, Task.LLE]
    splits    : list[Split] = [Split.TRAIN, Split.VAL, Split.TEST]
    modalities: Modalities  = {
        "image": Modality(name="image",   type="image", module=Image,           train=True, test=True, primary=True),
        "depth": Modality(name=DepthName, type="image", module=DefaultDepthMap, train=True, test=True),
        "ref"  : Modality(name="ref_c",   type="image", module=Image,           train=True, test=True),
        "ref_a": Modality(name="ref_a",   type="image", module=Image,           train=True, test=True),
        "ref_b": Modality(name="ref_b",   type="image", module=Image,           train=True, test=True),
        "ref_c": Modality(name="ref_c",   type="image", module=Image,           train=True, test=True),
        "ref_d": Modality(name="ref_d",   type="image", module=Image,           train=True, test=True),
        "ref_e": Modality(name="ref_e",   type="image", module=Image,           train=True, test=True),
    }
    classlist : ClassList   = None


@DATASETS.register(name="fiveka")
class FiveKA(ImageDataset):
    """FiveK-A dataset."""

    subset    : str         = "fivek"
    tasks     : list[Task]  = [Task.RETOUCH, Task.EXPOSURE, Task.LLE]
    splits    : list[Split] = [Split.TRAIN, Split.VAL, Split.TEST]
    modalities: Modalities  = {
        "image": Modality(name="image",   type="image", module=Image,           train=True, test=True, primary=True),
        "depth": Modality(name=DepthName, type="image", module=DefaultDepthMap, train=True, test=True),
        "ref"  : Modality(name="ref_a",   type="image", module=Image,           train=True, test=True),
    }
    classlist : ClassList   = None


@DATASETS.register(name="fivekb")
class FiveKB(FiveKA):
    """FiveK-B dataset."""

    modalities: Modalities = {
        "image": Modality(name="image",   type="image", module=Image,           train=True, test=True, primary=True),
        "depth": Modality(name=DepthName, type="image", module=DefaultDepthMap, train=True, test=True),
        "ref"  : Modality(name="ref_b",   type="image", module=Image,           train=True, test=True),
    }


@DATASETS.register(name="fivekc")
class FiveKC(FiveKA):
    """FiveK-C dataset."""

    modalities: Modalities = {
        "image": Modality(name="image",   type="image", module=Image,           train=True, test=True, primary=True),
        "depth": Modality(name=DepthName, type="image", module=DefaultDepthMap, train=True, test=True),
        "ref"  : Modality(name="ref_c",   type="image", module=Image,           train=True, test=True),
    }


@DATASETS.register(name="fivekd")
class FiveKD(FiveKA):
    """FiveK-D dataset."""

    modalities: Modalities = {
        "image": Modality(name="image",   type="image", module=Image,           train=True, test=True, primary=True),
        "depth": Modality(name=DepthName, type="image", module=DefaultDepthMap, train=True, test=True),
        "ref"  : Modality(name="ref_d",   type="image", module=Image,           train=True, test=True),
    }


@DATASETS.register(name="fiveke")
class FiveKE(FiveKA):
    """FiveK-E dataset."""

    modalities: Modalities = {
        "image": Modality(name="image",   type="image", module=Image,           train=True, test=True, primary=True),
        "depth": Modality(name=DepthName, type="image", module=DefaultDepthMap, train=True, test=True),
        "ref"  : Modality(name="ref_e",   type="image", module=Image,           train=True, test=True),
    }

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""LOL-v2 dataset.

This module provides the LOL-v2 dataset for low-light image enhancement.
"""

from __future__ import annotations

__all__ = [
    "LOLv2Real",
    "LOLv2Syn",
]

from ...api import *


# ==============================================================================
# region DATASETS
# ==============================================================================

@DATASETS.register(name="lolv2_real")
class LOLv2Real(ImageDataset, RegistrableMixin):
    """LOL-v2 Real dataset."""

    name: str = "lolv2_real"
    tasks: list[Task] = [Task.LLE]
    subroot: str = "real"
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


@DATASETS.register(name="lolv2_syn")
class LOLv2Syn(ImageDataset, RegistrableMixin):
    """LOL-v2 Synthetic dataset."""

    name: str = "lolv2_syn"
    tasks: list[Task] = [Task.LLE]
    subroot: str = "syn"
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

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Unpaired low-light image enhancement datasets.

This module provides several sub-datasets usually gathered under the umbrella
of the "Unpaired" dataset.
"""

from __future__ import annotations

__all__ = [
    "DICM",
    "Fusion",
    "LIME",
    "MEF",
    "NPE",
    "VV",
]

from ...api import *


# ==============================================================================
# region DATASETS
# ==============================================================================

@DATASETS.register(name="dicm")
class DICM(ImageDataset, RegistrableMixin):
    """DICM dataset."""

    name: str = "dicm"
    tasks: list[Task] = [Task.LLE]
    subroot: str = "dicm"
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
    classlist: ClassList | None = None


@DATASETS.register(name="fusion")
class Fusion(ImageDataset, RegistrableMixin):
    """Fusion dataset."""

    name: str = "fusion"
    tasks: list[Task] = [Task.LLE]
    subroot: str = "fusion"
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
    classlist: ClassList = None


@DATASETS.register(name="lime")
class LIME(ImageDataset, RegistrableMixin):
    """LIME dataset."""

    name: str = "lime"
    tasks: list[Task] = [Task.LLE]
    subroot: str = "lime"
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
    classlist: ClassList = None


@DATASETS.register(name="mef")
class MEF(ImageDataset, RegistrableMixin):
    """MEF dataset."""

    name: str = "mef"
    tasks: list[Task] = [Task.LLE]
    subroot: str = "mef"
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
    classlist: ClassList = None


@DATASETS.register(name="npe")
class NPE(ImageDataset, RegistrableMixin):
    """NPE dataset."""

    name: str = "npe"
    tasks: list[Task] = [Task.LLE]
    subroot: str = "npe"
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
    classlist: ClassList = None


@DATASETS.register(name="vv")
class VV(ImageDataset, RegistrableMixin):
    """VV dataset."""

    name: str = "vv"
    tasks: list[Task] = [Task.LLE]
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
    classlist: ClassList = None


# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion

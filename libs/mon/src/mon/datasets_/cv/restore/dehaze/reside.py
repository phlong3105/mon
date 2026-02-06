#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""RESIDE datasets.

This module provides the theRESIDE dataset for image de-hazing.
"""

from __future__ import annotations

__all__ = [
    "RESIDE_HSTSReal",
    "RESIDE_HSTSSyn",
    "RESIDE_ITS",
    "RESIDE_OTS",
    "RESIDE_RTTS",
    "RESIDE_SOTSIndoor",
    "RESIDE_SOTSOutdoor",
    "RESIDE_URHI",
]

from ....api import *


# ==============================================================================
# region DATASETS
# ==============================================================================

@DATASETS.register(name="reside_hstsreal")
class RESIDE_HSTSReal(ImageDataset, RegistrableMixin):
    """RESIDE-HSTS-Real dataset."""

    name: str = "reside_hstsreal"
    tasks: list[Task] = [Task.DEHAZE]
    subroot: str = "hsts/real"
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


@DATASETS.register(name="reside_hstssyn")
class RESIDE_HSTSSyn(ImageDataset, RegistrableMixin):
    """RESIDE-HSTS-Synthetic dataset."""

    name: str = "reside_hstssyn"
    tasks: list[Task] = [Task.DEHAZE]
    subroot: str = "hsts/synthetic"
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
        "ref": Modality(
            name="ref",
            type="image",
            module=Image,
            train=True,
            test=True,
        ),
    }
    classlist: ClassList = None


@DATASETS.register(name="reside_its")
class RESIDE_ITS(ImageDataset, RegistrableMixin):
    """RESIDE-ITS dataset."""

    name: str = "reside_its"
    tasks: list[Task] = [Task.DEHAZE]
    subroot: str = "its"
    splits: list[Split] = [Split.TRAIN]
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
            test=False,
        ),
    }
    classlist: ClassList = None


@DATASETS.register(name="reside_ots")
class RESIDE_OTS(ImageDataset, RegistrableMixin):
    """RESIDE-OTS dataset."""

    name: str = "reside_ots"
    tasks: list[Task] = [Task.DEHAZE]
    subroot: str = "ots"
    splits: list[Split] = [Split.TRAIN]
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
            test=False,
        ),
    }
    classlist: ClassList = None


@DATASETS.register(name="reside_rtts")
class RESIDE_RTTS(ImageDataset, RegistrableMixin):
    """RESIDE-RTTS dataset."""

    name: str = "reside_rtts"
    tasks: list[Task] = [Task.DEHAZE, Task.DETECT]
    subroot: str = "rtts"
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


@DATASETS.register(name="reside_sotsindoor")
class RESIDE_SOTSIndoor(ImageDataset, RegistrableMixin):
    """RESIDE-SOTS-Indoor dataset."""

    name: str = "reside_sotsindoor"
    tasks: list[Task] = [Task.DEHAZE]
    subroot: str = "sots/indoor"
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
        "ref": Modality(
            name="ref",
            type="image",
            module=Image,
            train=True,
            test=True,
        ),
    }
    classlist: ClassList = None


@DATASETS.register(name="reside_sotsoutdoor")
class RESIDE_SOTSOutdoor(ImageDataset, RegistrableMixin):
    """RESIDE-SOTS-Outdoor dataset."""

    name: str = "reside_sotsoutdoor"
    tasks: list[Task] = [Task.DEHAZE]
    subroot: str = "sots/outdoor"
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
        "ref": Modality(
            name="ref",
            type="image",
            module=Image,
            train=True,
            test=True,
        ),
    }
    classlist: ClassList = None


@DATASETS.register(name="reside_urhi")
class RESIDE_URHI(ImageDataset, RegistrableMixin):
    """RESIDE-URHI dataset."""

    name: str = "reside_urhi"
    tasks: list[Task] = [Task.DEHAZE]
    subroot: str = "urhi"
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

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


@DATASETS.register()
class RESIDE_HSTSReal(ImageDataset, RegistrableMixin):
    """RESIDE-HSTS-Real dataset."""

    _name      : str         = "reside_hstsreal"
    _tasks     : list[Task]  = [Task.DEHAZE]
    _subset    : str         = "hsts/real"
    _splits    : list[Split] = [Split.TEST]
    _modalities: Modalities  = {
        "image": Modality(
            name    = "image",
            type    = "image",
            module  = Image,
            train   = True,
            test    = True,
            primary = True,
        ),
        "depth": Modality(
            name    = DepthName,
            type    = "image",
            module  = DefaultDepthMap,
            train   = True,
            test    = True,
        ),
    }
    _classlist : ClassList   = None
    
        
@DATASETS.register()
class RESIDE_HSTSSyn(ImageDataset, RegistrableMixin):
    """RESIDE-HSTS-Synthetic dataset."""

    _name      : str         = "reside_hstssyn"
    _tasks     : list[Task]  = [Task.DEHAZE]
    _subset    : str         = "hsts/synthetic"
    _splits    : list[Split] = [Split.TEST]
    _modalities: Modalities  = {
        "image": Modality(
            name    = "image",
            type    = "image",
            module  = Image,
            train   = True,
            test    = True,
            primary = True,
        ),
        "depth": Modality(
            name    = DepthName,
            type    = "image",
            module  = DefaultDepthMap,
            train   = True,
            test    = True,
        ),
        "ref"  : Modality(
            name    = "ref",
            type    = "image",
            module  = Image,
            train   = True,
            test    = True,
        ),
    }
    _classlist : ClassList   = None
    

@DATASETS.register()
class RESIDE_ITS(ImageDataset, RegistrableMixin):
    """RESIDE-ITS dataset."""

    _name      : str         = "reside_its"
    _tasks     : list[Task]  = [Task.DEHAZE]
    _subset    : str         = "its"
    _splits    : list[Split] = [Split.TRAIN]
    _modalities: Modalities  = {
        "image": Modality(
            name    = "image",
            type    = "image",
            module  = Image,
            train   = True,
            test    = True,
            primary = True,
        ),
        "depth": Modality(
            name    = DepthName,
            type    = "image",
            module  = DefaultDepthMap,
            train   = True,
            test    = True,
        ),
        "ref"  : Modality(
            name    = "ref",
            type    = "image",
            module  = Image,
            train   = True,
            test    = False,
        ),
    }
    _classlist : ClassList   = None


@DATASETS.register()
class RESIDE_OTS(ImageDataset, RegistrableMixin):
    """RESIDE-OTS dataset."""

    _name      : str         = "reside_ots"
    _tasks     : list[Task]  = [Task.DEHAZE]
    _subset    : str         = "ots"
    _splits    : list[Split] = [Split.TRAIN]
    _modalities: Modalities  = {
        "image": Modality(
            name    = "image",
            type    = "image",
            module  = Image,
            train   = True,
            test    = True,
            primary = True,
        ),
        "depth": Modality(
            name    = DepthName,
            type    = "image",
            module  = DefaultDepthMap,
            train   = True,
            test    = True,
        ),
        "ref"  : Modality(
            name    = "ref",
            type    = "image",
            module  = Image,
            train   = True,
            test    = False,
        ),
    }
    _classlist : ClassList   = None


@DATASETS.register()
class RESIDE_RTTS(ImageDataset, RegistrableMixin):
    """RESIDE-RTTS dataset."""

    _name      : str         = "reside_rtts"
    _tasks     : list[Task]  = [Task.DEHAZE, Task.DETECT]
    _subset    : str         = "rtts"
    _splits    : list[Split] = [Split.TEST]
    _modalities: Modalities  = {
        "image": Modality(
            name    = "image",
            type    = "image",
            module  = Image,
            train   = True,
            test    = True,
            primary = True,
        ),
        "depth": Modality(
            name    = DepthName,
            type    = "image",
            module  = DefaultDepthMap,
            train   = True,
            test    = True,
        ),
    }
    _classlist : ClassList   = None
    
        

@DATASETS.register()
class RESIDE_SOTSIndoor(ImageDataset, RegistrableMixin):
    """RESIDE-SOTS-Indoor dataset."""

    _name      : str         = "reside_sotsindoor"
    _tasks     : list[Task]  = [Task.DEHAZE]
    _subset    : str         = "sots/indoor"
    _splits    : list[Split] = [Split.TEST]
    _modalities: Modalities  = {
        "image": Modality(
            name    = "image",
            type    = "image",
            module  = Image,
            train   = True,
            test    = True,
            primary = True,
        ),
        "depth": Modality(
            name    = DepthName,
            type    = "image",
            module  = DefaultDepthMap,
            train   = True,
            test    = True,
        ),
        "ref"  : Modality(
            name    = "ref",
            type    = "image",
            module  = Image,
            train   = True,
            test    = True,
        ),
    }
    _classlist : ClassList   = None


@DATASETS.register()
class RESIDE_SOTSOutdoor(ImageDataset, RegistrableMixin):
    """RESIDE-SOTS-Outdoor dataset."""

    _name      : str         = "reside_sotsoutdoor"
    _tasks     : list[Task]  = [Task.DEHAZE]
    _subset    : str         = "sots/outdoor"
    _splits    : list[Split] = [Split.TEST]
    _modalities: Modalities  = {
        "image": Modality(
            name    = "image",
            type    = "image",
            module  = Image,
            train   = True,
            test    = True,
            primary = True,
        ),
        "depth": Modality(
            name    = DepthName,
            type    = "image",
            module  = DefaultDepthMap,
            train   = True,
            test    = True,
        ),
        "ref"  : Modality(
            name    = "ref",
            type    = "image",
            module  = Image,
            train   = True,
            test    = True,
        ),
    }
    _classlist : ClassList   = None


@DATASETS.register()
class RESIDE_URHI(ImageDataset, RegistrableMixin):
    """RESIDE-URHI dataset."""

    _name      : str         = "reside_urhi"
    _tasks     : list[Task]  = [Task.DEHAZE]
    _subset    : str         = "urhi"
    _splits    : list[Split] = [Split.TEST]
    _modalities: Modalities  = {
        "image": Modality(
            name    = "image",
            type    = "image",
            module  = Image,
            train   = True,
            test    = True,
            primary = True,
        ),
        "depth": Modality(
            name    = DepthName,
            type    = "image",
            module  = DefaultDepthMap,
            train   = True,
            test    = True,
        ),
    }
    _classlist : ClassList   = None

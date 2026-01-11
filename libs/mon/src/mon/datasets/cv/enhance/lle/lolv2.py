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

from ....api import *


@DATASETS.register()
class LOLv2Real(ImageDataset, RegistrableMixin):
    """LOL-v2 Real dataset."""
    
    _name      : str         = "lolv2"
    _tasks     : list[Task]  = [Task.LLE]
    _subset    : str         = "real"
    _splits    : list[Split] = [Split.TRAIN, Split.TEST]
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
class LOLv2Syn(ImageDataset, RegistrableMixin):
    """LOL-v2 Synthetic dataset."""
    
    _name      : str         = "lolv2"
    _tasks     : list[Task]  = [Task.LLE]
    _subset    : str         = "syn"
    _splits    : list[Split] = [Split.TRAIN, Split.TEST]
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

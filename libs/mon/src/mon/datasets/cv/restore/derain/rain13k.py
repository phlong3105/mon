#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Rain13K dataset.

This module provides the Rain13K dataset for image de-raining.
"""

from __future__ import annotations

__all__ = [
    "Rain13K",
]

from ....api import *


@DATASETS.register()
class Rain13K(ImageDataset, RegistrableMixin):
    """Rain13K dataset."""

    _name      : str         = "rain13k"
    _tasks     : list[Task]  = [Task.DERAIN]
    _subset    : str         = None
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
        "ref"  : Modality(
            name    = "ref",
            type    = "image",
            module  = Image,
            train   = True,
            test    = False,
        ),
    }
    _classlist : ClassList   = None

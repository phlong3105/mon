#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""RCD dataset.

This module provides the Radiometry Correction Dataset (RCD) dataset for
exposure correction and multi-exposure fusion.

References:
    - Paper: "Unsupervised Exposure Correction," ECCV 2024.
    - Code: https://github.com/BeyondHeaven/uec_code
"""

from __future__ import annotations

__all__ = [
    "RCD",
]

from ....api import *


# ==============================================================================
# region DATASETS
# ==============================================================================

@DATASETS.register()
class RCD(ImageDataset, RegistrableMixin):
    """RCD dataset."""

    name      : str         = "rcd"
    tasks     : list[Task]  = [Task.EXPOSURE, Task.MEF]
    subroot    : str         = None
    splits    : list[Split] = [Split.TRAIN, Split.VAL, Split.TEST]
    modalities: Modalities  = {
        "image"      : Modality(
            name    = "image_ev_0",
            type    = "image",
            module  = Image,
            train   = True,
            test    = True,
            primary = True,
        ),
        "image_ev_n3": Modality(
            name    = "image_ev_n3",
            type    = "image",
            module  = Image,
            train   = True,
            test    = True,
        ),
        "image_ev_n2": Modality(
            name    = "image_ev_n2",
            type    = "image",
            module  = Image,
            train   = True,
            test    = True,
        ),
        "image_ev_n1": Modality(
            name    = "image_ev_n1",
            type    = "image",
            module  = Image,
            train   = True,
            test    = True,
        ),
        "image_ev_0" : Modality(
            name    = "image_ev_0",
            type    = "image",
            module  = Image,
            train   = True,
            test    = True,
        ),
        "image_ev_p1": Modality(
            name    = "image_ev_p1",
            type    = "image",
            module  = Image,
            train   = True,
            test    = True,
        ),
        "image_ev_p2": Modality(
            name    = "image_ev_p2",
            type    = "image",
            module  = Image,
            train   = True,
            test    = True,
        ),
        "image_ev_p3": Modality(
            name    = "image_ev_p3",
            type    = "image",
            module  = Image,
            train   = True,
            test    = True,
        ),
        "ref"        : Modality(
            name    = "ref",
            type    = "image",
            module  = Image,
            train   = True,
            test    = True,
        ),
    }
    classlist : ClassList   = None

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion

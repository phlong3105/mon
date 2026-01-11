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


class LOLBlur(ImageDataset, abc.ABC):
    """LOL-Blur dataset."""
    
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
class LOLBlurB(LOLBlur, RegistrableMixin):
    """LOL-Blur-B (Blur) dataset."""
    
    _name  : str        = "lolblurb"
    _tasks : list[Task] = [Task.DEBLUR]
    _subset: str        = "b"


@DATASETS.register()
class LOLBlurBN(LOLBlur, RegistrableMixin):
    """LOL-Blur-BN (Blur + Noise) dataset."""
    
    _name  : str        = "lolblurbn"
    _tasks : list[Task] = [Task.DEBLUR, Task.DENOISE]
    _subset: str        = "bn"


@DATASETS.register()
class LOLBlurL(LOLBlur, RegistrableMixin):
    """LOL-Blur-L (Low-Light) dataset."""
    
    _name  : str        = "lolblurl"
    _tasks : list[Task] = [Task.LLE]
    _subset: str        = "l"


@DATASETS.register()
class LOLBlurLB(LOLBlur, RegistrableMixin):
    """LOL-Blur-LB (Low-Light + Blur) dataset."""
    
    _name  : str        = "lolblurlb"
    _tasks : list[Task] = [Task.DEBLUR, Task.LLE]
    _subset: str        = "lb"


@DATASETS.register()
class LOLBlurLBN(LOLBlur, RegistrableMixin):
    """LOL-Blur-LBN (Low-Light + Blur + Noise) dataset."""
    
    _name  : str        = "lolblurlbn"
    _tasks : list[Task] = [Task.DEBLUR, Task.DENOISE, Task.LLE]
    _subset: str        = "lbn"


@DATASETS.register()
class LOLBlurN(LOLBlur, RegistrableMixin):
    """LOL-Blur-N (Noise) dataset."""
    
    _name  : str        = "lolblurn"
    _tasks : list[Task] = [Task.DENOISE]
    _subset: str        = "n"

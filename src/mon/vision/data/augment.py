#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This package implements data augmentation functions by extending
:mod:`albumentations` package.`
"""

from __future__ import annotations

from typing import Any

# noinspection PyPackageRequirements,PyUnresolvedReferences
import albumentations
import numpy as np
# noinspection PyPackageRequirements,PyUnresolvedReferences
from albumentations import *


# region Crop

class CropPatch(DualTransform):
    """Crop a patch of the image according to
    `<https://github.com/ZhendongWang6/Uformer/blob/main/dataset/dataset_denoise.py>__`
    """
    
    def __init__(
        self,
        patch_size  : int   = 128,
        always_apply: bool  = False,
        p           : float = 0.5,
    ):
        super().__init__(
            always_apply = always_apply,
            p            = p,
        )
        self.patch_size = patch_size
        self.r = 0
        self.c = 0
    
    def apply(self, img: np.ndarray, r: int, c: int, **params) -> np.ndarray:
        return img[r:r + self.patch_size, c:c + self.patch_size, :]
        
    def apply_to_mask(self, img: np.ndarray, r: int, c: int, **params) -> np.ndarray:
        return img[r:r + self.patch_size, c:c + self.patch_size, :]
    
    @property
    def targets_as_params(self) -> list[str]:
        return ["image"]
    
    def get_params_dependent_on_targets(self, params: dict[str, Any]) -> dict[str, Any]:
        image   = params["image"]
        h, w, c = image.shape
        if h - self.patch_size == 0:
            r = 0
            c = 0
        else:
            r = np.random.randint(0, h - self.patch_size)
            c = np.random.randint(0, w - self.patch_size)
        return {"r": r, "c": c}

# endregion

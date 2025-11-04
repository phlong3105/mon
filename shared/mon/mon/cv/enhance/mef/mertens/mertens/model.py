#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements Mertens et. al Exposure Fusion method.

References:
    - Paper: "Exposure Fusion," PG 2007.
    - Code: https://github.com/Jamy-L/Pytorch-Exposure-Fusion
"""

__all__ = [
    "Mertens",
    "mertens",
    "mertens_cv2",
]

from typing import Sequence

import box
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from mon.constants import MODELS
from mon.core import MLType, ModelMixin, nn, Path, Task, image as I
from .pyramid import (
    collapse_pyramid,
    compute_gaussian_pyramid,
    compute_laplacian_pyramid,
    merge_laplacian_pyramid,
)

current_file = Path(__file__).absolute()
root_dir     = current_file.parents[1]


# ----- Functional -----
def mertens(
    images  : torch.Tensor | list[torch.Tensor],
    w_sat   : float = 1,
    w_cont  : float = 1,
    w_exp   : float = 1,
    n_levels: int   = 4
) -> torch.Tensor:
    """Mertens et. al exposure fusion algorithm from Mertens et al.
    Combines a burst of images with different exposures into a single image with
    compressed dynamic range.
    
    Args:
        images: Input sequence of shape ``(N, C, H, W)``, where ``N`` is the
            number of images in the exposure range. Or a list of ``N`` images
            of shape ``(C, H, W)``.
        w_sat: The saturation importance weight. Default: ``1``.
        w_cont: The contrast importance weight. Default: ``1``.
        w_exp: The well-exposed importance weight. Default: ``1``.
        n_levels: The number of levels in the pyramids. Default: ``4``.
    
    Returns:
        The fused image, of shape ``(1, C, H, W)``.
    """
    if isinstance(images, Sequence):
        images = torch.stack(images, dim=0)
        
    gray_images = torch.mean(images, dim=1, keepdim=True)
    cont        = compute_contrast(gray_images)
    sat         = compute_saturation(images, gray_images)
    exp         = compute_well_exposedness(images)
    
    weights = (cont ** w_cont) * (sat ** w_sat) * (exp ** w_exp)
    # Normalize weights
    weights = weights / weights.sum(dim=0, keepdim=True)
    # Normalization will give Nan if all frames have 0 weight at 1 pixel.
    # In this case, all of them get the same weight
    weights =  weights.nan_to_num(nan =1 / images.size(0))

    # Get gaussian pyramid for weights and images
    img_gaussian_pyramid    = compute_gaussian_pyramid(images, n_levels=n_levels)
    img_laplacian_pyramid   = compute_laplacian_pyramid(img_gaussian_pyramid)
    weight_gaussian_pyramid = compute_gaussian_pyramid(weights, n_levels=n_levels)
    fused_laplacian_pyramid = merge_laplacian_pyramid(img_laplacian_pyramid, weight_gaussian_pyramid)
    fused_image             = collapse_pyramid(fused_laplacian_pyramid)
    
    return fused_image


def mertens_cv2(
    images  : list[np.ndarray] | list[torch.Tensor] | torch.Tensor,
    w_sat   : float = 1,
    w_cont  : float = 1,
    w_exp   : float = 1,
    n_levels: int   = 4
) -> np.ndarray:
    if isinstance(images, torch.Tensor):
        images = torch.split(images, 1, dim=0)
    if isinstance(images, Sequence) and all(isinstance(img, torch.Tensor) for img in images):
        images = [I.to_array(img) for img in images]
    
    align_mtb = cv2.createAlignMTB()
    align_mtb.process(images, images)
    merge_mertens = cv2.createMergeMertens()
    merge_mertens.setContrastWeight(w_cont)
    merge_mertens.setSaturationWeight(w_sat)
    merge_mertens.setExposureWeight(w_exp)
    exposure_fusion = merge_mertens.process(images)
    exposure_fusion = np.clip(exposure_fusion * 255, 0, 255).astype(np.uint8)
    return exposure_fusion


def compute_contrast(gray_images: torch.Tensor) -> torch.Tensor:
    k_laplacian = torch.tensor(
        data = [[0,  1, 0],
                [1, -4, 1],
                [0,  1, 0]],
        device = gray_images.device,
        dtype  = gray_images.dtype).unsqueeze(0).unsqueeze(0)
    contrast = torch.abs(F.conv2d(gray_images, k_laplacian, padding="same"))
    return contrast


def compute_saturation(images: torch.Tensor, gray_images: torch.Tensor) -> torch.Tensor:
    sat = torch.sqrt(torch.mean((images - gray_images) ** 2, dim=1, keepdim=True))
    return sat


def compute_well_exposedness(images: torch.Tensor) -> torch.Tensor:
    sigma = 0.2
    well_exposedness = torch.exp(-torch.sum((images - 0.5) ** 2, dim=1, keepdim=True) / (2 * sigma))
    return well_exposedness


# ----- Model -----
@MODELS.register(name="mertens", arch="mertens")
class Mertens(nn.Module, ModelMixin):
    """Mertens et. al Exposure Fusion method.
    
    References:
        - Paper: "Exposure Fusion," PG 2007.
        - Code: https://github.com/Jamy-L/Pytorch-Exposure-Fusion
    """

    arch     : str          = "mertens"
    name     : str          = "mertens"
    tasks    : list[Task]   = [Task.MEF]
    mltypes  : list[MLType] = [MLType.TRADITIONAL]
    model_dir: Path         = root_dir
    zoo      : dict         = box.Box()
    
    def __init__(
        self,
        w_sat   : float = 1,
        w_cont  : float = 1,
        w_exp   : float = 1,
        n_levels: int   = 4,
        backend : str   = "cv2"
    ):
        super().__init__()
        self.w_sat    = w_sat
        self.w_cont   = w_cont
        self.w_exp    = w_exp
        self.n_levels = n_levels
        self.backend  = backend
    
    def forward(self, images: list[np.ndarray] | list[torch.Tensor] | torch.Tensor) -> torch.Tensor:
        if self.backend == "cv2":
            f = mertens_cv2
        else:
            if isinstance(images, torch.Tensor) or all(isinstance(img, torch.Tensor) for img in images):
                f = mertens
            elif all(isinstance(img, np.ndarray) for img in images):
                f = mertens_cv2
            else:
                raise TypeError(f"[images] must be a torch.Tensor or np.ndarray, got {type(images).__name__}.")

        return f(
            images   = images,
            w_sat    = self.w_sat,
            w_cont   = self.w_cont,
            w_exp    = self.w_exp,
            n_levels = self.n_levels,
        )

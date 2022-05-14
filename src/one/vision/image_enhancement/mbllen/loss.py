#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Loss function proposed in the paper "MBLLEN: Low-light Image/Video
Enhancement Using CNNs". It consists of: Structure loss, Region loss, and
Context loss.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F

from one.nn import LOSSES
from one.vision.image_classification import VGG19

__all__ = [
    "MBLLENLoss"
]


def ssim(
    input      : Tensor,
    target     : Tensor,
    cs_map     : bool  = False,
    mean_metric: bool  = True,
    depth      : int   = 1,
    size       : int   = 11,
    sigma      : float = 1.5,
) -> Optional[Tensor]:
    """Calculate the SSIM (Structural Similarity Index) score between 2
    4D-/3D- channel-first- images.
    """
    input  = input.to(torch.float64)
    target = target.to(torch.float64)

    # Window shape [size, size]
    window = _fspecial_gauss(size=size, sigma=sigma)
    window = window.to(input.get_device())
    
    # Depth of image (255 in case the image has a different scale)
    l      = depth
    c1     = (0.01 * l) ** 2
    c2     = (0.03 * l) ** 2

    mu1       = F.conv2d(input=input, weight=window, stride=1)
    mu2       = F.conv2d(input=target, weight=window, stride=1)
    mu1_sq    = mu1 * mu1
    mu2_sq    = mu2 * mu2
    mu1_mu2   = mu1 * mu2
    sigma1_sq = F.conv2d(input=input * input, weight=window, stride=1) - mu1_sq
    sigma2_sq = F.conv2d(input=target * target, weight=window, stride=1) - mu2_sq
    sigma12   = F.conv2d(input=input * target, weight=window, stride=1) - mu1_mu2

    if cs_map:
        score = (
            ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) /
            ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)),
            (2.0 * sigma12 + c2) / (sigma1_sq + sigma2_sq + c2)
        )
    else:
        score = (
            ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2))
            / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
        )

    if mean_metric:
        score = torch.mean(score)
    score = score.detach()
    return score


def mae(input: Tensor, target: Tensor) -> Tensor:
    """Calculate MAE (Absolute Error) score between 2 4D-/3D- channel-first-
    images.
    """
    input  = input.to(torch.float64)
    target = target.to(torch.float64)
    score  = torch.mean(torch.abs(input - target) ** 2)
    return score


# noinspection PyTypeChecker
def _fspecial_gauss(size: int, sigma: float) -> Tensor:
    """Function to mimic the `special` gaussian MATLAB function.

    Args:
        size (int):
            Size of gaussian's window. Default: `11`.
        sigma (float):
            Sigma value of gaussian's window. Default: `1.5`.
    """
    x_data, y_data = np.mgrid[-size // 2 + 1: size // 2 + 1,
                              -size // 2 + 1: size // 2 + 1]
    x_data = np.expand_dims(x_data, axis=0)
    x_data = np.expand_dims(x_data, axis=0)
    y_data = np.expand_dims(y_data, axis=0)
    y_data = np.expand_dims(y_data, axis=0)
    x      = torch.from_numpy(x_data)
    y      = torch.from_numpy(y_data)
    x      = x.type(torch.float64)
    y      = y.type(torch.float64)
    z      = -((x ** 2 + y ** 2) / (2.0 * sigma ** 2))
    g      = torch.exp(z)
    return g / torch.sum(g)


def _range_scale(input: Tensor) -> Tensor:
    return input * 2.0 - 1.0


# MARK: - MBLLENLoss

# noinspection PyMethodMayBeStatic
@LOSSES.register(name="mbllen_loss")
class MBLLENLoss(nn.Module):
    """Implementation of loss function defined in the paper "MBLLEN: Low-light
    Image/Video Enhancement Using CNNs".
    """
    
    # MARK: Magic Functions
    
    def __init__(self):
        super().__init__()
        self.name = "mbllen_loss"
        self.vgg  = VGG19(out_indexes=26, pretrained=True)
        self.vgg.freeze()
        
    # MARK: Forward Pass

    # noinspection PyMethodMayBeStatic
    def forward(self, input: Tensor, target: Tensor, **_) -> Tensor:
        """Mbllen_loss = (structure_loss + (context_loss / 3.0) + 3 +
        region_loss)

        Args:
            input (Tensor):
                Enhanced images.
            target (Tensor):
                Normal-light images.
        
        Returns:
            loss (Tensor):
                Loss.
        """
        loss = (
            self.structure_loss(input, target) +
            self.context_loss(input, target) / 3.0 +
            3 + self.region_loss(input, target)
		)
        return loss
    
    def context_loss(self, input: Tensor, target: Tensor) -> Tensor:
        b, c, h, w     = [int(x) for x in target.shape]
        y_hat_scale    = _range_scale(input)
        y_hat_features = self.vgg.forward_features(y_hat_scale)
        y_hat_features = torch.reshape(y_hat_features, shape=(-1, 16, h, w))
    
        y_scale    = _range_scale(target)
        y_features = self.vgg.forward_features(y_scale)
        y_features = torch.reshape(y_features, shape=(-1, 16, h, w))
    
        loss = torch.mean(
            torch.abs(y_hat_features[:, :16, :, :] - y_features[:, :16, :, :])
        )
        return loss
    
    def region_loss(
        self, input: Tensor, target: Tensor, dark_pixel_percent: float = 0.4
    ) -> Tensor:
        """Implementation of region loss function defined in the paper
        "MBLLEN: Low-light Image/Video Enhancement Using CNNs".
        
        Args:
            input (Tensor):
                Enhanced images.
            target (Tensor):
                Normal-light images.
            dark_pixel_percent (float):
                Default: `0.4`.
                
        Returns:
            loss (Tensor):
                Region loss.
        """
        index     = int(256 * 256 * dark_pixel_percent - 1)
        gray1     = (0.39 * input[:, 0, :, :] + 0.5 * input[:, 1, :, :] +
                     0.11 * input[:, 2, :, :])
        gray      = torch.reshape(gray1, [-1, 256 * 256])
        gray_sort = torch.topk(-gray, k=256 * 256)[0]
        yu        = gray_sort[:, index]
        yu        = torch.unsqueeze(input=torch.unsqueeze(input=yu, dim=-1),
                                    dim=-1)
        mask      = (gray1 <= yu).type(torch.float64)
        mask1     = torch.unsqueeze(input=mask, dim=1)
        mask      = torch.cat(tensors=[mask1, mask1, mask1], dim=1)
    
        low_fake_clean  = torch.mul(mask, input[:, :3, :, :])
        high_fake_clean = torch.mul(1 - mask, input[:, :3, :, :])
        low_clean       = torch.mul(mask, target[:, : , :, :])
        high_clean      = torch.mul(1 - mask, target[:, : , :, :])
        loss            = torch.mean(torch.abs(low_fake_clean - low_clean) * 4 +
                                     torch.abs(high_fake_clean - high_clean))
        
        return loss
    
    def structure_loss(self, input: Tensor, target: Tensor) -> Tensor:
        """Implementation of structure loss function defined in the paper
        "MBLLEN: Low-light Image/Video Enhancement Using CNNs".
        
        Args:
            input (Tensor):
                Enhanced images.
            target (Tensor):
                Normal-light images.
    
        Returns:
            loss (Tensor):
                Fstructure loss.
        """
        mae_loss  = mae(input[:, :3, :, :], target)
        ssim_loss = (
            ssim(torch.unsqueeze(input[:, 0, :, :], dim=1),
                 torch.unsqueeze(target[:, 0, :, :], dim=1))
            + ssim(torch.unsqueeze(input[:, 1, :, :], dim=1),
                   torch.unsqueeze(target[:, 1, :, :], dim=1))
            + ssim(torch.unsqueeze(input[:, 2, :, :], dim=1),
                   torch.unsqueeze(target[:, 2, :, :], dim=1))
        )
        loss = mae_loss - ssim_loss
        return loss

#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

__all__ = [
    "L_exp",
    "L_TV",
    "ConfidenceGatedDepthLoss",
]

import torch
from torch import nn, Tensor


class L_exp(nn.Module):

    def __init__(self, patch_size: int, mean_val: float):
        super().__init__()
        self.pool = nn.AvgPool2d(patch_size)
        self.mean_val = mean_val

    def forward(self, x: Tensor) -> Tensor:
        mean = self.pool(x) ** 0.5
        d = torch.abs(
            torch.mean(
                torch.pow(mean - torch.FloatTensor([self.mean_val]).to(x.device), 2)
            )
        )
        return d


class L_TV(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = (x.size()[2] - 1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return 2 * (h_tv / count_h + w_tv / count_w) / batch_size


class ConfidenceGatedDepthLoss(nn.Module):
    """Loss function for depth-guided illumination smoothness.

    Encourage smoothness in the illumination map while preserving depth
    discontinuities using a confidence-gated mechanism.
    """

    # --- Callable & Context Manager ---
    def forward(self, image_v: Tensor, depth: Tensor, illu: Tensor) -> Tensor:
        """Calculate the loss between the input image and depth map.

        Args:
            image_v (Tensor): Input image Tensor of shape (B, 3, H, W) and
                values in the range [0.0, 1.0].
            depth (Tensor): Depth map tensor of shape (B, 1, H, W) and values
                in the range [0.0, 1.0].
            illu (Tensor): Illumination map tensor of shape (B, 1, H, W) and
                values in the range [0.0, 1.0].

        Returns:
            Tensor: A scalar tensor representing the computed loss.
        """
        # 1. Compute gradients
        illu_h,  illu_w  = self.gradient(illu)
        depth_h, depth_w = self.gradient(depth)

        # 2. Compute depth weights (Standard Retinex Logic)
        # "If depth gradient is high, weight should be low (allow edges)"
        # "If depth gradient is low (flat), weight should be high (enforce smoothness)"
        weight_h = torch.exp(-torch.abs(depth_h))
        weight_w = torch.exp(-torch.abs(depth_w))

        # 3. Compute confidence mask
        # Logic: If image_v is close to 0, confidence is 0.
        # We use a steep sigmoid centered at intensity 0.05
        # We align mask dimensions with gradients (which are 1 pixel smaller)
        mask_h = torch.sigmoid((image_v[:, :, :-1, :] - 0.05) * 20)
        mask_w = torch.sigmoid((image_v[:, :, :, :-1] - 0.05) * 20)

        # 4. Gated loss
        # We only penalize non-smoothness if:
        # a) The depth map says it should be smooth (weight is high) AND
        # b) We trust the depth map (mask is high)
        loss_h = illu_h.abs() * weight_h * mask_h
        loss_w = illu_w.abs() * weight_w * mask_w

        loss = loss_h.mean() + loss_w.mean()
        return loss

    @staticmethod
    def gradient(x: Tensor) -> tuple[Tensor, Tensor]:
        """Compute horizontal and vertical gradients of the input tensor."""
        h_grad = x[:, :, :-1, :] - x[:, :, 1:, :]
        w_grad = x[:, :, :, :-1] - x[:, :, :, 1:]
        return h_grad, w_grad

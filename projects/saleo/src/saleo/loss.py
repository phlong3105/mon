#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Loss functions.

This module provides various loss functions.
"""

from __future__ import annotations

__all__ = [
    "ConfidenceGatedDepthLoss",
]

import torch

from mon.training import loss as L


# ==============================================================================
# region LOSS FUNCTIONS
# ==============================================================================

class ConfidenceGatedDepthLoss(L.BaseLoss):
    """Loss function for depth-guided illumination smoothness.

    Encourage smoothness in the illumination map while preserving depth
    discontinuities using a confidence-gated mechanism.
    """

    # --- Lifecycle & Initialization ---
    def __init__(self, reduction: str = "mean"):
        super().__init__(reduction=reduction)

    # --- Callable & Context Manager ---
    def forward(
        self,
        image_v: torch.Tensor,
        depth  : torch.Tensor,
        illu   : torch.Tensor,
    ) -> torch.Tensor:
        """Calculate the loss between the input image and depth map.

        Args:
            image_v: Input image value (V-channel), formatted as a torch.Tensor
                of shape (B, 1, H, W) and values ranging from 0.0 to 1.0.
            depth: Depth map, formatted as a torch.Tensor of shape (B, 1, H, W)
                and values ranging from 0.0 to 1.0.
            illu: Predicted illumination map, formatted as a torch.Tensor of
                shape (B, 1, H, W) and values ranging from 0.0 to 1.0.

        Returns:
            Loss value.
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
    def gradient(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute horizontal and vertical gradients of the input tensor."""
        h_grad = x[:, :, :-1, :] - x[:, :, 1:, :]
        w_grad = x[:, :, :, :-1] - x[:, :, :, 1:]
        return h_grad, w_grad

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion

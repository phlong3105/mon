#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Loss Functions.

This module provides custom loss functions for PairLIE.
"""

from __future__ import annotations

__all__ = [
    "L_C",
    "L_P",
    "L_R",
    "L_tv",
]

import torch
from torch import Tensor
from torch.nn import functional as F

from mon.nn import Loss


# ==============================================================================
# region LOSS FUNCTIONS
# ==============================================================================

class L_tv(Loss):

    # --- Callable & Context Manager ---
    def forward(self, illu: Tensor) -> Tensor:
        """Calculate the color constancy loss on the ``input``.

        Args:
            illu (Tensor): Illumination tensor of shape (B, 3, H, W) and values
                ranging from 0.0 to 1.0.

        Returns:
            Tensor: Loss value.
        """
        b, c, h, w = illu.shape
        gradient_h = (illu[:, :, 2:, :] - illu[:, :, :h - 2, :]).abs()
        gradient_w = (illu[:, :, :, 2:] - illu[:, :, :, :w - 2]).abs()
        loss_h = gradient_h
        loss_w = gradient_w
        loss = loss_h + loss_w
        return self.reduce(loss)


class L_C(Loss):

    # --- Callable & Context Manager ---
    def forward(self, R1: Tensor, R2: Tensor) -> Tensor:
        loss = F.mse_loss(R1, R2)
        return self.reduce(loss)


class L_R(Loss):

    # --- Lifecycle & Initialization ---
    def __init__(self, reduction: str = "mean"):
        """Initialize a new instance.

        Args:
            reduction (str, optional): Reduction method to apply to the loss.
                One of: ["mean", "sum", "none"]. Defaults to "mean".
        """
        super().__init__(reduction=reduction)
        self.L_tv = L_tv()

    # --- Callable & Context Manager ---
    def forward(self, L1: Tensor, R1: Tensor, im1: Tensor, X1: Tensor) -> Tensor:
        max_rgb1, _ = torch.max(im1, 1)
        max_rgb1 = max_rgb1.unsqueeze(1)
        loss1 = F.mse_loss(L1 * R1, X1) + F.mse_loss(R1, X1 / L1.detach())
        loss2 = F.mse_loss(L1, max_rgb1) + self.L_tv(L1)
        loss = loss1 + loss2
        return self.reduce(loss)


class L_P(Loss):

    # --- Callable & Context Manager ---
    def forward(self, im1: Tensor, X1: Tensor) -> Tensor:
        loss = F.mse_loss(im1, X1)
        return self.reduce(loss)

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion

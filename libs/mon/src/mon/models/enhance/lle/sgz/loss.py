#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Loss Functions.

This module provides custom loss functions for SGZ.
"""

from __future__ import annotations

__all__ = [
    "L1_exp",
    "L_col",
    "L_exp",
    "L_focal",
    "L_spa",
    "L_spa8",
    "L_tv",
]

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from mon.nn import Loss


# ==============================================================================
# region LOSS FUNCTIONS
# ==============================================================================

class L_col(Loss):
    """Loss function for color constancy.

    Encourage the enhanced image to maintain color constancy by minimizing
    the differences between the mean RGB channels.
    """

    # --- Callable & Context Manager ---
    def forward(self, input: Tensor) -> Tensor:
        """Calculate the color constancy loss on the ``input``.

        Args:
            input (Tensor): Input (predictions) tensor of shape (B, C, H, W)
                and values ranging from 0.0 to 1.0.

        Returns:
            Tensor: Loss value.
        """
        mean_rgb = torch.mean(input, [2, 3], keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        d_rg = torch.pow(mr - mg, 2)
        d_rb = torch.pow(mr - mb, 2)
        d_gb = torch.pow(mb - mg, 2)
        k = torch.pow(
            torch.pow(d_rg, 2) +
            torch.pow(d_rb, 2) +
            torch.pow(d_gb, 2),
            0.5
        )
        return k


class L_spa(Loss):
    """Loss function for spatial consistency.

    Encourage spatial consistency between the input and predicted images by
    minimizing the differences in gradients.
    """

    # --- Lifecycle & Initialization ---
    def __init__(self, reduction: str = "mean"):
        """Initialize a new instance.

        Args:
            reduction (str, optional): Reduction method to apply to the loss.
                One of: ["mean", "sum", "none"]. Defaults to "mean".
        """
        super().__init__(reduction=reduction)

        # Build conv kernels
        kernel_left = torch.FloatTensor( [[0,  0, 0], [-1, 1,  0], [0,  0, 0]]).unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor( [[0,  0, 0], [ 0, 1, -1], [0,  0, 0]]).unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor( [[0, -1, 0], [ 0, 1,  0], [0,  0, 0]]).unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor( [[0,  0, 0], [ 0, 1,  0], [0, -1, 0]]).unsqueeze(0).unsqueeze(0)

        # Convert to parameters
        self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)

        # Pooling layer
        self.pool = nn.AvgPool2d(4)

    # --- Callable & Context Manager ---
    def forward(self, input: Tensor, pred: Tensor) -> Tensor:
        """Calculate the loss between ``input`` and ``pred``.

        Args:
            input (Tensor): Input image tensor of shape (B, C, H, W) and values
                ranging from 0.0 to 1.0.
            pred (Tensor): Predicted image tensor of shape (B, C, H, W) and
                values ranging from 0.0 to 1.0.

        Returns:
            Tensor: Loss value.
        """
        input_mean = torch.mean(input, 1, keepdim=True)
        pred_mean = torch.mean(pred, 1, keepdim=True)

        input_pool = self.pool(input_mean)
        pred_pool = self.pool(pred_mean)

        D_input_left = F.conv2d(input_pool, self.weight_left, padding=1)
        D_input_right = F.conv2d(input_pool, self.weight_right, padding=1)
        D_input_up = F.conv2d(input_pool, self.weight_up, padding=1)
        D_input_down = F.conv2d(input_pool, self.weight_down, padding=1)

        D_enhanced_left = F.conv2d(pred_pool, self.weight_left, padding=1)
        D_enhanced_right = F.conv2d(pred_pool, self.weight_right, padding=1)
        D_enhanced_up = F.conv2d(pred_pool, self.weight_up, padding=1)
        D_enhanced_down = F.conv2d(pred_pool, self.weight_down, padding=1)

        D_left = torch.pow(D_input_left - D_enhanced_left, 2)
        D_right = torch.pow(D_input_right - D_enhanced_right, 2)
        D_up = torch.pow(D_input_up - D_enhanced_up, 2)
        D_down = torch.pow(D_input_down - D_enhanced_down, 2)
        E = (D_left + D_right + D_up + D_down)
        # E = 25 * (D_left + D_right + D_up + D_down)

        return E


class L_spa8(Loss):
    """Loss function for spatial consistency.

    Encourage spatial consistency between the input and predicted images by
    minimizing the differences in gradients.
    """

    # --- Lifecycle & Initialization ---
    def __init__(self, reduction: str = "mean"):
        """Initialize a new instance.

        Args:
            reduction (str, optional): Reduction method to apply to the loss.
                One of: ["mean", "sum", "none"]. Defaults to "mean".
        """
        super().__init__(reduction=reduction)

        # Build conv kernels
        kernel_left = torch.FloatTensor( [[0,0,0],[-1,1,0],[0,0,0]]).unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor( [[0,0,0],[0,1,-1],[0,0,0]]).unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor( [[0,-1,0],[0,1, 0 ],[0,0,0]]).unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor( [[0,0,0],[0,1, 0],[0,-1,0]]).unsqueeze(0).unsqueeze(0)
        kernel_upleft = torch.FloatTensor( [[-1,0,0],[0,1,0],[0,0,0]]).unsqueeze(0).unsqueeze(0)
        kernel_upright = torch.FloatTensor( [[0,0,-1],[0,1,0],[0,0,0]]).unsqueeze(0).unsqueeze(0)
        kernel_loleft = torch.FloatTensor( [[0,0,0],[0,1,0],[-1,0,0]]).unsqueeze(0).unsqueeze(0)
        kernel_loright = torch.FloatTensor( [[0,0,0],[0,1,0],[0,0,-1]]).unsqueeze(0).unsqueeze(0)

        # Convert to parameters
        self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)
        self.weight_upleft = nn.Parameter(data=kernel_upleft, requires_grad=False)
        self.weight_upright = nn.Parameter(data=kernel_upright, requires_grad=False)
        self.weight_loleft = nn.Parameter(data=kernel_loleft, requires_grad=False)
        self.weight_loright = nn.Parameter(data=kernel_loright, requires_grad=False)

        # Pooling layer
        self.pool = nn.AvgPool2d(4)

    # --- Callable & Context Manager ---
    def forward(self, input: Tensor, pred: Tensor) -> Tensor:
        """Calculate the loss between ``input`` and ``pred``.

        Args:
            input (Tensor): Input image tensor of shape (B, C, H, W) and values
                ranging from 0.0 to 1.0.
            pred (Tensor): Predicted image tensor of shape (B, C, H, W) and
                values ranging from 0.0 to 1.0.

        Returns:
            Tensor: Loss value.
        """
        input_mean = torch.mean(input, 1, keepdim=True)
        pred_mean = torch.mean(pred, 1, keepdim=True)

        input_pool = self.pool(input_mean)
        pred_pool = self.pool(pred_mean)

        D_input_left = F.conv2d(input_pool , self.weight_left, padding=1)
        D_input_right = F.conv2d(input_pool , self.weight_right, padding=1)
        D_input_up = F.conv2d(input_pool , self.weight_up, padding=1)
        D_input_down = F.conv2d(input_pool , self.weight_down, padding=1)
        D_input_upleft = F.conv2d(input_pool , self.weight_upleft , padding=1)
        D_input_upright = F.conv2d(input_pool , self.weight_upright, padding=1)
        D_input_loleft = F.conv2d(input_pool , self.weight_loleft, padding=1)
        D_input_loright = F.conv2d(input_pool , self.weight_loright, padding=1)

        D_enhanced_left = F.conv2d(pred_pool , self.weight_left, padding=1)
        D_enhanced_right = F.conv2d(pred_pool , self.weight_right, padding=1)
        D_enhanced_up = F.conv2d(pred_pool , self.weight_up, padding=1)
        D_enhanced_down = F.conv2d(pred_pool , self.weight_down, padding=1)
        D_enhanced_upleft = F.conv2d(pred_pool, self.weight_upleft, padding=1)
        D_enhanced_upright = F.conv2d(pred_pool, self.weight_upright, padding=1)
        D_enhanced_loleft = F.conv2d(pred_pool, self.weight_loleft, padding=1)
        D_enhanced_loright = F.conv2d(pred_pool, self.weight_loright, padding=1)

        D_left = torch.pow(D_input_left - D_enhanced_left, 2)
        D_right = torch.pow(D_input_right - D_enhanced_right, 2)
        D_up = torch.pow(D_input_up - D_enhanced_up, 2)
        D_down = torch.pow(D_input_down - D_enhanced_down, 2)
        D_upleft = torch.pow(D_input_upleft - D_enhanced_upleft, 2)
        D_upright = torch.pow(D_input_upright - D_enhanced_upright, 2)
        D_loleft = torch.pow(D_input_loleft - D_enhanced_loleft, 2)
        D_loright = torch.pow(D_input_loright - D_enhanced_loright, 2)

        E = (D_left + D_right + D_up +D_down) + 0.5 * (D_upleft + D_upright + D_loleft + D_loright)

        return E


class L_exp(Loss):
    """Loss function for exposure control.

    Encourage well-exposedness in the predicted image by minimizing the
    difference between local patch means and a target mean value.
    """

    # --- Lifecycle & Initialization ---
    def __init__(self, patch_size: int, E: float, reduction: str = "mean"):
        """Initialize a new instance.

        Args:
            patch_size (int): Size of the local patch to compute the mean.
            E (float): Target mean value for well-exposedness, typically
                around 0.6.
            reduction (str, optional): Reduction method to apply to the loss.
                One of: ["mean", "sum", "none"]. Defaults to "mean".
        """
        super().__init__(reduction=reduction)
        self.pool = nn.AvgPool2d(patch_size)
        self.E = E

    # --- Callable & Context Manager ---
    def forward(self, input: Tensor) -> Tensor:
        """Calculate the loss between the ``input`` and the target exposure.

        Args:
            input (Tensor): Predicted image tensor of shape (B, C, H, W) and
                values ranging from 0.0 to 1.0.

        Returns:
            Tensor: Loss value.
        """
        mean = self.pool(torch.mean(input, 1, keepdim=True))
        loss = torch.pow(mean - torch.FloatTensor([self.E]).to(input.device), 2)
        return self.reduce(loss)


class L1_exp(Loss):
    """Loss function for exposure control.

    Encourage well-exposedness in the predicted image by minimizing the
    difference between local patch means and a target mean value.
    """

    # --- Lifecycle & Initialization ---
    def __init__(self, patch_size: int, E: float, reduction: str = "mean"):
        """Initialize a new instance.

        Args:
            patch_size (int): Size of the local patch to compute the mean.
            E (float): Target mean value for well-exposedness, typically
                around 0.6.
            reduction (str, optional): Reduction method to apply to the loss.
                One of: ["mean", "sum", "none"]. Defaults to "mean".
        """
        super().__init__(reduction=reduction)
        self.pool = nn.AvgPool2d(patch_size)
        self.E = E
        self.l1_loss = nn.SmoothL1Loss()

    # --- Callable & Context Manager ---
    def forward(self, input: Tensor) -> Tensor:
        """Calculate the loss between the ``input`` and the target exposure.

        Args:
            input (Tensor): Predicted image tensor of shape (B, C, H, W) and
                values ranging from 0.0 to 1.0.

        Returns:
            Tensor: Loss value.
        """
        mean = self.pool(torch.mean(input, 1, keepdim=True))
        mean_t = torch.FloatTensor([mean]).to(input.device)
        loss = self.l1_loss(mean, mean_t)
        return self.reduce(loss)


class L_tv(Loss):
    """Loss function for reducing noise by encouraging piecewise smoothness.

    Encourage spatial smoothness in the enhanced image by penalizing large
    intensity variations between neighboring pixels.
    """

    # --- Callable & Context Manager ---
    def forward(self, input: Tensor) -> Tensor:
        """Calculate the loss for the ``input`` tensor.

        Args:
            input (Tensor): Predicted image tensor of shape (B, C, H, W) and
                values ranging from 0.0 to 1.0.

        Returns:
            Tensor: Loss value.
        """
        x = input
        b, c, h, w = x.shape
        count_h = (x.size()[2] - 1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w - 1]), 2).sum()
        return 2 * (h_tv / count_h + w_tv / count_w) / b


class L_focal(Loss):

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        gamma: float = 0,
        eps: float = 1e-7,
        size_average: bool = True,
        reduction: str = "mean"
    ):
        """Initialize a new instance.

        Args:
            gamma (float, optional): Focusing parameter that adjusts the rate
                at which easy examples are down-weighted. Defaults to 0.
            eps (float, optional): Small value to avoid division by zero.
                Defaults to 1e-7.
            size_average (bool, optional): Whether to average the loss over
                the batch. Defaults to True.
            reduction (str, optional): Reduction method to apply to the loss.
                One of: ["mean", "sum", "none"]. Defaults to "mean".
        """
        super().__init__(reduction=reduction)
        self.gamma = gamma
        self.eps = eps
        self.size_average = size_average

    # --- Callable & Context Manager ---
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        y = self._one_hot(target, input.size(1))
        probs = F.softmax(input, dim=1)
        probs = (probs * y).sum(1)  # dimension ???
        probs = probs.clamp(self.eps, 1.0 - self.eps)
        log_p = probs.log()
        loss = -(torch.pow((1 - probs), self.gamma)) * log_p
        return self.reduce(loss)

    def _one_hot(self, x: Tensor, num_classes: int) -> Tensor:
        """Convert a tensor of indices to one-hot vectors."""
        size = x.size()[:1] + (num_classes,) + x.size()[1:]
        view = x.size()[:1] + (1,) + x.size()[1:]
        mask = Tensor(size).fill_(0).to(x)
        index = x.view(view)
        return mask.scatter_(1, index, 1.0)

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion

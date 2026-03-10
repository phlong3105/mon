#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Losses.

This module provides various loss functions for training IZ-DCE models.
"""

from __future__ import annotations

__all__ = [
    "L_col",
    "L_col_pre",
    "L_exp",
    "L_exp_asym",
    "L_spa",
    "L_tv",
    "L_tv_image",
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
        loss = torch.pow(
            torch.pow(d_rg, 2) +
            torch.pow(d_rb, 2) +
            torch.pow(d_gb, 2),
            0.5
        )
        return self.reduce(loss)


class L_col_pre(Loss):
    """Loss function for color constancy.

    Forces the enhanced pixels to maintain the same RGB ratio (hue) as the
    original low-light pixels using Cosine Distance.
    """

    # --- Callable & Context Manager ---
    def forward(self, input: Tensor, pred: Tensor) -> Tensor:
        """Calculate the color constancy loss on the ``input``.

        Args:
            input (Tensor): Input image tensor of shape (B, C, H, W) and values
                ranging from 0.0 to 1.0.
            pred (Tensor): Predicted image tensor of shape (B, C, H, W) and
                values ranging from 0.0 to 1.0.

        Returns:
            Tensor: Loss value.
        """
        # Calculate cosine similarity along the RGB channel dimension (dim=2)
        # Adding eps prevents division by zero for completely black pixels
        eps = 1e-6
        cos_sim = F.cosine_similarity(pred + eps, input + eps, dim=2)

        # We want the similarity to be exactly 1.0 (perfectly aligned vectors)
        # So the loss is the difference from 1.0
        return self.reduce(1.0 - cos_sim)


class L_spa(Loss):
    """Loss function for spatial consistency.

    Encourage spatial consistency between the input and predicted images by
    minimizing the differences in gradients. Applies a dynamic depth weight
    to strictly preserve foreground structures while allowing flexible
    gradient changes in distant background regions.
    """

    # --- Lifecycle & Initialization ---
    def __init__(self, eps: float = 0.1, reduction: str = "mean"):
        """Initialize a new instance.

        Args:
            eps (float, optional): Epsilon value to ensure background regions
                still receive a minimal structural penalty. Defaults to 0.1.
            reduction (str, optional): Reduction method to apply to the loss.
                One of: ["mean", "sum", "none"]. Defaults to "mean".
        """
        super().__init__(reduction=reduction)
        self.eps = eps

        kernel_left = torch.FloatTensor( [[0,  0, 0], [-1, 1,  0], [0,  0, 0]]).unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor( [[0,  0, 0], [ 0, 1, -1], [0,  0, 0]]).unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor( [[0, -1, 0], [ 0, 1,  0], [0,  0, 0]]).unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor( [[0,  0, 0], [ 0, 1,  0], [0, -1, 0]]).unsqueeze(0).unsqueeze(0)

        self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)
        self.pool = nn.AvgPool2d(4)

    # --- Callable & Context Manager ---
    def forward(self, input: Tensor, pred: Tensor, depth: Tensor | None = None) -> Tensor:
        """Calculate the loss between ``input`` and ``pred``.

        Args:
            input (Tensor): Input image tensor of shape (B, C, H, W) and values
                ranging from 0.0 to 1.0.
            pred (Tensor): Predicted image tensor of shape (B, C, H, W) and
                values ranging from 0.0 to 1.0.
            depth (Tensor, optional): Depth tensor of shape (B, 1, H, W) and values
                ranging from 0.0 (far) to 1.0 (near). Defaults to None.

        Returns:
            Tensor: Loss value.
        """
        input_mean = torch.mean(input, 1, keepdim=True)
        pred_mean = torch.mean(pred, 1, keepdim=True)

        # 1. Pool the images and the depth map to match spatial dimensions
        input_pool = self.pool(input_mean)
        pred_pool = self.pool(pred_mean)
        if depth is not None:
            depth_pool = self.pool(depth)  # [B, 1, H/4, W/4]
        else:
            depth_pool = torch.ones_like(input_pool)

        # 2. Create the depth weight mask
        # Foreground (~1.0) gets high penalty, Background (~0.0) gets minimal
        # penalty (eps)
        weight = depth_pool + self.eps

        # 3. Extract directional gradients via convolution
        D_input_left = F.conv2d(input_pool, self.weight_left, padding=1)
        D_input_right = F.conv2d(input_pool, self.weight_right, padding=1)
        D_input_up = F.conv2d(input_pool, self.weight_up, padding=1)
        D_input_down = F.conv2d(input_pool, self.weight_down, padding=1)

        D_enhanced_left = F.conv2d(pred_pool, self.weight_left, padding=1)
        D_enhanced_right = F.conv2d(pred_pool, self.weight_right, padding=1)
        D_enhanced_up = F.conv2d(pred_pool, self.weight_up, padding=1)
        D_enhanced_down = F.conv2d(pred_pool, self.weight_down, padding=1)

        # 4. Apply the depth weight to the squared differences
        D_left = torch.pow(D_input_left - D_enhanced_left, 2) * weight
        D_right = torch.pow(D_input_right - D_enhanced_right, 2) * weight
        D_up = torch.pow(D_input_up - D_enhanced_up, 2) * weight
        D_down = torch.pow(D_input_down - D_enhanced_down, 2) * weight

        E = (D_left + D_right + D_up + D_down)

        return self.reduce(E)


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
        loss = torch.mean(torch.pow(mean - torch.FloatTensor([self.E]).to(input.device), 2))
        return loss


class L_exp_asym(Loss):
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
        self.pool = nn.AvgPool2d(patch_size, stride=16)
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
        mean = self.pool(input)

        # Standard difference from the target exposure
        diff = mean - self.E

        # NEW: Asymmetric weighting
        # If the patch is darker than E (diff < 0), apply full weight (1.0) to pull it up.
        # If the patch is brighter than E (diff > 0), apply a lighter weight (0.5) so the
        # ODE solver doesn't panic and try to crush the highlights down, or push them too high.
        weight = torch.where(
            diff < 0,
            torch.tensor(1.0, device=diff.device),
            torch.tensor(0.5, device=diff.device)
        )

        loss = torch.mean(weight * torch.abs(diff))
        return loss


class L_tv(Loss):
    """Loss function for reducing noise by encouraging piecewise smoothness.

    Encourage spatial smoothness in the enhanced image or parameter maps by
    penalizing large intensity variations between neighboring pixels.
    Applies dynamic weighting based on monocular depth to preserve
    foreground details while aggressively smoothing background noise.
    """

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        eps: float = 0.1,
        weight: float = 1.0,
        reduction: str = "mean"
    ):
        """Initialize a new instance.

        Args:
            eps (float, optional): Epsilon value to ensure background regions
                still receive a minimal structural penalty. Defaults to 0.1.
            weight (float, optional): Overall weight for the TV loss.
                Defaults to 1.0.
            reduction (str, optional): Reduction method to apply to the loss.
                One of: ["mean", "sum", "none"]. Defaults to "mean".
        """
        super().__init__(reduction=reduction)
        self.eps = eps
        self.weight = weight

    # --- Callable & Context Manager ---
    def forward(self, input: Tensor, depth: Tensor | None = None) -> Tensor:
        """Calculate the loss for the ``input`` tensor.

        Args:
            input (Tensor): Predicted image tensor of shape (B, C, H, W) and
                values ranging from 0.0 to 1.0.
            depth (Tensor): Depth tensor of shape (B, 1, H, W) and values
                ranging from 0.0 (far) to 1.0 (near). Defaults to None.

        Returns:
            Tensor: Loss value.
        """
        x = input
        b, c, h, w = x.shape

        # Original code
        """
        count_h = (x.size()[2] - 1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w - 1]), 2).sum()
        return 2 * (h_tv / count_h + w_tv / count_w) / b
        """

        # 1. Calculate base squared differences between neighboring pixels
        diff_h = torch.pow((x[:, :, 1:, :] - x[:, :, :h - 1, :]), 2)
        diff_w = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w - 1]), 2)

        # 2. Calculate spatial weights from the depth map
        # Invert depth: 1.0 (far/background) -> maximum smoothing
        #               0.0 (near/foreground) -> minimal smoothing (self.eps)
        if depth is not None:
            weight = (1.0 - depth) + self.eps # [B, 1, H, W]
            # Align the weight masks with the shifted difference tensors
            weight_h = weight[:, :, :h - 1, :]
            weight_w = weight[:, :, :, :w - 1]
        else:
            weight_h = 1.0
            weight_w = 1.0

        # 3. Apply weights and sum
        h_tv = (diff_h * weight_h).sum()
        w_tv = (diff_w * weight_w).sum()

        # 4. Normalize by batch size and element count
        count_h = (h - 1) * w
        count_w = h * (w - 1)

        loss = 2 * (h_tv / count_h + w_tv / count_w) / b
        # return self.weight * self.reduce(loss)
        return self.weight * loss


class L_tv_image(Loss):
    """Loss function for direct image denoising.

    Applies Total Variation smoothing directly to the enhanced image pixels.
    Uses a joint mask of Depth and Original Illumination to aggressively
    denoise dark background regions while completely preserving well-lit
    areas and foreground textures.
    """

    # --- Lifecycle & Initialization ---
    def __init__(self, eps: float = 0.05, reduction: str = "mean"):
        """Initialize a new instance.

        Args:
            eps (float, optional): Epsilon value to ensure background regions
                still receive a minimal structural penalty. Defaults to 0.05.
            reduction (str, optional): Reduction method to apply to the loss.
                One of: ["mean", "sum", "none"]. Defaults to "mean".
        """
        super().__init__(reduction=reduction)
        self.eps = eps

    # --- Callable & Context Manager ---
    def forward(self, input: Tensor, pred: Tensor, depth: Tensor | None = None) -> Tensor:
        """Calculate the loss between ``input`` and ``pred``.

        Args:
            input (Tensor): Input image tensor of shape (B, C, H, W) and values
                ranging from 0.0 to 1.0.
            pred (Tensor): Predicted image tensor of shape (B, C, H, W) and
                values ranging from 0.0 to 1.0.
            depth (Tensor, optional): Depth tensor of shape (B, 1, H, W) and
                values ranging from 0.0 (far) to 1.0 (near). Defaults to None.

        Returns:
            Tensor: Loss value.
        """
        b, c, h, w = pred.shape

        # 1. Calculate image pixel differences
        diff_h = torch.pow((pred[:, :, 1:, :] - pred[:, :, :h - 1, :]), 2)
        diff_w = torch.pow((pred[:, :, :, 1:] - pred[:, :, :, :w - 1]), 2)

        # 2. Extract original illumination (intensity)
        # We detach it because we don't want to calculate gradients for the original image
        illumination = torch.mean(input, dim=1, keepdim=True).detach()

        # 3. Create the Joint Mask
        # High noise probability = originally dark (1.0 - illum) AND far away (1.0 - depth)
        depth = depth or 0.0
        noise_prob = (1.0 - illumination) * (1.0 - depth)

        # The weight is strictly bounded. If it's bright OR foreground, weight approaches self.eps
        weight = noise_prob + self.eps # [B, 1, H, W]

        weight_h = weight[:, :, :h - 1, :]
        weight_w = weight[:, :, :, :w - 1]

        # 4. Apply weights
        h_tv = (diff_h * weight_h).sum()
        w_tv = (diff_w * weight_w).sum()

        count_h = (h - 1) * w
        count_w = h * (w - 1)

        loss = 2 * (h_tv / count_h + w_tv / count_w) / b
        return self.reduce(loss)

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion

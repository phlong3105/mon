#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Loss functions for tasks where the predictions and targets are images.

This module provides various loss functions commonly used in training deep
learning models where the outputs and targets are images.
"""

__all__ = [
    "ColorConstancyLoss",
    "DepthAwareIlluminationLoss",
    "EdgeLoss",
    "ExposureControlLoss",
    "ExposureValueControlLoss",
    "PSNRLoss",
    "SpatialConsistencyLoss",
    "StructureTextureDecompositionLoss",
    "TotalVariationLoss",
]

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur

from .base import BaseLoss
from .basic import CharbonnierLoss


# ==============================================================================
# PIXEL & INTENSITY LOSSES
# ==============================================================================

class ExposureControlLoss(BaseLoss):
    """A loss function for managing the average luminance.
    
    Ensure the exposure level of the enhanced image by penalizing the deviation
    of the average intensity from a well-exposedness level.

    References:
        - https://github.com/Li-Chongyi/Zero-DCE/blob/master/Zero-DCE_code/Myloss.py#L74
    
    Attributes:
        channel_mean (bool): If True, compute the mean across channels before pooling.
        mean_val (nn.Parameter): Learnable parameter representing the well-exposedness level.
        pool (nn.AvgPool2d): Average pooling layer for patch-wise mean calculation.
    """
    
    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        patch_size   : int   = 16,
        mean_val     : float = 0.6,
        required_grad: bool  = True,
        channel_mean : bool  = True,
        reduction    : str   = "mean",
    ):
        """Initialize a new instance.
        
        Args:
            patch_size: Kernel size for pooling layer. Defaults to 16.
            mean_val: Well-exposedness level E; lower values produce, brighter
                images. Defaults to 0.6.
            required_grad: If True, ``mean_val`` is learnable. Defaults to True.
            channel_mean: If True, compute the mean across channels before pooling.
                Defaults to True.
            reduction: Reduction method to apply to the loss. Can be one of
                "none", "mean", or "sum". Defaults to "mean".
        """
        super().__init__(reduction=reduction)
        self.channel_mean = channel_mean
        self.mean_val     = nn.Parameter(torch.full([1], mean_val), requires_grad=required_grad)
        self.pool         = nn.AvgPool2d(patch_size)
    
    # --- Callable & Context Manager ---
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Calculate the loss between the input and the well-exposedness level.
        
        Args:
            input: Predicted image, formatted as a torch.Tensor of dimensions
                (B, C, H, W) and values ranging from 0.0 to 1.0.
        
        Returns:
            Loss value, formatted according to the specified reduction method.
        """
        x = input
        if self.channel_mean:
            x = torch.mean(input, 1, keepdim=True)
        mean = self.pool(x)
        loss = torch.pow(mean - self.mean_val, 2)
        loss = self.reduce(loss=loss)
        return loss


class ExposureValueControlLoss(BaseLoss):
    """A variation of ExposureControlLoss for non-linear exposure adjustment.
    
    References:
        - https://github.com/Li-Chongyi/Zero-DCE/blob/master/Zero-DCE_code/Myloss.py#L74
        
    Attributes:
        channel_mean (bool): If True, compute the mean across channels before pooling.
        mean_val (nn.Parameter): Learnable parameter representing the well-exposedness level.
        pool (nn.AvgPool2d): Average pooling layer for patch-wise mean calculation.
    """
    
    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        patch_size   : int   = 16,
        mean_val     : float = 0.6,
        required_grad: bool  = True,
        channel_mean : bool  = True,
        reduction    : str   = "mean",
    ):
        """Initialize a new instance.
        
        Args:
            patch_size: Kernel size for pooling layer. Defaults to 16.
            mean_val: Well-exposedness level E; lower values produce, brighter
                images. Defaults to 0.6.
            required_grad: If True, ``mean_val`` is learnable. Defaults to True.
            channel_mean: If True, compute the mean across channels before pooling.
                Defaults to True.
            reduction: Reduction method to apply to the loss. Can be one of
                "none", "mean", or "sum". Defaults to "mean".
        """
        super().__init__(reduction=reduction)
        self.channel_mean = channel_mean
        self.mean_val     = nn.Parameter(torch.full([1], mean_val), requires_grad=required_grad)
        self.pool         = nn.AvgPool2d(patch_size)
    
    # --- Callable & Context Manager ---
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Calculate the loss between the input and the well-exposedness level.
        
        Args:
            input: Predicted image, formatted as a torch.Tensor of dimensions
                (B, C, H, W) and values ranging from 0.0 to 1.0.
        
        Returns:
            Loss value, formatted according to the specified reduction method.
        """
        x = input
        if self.channel_mean:
            x = torch.mean(x, 1, keepdim=True)  # Channel-wise mean: [B, 1, H, W]
        mean = self.pool(x) ** 0.5              # Pooled mean:       [B, 1, H, W]
        loss = torch.pow((mean - self.mean_val), 2)
        loss = torch.abs(torch.mean(loss))
        return loss
    

# ==============================================================================
# COLOR & FIDELITY LOSSES
# ==============================================================================

class ColorConstancyLoss(BaseLoss):
    """A loss function for preventing color shifting by balancing RGB means.
    
    Ensure the color consistency of the enhanced image by penalizing the
    variance of the mean of R, G, and B channels.
    
    References:
        - https://github.com/Li-Chongyi/Zero-DCE/blob/master/Zero-DCE_code/Myloss.py#L9
    
    Attributes:
        eps (float): Small constant for numerical stability.
    """
    
    # --- Lifecycle & Initialization ---
    def __init__(self, eps: float = 1e-6, reduction: str = "mean"):
        """Initialize a new instance.
        
        Args:
            eps: Small constant for numerical stability. Defaults to 1e-6.
            reduction: Reduction method to apply to the loss. Can be one of
                "none", "mean", or "sum". Defaults to "mean".
        """
        super().__init__(reduction=reduction)
        self.eps = eps
    
    # --- Callable & Context Manager ---
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Calculate the loss for the input.
        
        Args:
            input: Predicted image, formatted as a torch.Tensor of dimensions
                (B, C, H, W) and values ranging from 0.0 to 1.0.
        
        Returns:
            Loss value, formatted according to the specified reduction method.
        """
        mean_rgb   = torch.mean(input, [2, 3], keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        d_rg       = torch.pow(torch.abs(mr - mg), 2)
        d_rb       = torch.pow(torch.abs(mr - mb), 2)
        d_gb       = torch.pow(torch.abs(mb - mg), 2)
        d_rg2      = torch.pow(d_rg, 2)
        d_rb2      = torch.pow(d_rb, 2)
        d_gb2      = torch.pow(d_gb, 2)
        loss       = d_rg2 + d_rb2 + d_gb2
        loss       = torch.pow(loss + self.eps, 0.5)
        loss       = self.reduce(loss=loss)
        return loss


class PSNRLoss(BaseLoss):
    """A loss function based on Peak Signal-to-Noise Ratio (PSNR).
    
    Measure the fidelity of the enhanced image compared to the ground truth image.
    
    Attributes:
        to_y (bool): If True, use Y-channel for computing PSNR.
        coef (torch.Tensor): Coefficients for RGB to Y-channel conversion.
        first (bool): Flag to indicate if the coefficients need to be moved to
            the input device.
    """
    
    # --- Lifecycle & Initialization ---
    def __init__(self, to_y: bool = False, reduction: str = "mean"):
        """Initialize a new instance.
        
        Args:
            to_y: IIf True, use Y-channel for computing PSNR.
            reduction: Reduction method to apply to the loss. Can be one of
                "none", "mean", or "sum". Defaults to "mean".
        """
        super().__init__(reduction=reduction)
        self.to_y  = to_y
        self.coef  = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    # --- Callable & Context Manager ---
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate the loss between input and target.
        
        Args:
            input: Predicted image, formatted as a torch.Tensor of dimensions
                (B, C, H, W) and values ranging from 0.0 to 1.0.
            target: Target image, formatted as a torch.Tensor of dimensions
                (B, C, H, W) and values ranging from 0.0 to 1.0.
        
        Returns:
            Loss value, formatted according to the specified reduction method.
        """
        if self.to_y:
            if self.first:
                self.coef  = self.coef.to(input.device)
                self.first = False
            input  = (input  * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.0
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.0
            input  = input  / 255.0
            target = target / 255.0
            pass
        
        diff = input - target
        rmse = ((diff ** 2).mean(dim=(1, 2, 3)) + 1e-8).sqrt()
        loss = 20 * torch.log10(1 / rmse).mean()
        loss = (50.0 - loss) / 100.0
        loss = self.reduce(loss=loss)
        return loss
    

# ==============================================================================
# SPATIAL & STRUCTURAL LOSSES
# ==============================================================================

class SpatialConsistencyLoss(BaseLoss):
    """A loss function for maintaining local gradients (crucial for Zero-DCE
    architectures).
    
    Ensure spatial coherence in the enhanced image by penalizing discrepancies
    in local gradients between the enhanced and input images.
    
    Attributes:
        num_regions (int): Number of directional regions to consider for gradient comparison.
        weight_left (torch.nn.Parameter): Convolution kernel for left gradient.
        weight_right (torch.nn.Parameter): Convolution kernel for right gradient.
        weight_up (torch.nn.Parameter): Convolution kernel for up gradient.
        weight_down (torch.nn.Parameter): Convolution kernel for down gradient.
        weight_upleft (torch.nn.Parameter): Convolution kernel for up-left gradient.
        weight_upright (torch.nn.Parameter): Convolution kernel for up-right gradient.
        weight_downleft (torch.nn.Parameter): Convolution kernel for down-left gradient.
        weight_downright (torch.nn.Parameter): Convolution kernel for down-right gradient.
        weight_left2 (torch.nn.Parameter): Convolution kernel for left gradient (2-pixel).
        weight_right2 (torch.nn.Parameter): Convolution kernel for right gradient (2-pixel).
        weight_up2 (torch.nn.Parameter): Convolution kernel for up gradient (2-pixel).
        weight_down2 (torch.nn.Parameter): Convolution kernel for down gradient (2-pixel).
        weight_up2left2 (torch.nn.Parameter): Convolution kernel for up-left gradient (2-pixel).
        weight_up2right2 (torch.nn.Parameter): Convolution kernel for up-right gradient (2-pixel).
        weight_down2left2 (torch.nn.Parameter): Convolution kernel for down-left gradient (2-pixel).
        weight_down2right2 (torch.nn.Parameter): Convolution kernel for down-right gradient (2-pixel).
        weight_up2left1 (torch.nn.Parameter): Convolution kernel for up-left gradient (mixed).
        weight_up2right1 (torch.nn.Parameter): Convolution kernel for up-right gradient (mixed).
        weight_up1left2 (torch.nn.Parameter): Convolution kernel for up-left gradient (mixed).
        weight_up1right2 (torch.nn.Parameter): Convolution kernel for up-right gradient (mixed).
        weight_down2left1 (torch.nn.Parameter): Convolution kernel for down-left gradient (mixed).
        weight_down2right1 (torch.nn.Parameter): Convolution kernel for down-right gradient (mixed).
    """
    
    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        num_regions: Literal[4, 8, 16, 24] = 4,
        patch_size : int = 4,
        reduction  : str = "mean",
    ):
        """Initialize a new instance.
        
        Args:
            num_regions: Number of directional regions to consider for gradient
                comparison. Can be one of 4, 8, 16, or 24. Defaults to 4.
            patch_size: Size of the Gaussian kernel for blurring. Defaults to 4.
            reduction: Reduction method to apply to the loss. Can be one of
                "none", "mean", or "sum". Defaults to "mean".
        """
        super().__init__(reduction=reduction)
        self.num_regions = num_regions
        
        kernel_left = torch.FloatTensor([
            [ 0,  0, 0],
            [-1,  1, 0],
            [ 0,  0, 0]
        ]).unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor([
            [0,  0,  0],
            [0,  1, -1],
            [0,  0,  0]
        ]).unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor([
            [0, -1, 0],
            [0,  1, 0],
            [0,  0, 0]
        ]).unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor([
            [0,  0, 0],
            [0,  1, 0],
            [0, -1, 0]
        ]).unsqueeze(0).unsqueeze(0)
        if self.num_regions in [8, 16]:
            kernel_upleft = torch.FloatTensor([
                [-1, 0, 0],
                [ 0, 1, 0],
                [ 0, 0, 0]
            ]).unsqueeze(0).unsqueeze(0)
            kernel_upright = torch.FloatTensor([
                [0, 0, -1],
                [0, 1,  0],
                [0, 0,  0]
            ]).unsqueeze(0).unsqueeze(0)
            kernel_downleft = torch.FloatTensor([
                [ 0, 0, 0],
                [ 0, 1, 0],
                [-1, 0, 0]
            ]).unsqueeze(0).unsqueeze(0)
            kernel_downright = torch.FloatTensor([
                [0, 0,  0],
                [0, 1,  0],
                [0, 0, -1]
            ]).unsqueeze(0).unsqueeze(0)
        if self.num_regions in [16, 24]:
            kernel_left2 = torch.FloatTensor([
                [0,  0,  0, 0, 0],
                [0,  0,  0, 0, 0],
                [-1, 0,  1, 0, 0],
                [0,  0,  0, 0, 0],
                [0,  0,  0, 0, 0]
            ]).unsqueeze(0).unsqueeze(0)
            kernel_right2 = torch.FloatTensor([
                [0, 0,  0, 0,  0],
                [0, 0,  0, 0,  0],
                [0, 0,  1, 0, -1],
                [0, 0,  0, 0,  0],
                [0, 0,  0, 0,  0]
            ]).unsqueeze(0).unsqueeze(0)
            kernel_up2 = torch.FloatTensor([
                [0, 0, -1, 0, 0],
                [0, 0,  0, 0, 0],
                [0, 0,  1, 0, 0],
                [0, 0,  0, 0, 0],
                [0, 0,  0, 0, 0]
            ]).unsqueeze(0).unsqueeze(0)
            kernel_down2 = torch.FloatTensor([
                [0, 0,  0, 0, 0],
                [0, 0,  0, 0, 0],
                [0, 0,  1, 0, 0],
                [0, 0,  0, 0, 0],
                [0, 0, -1, 0, 0]
            ]).unsqueeze(0).unsqueeze(0)
            kernel_up2left2 = torch.FloatTensor([
                [-1, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0],
                [ 0, 0, 1, 0, 0],
                [ 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0]
            ]).unsqueeze(0).unsqueeze(0)
            kernel_up2right2 = torch.FloatTensor([
                [0, 0, 0, 0, -1],
                [0, 0, 0, 0,  0],
                [0, 0, 1, 0,  0],
                [0, 0, 0, 0,  0],
                [0, 0, 0, 0,  0]
            ]).unsqueeze(0).unsqueeze(0)
            kernel_down2left2 = torch.FloatTensor([
                [ 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0],
                [ 0, 0, 1, 0, 0],
                [ 0, 0, 0, 0, 0],
                [-1, 0, 0, 0, 0]
            ]).unsqueeze(0).unsqueeze(0)
            kernel_down2right2 = torch.FloatTensor([
                [0, 0, 0, 0,  0],
                [0, 0, 0, 0,  0],
                [0, 0, 1, 0,  0],
                [0, 0, 0, 0,  0],
                [0, 0, 0, 0, -1]
            ]).unsqueeze(0).unsqueeze(0)
        if self.num_regions in [24]:
            kernel_up2left1 = torch.FloatTensor([
                [0, -1, 0, 0, 0],
                [0,  0, 0, 0, 0],
                [0,  0, 1, 0, 0],
                [0,  0, 0, 0, 0],
                [0,  0, 0, 0, 0]
            ]).unsqueeze(0).unsqueeze(0)
            kernel_up2right1 = torch.FloatTensor([
                [0, 0, 0, -1, 0],
                [0, 0, 0,  0, 0],
                [0, 0, 1,  0, 0],
                [0, 0, 0,  0, 0],
                [0, 0, 0,  0, 0]
            ]).unsqueeze(0).unsqueeze(0)
            kernel_up1left2 = torch.FloatTensor([
                [0,  0, 0, 0, 0],
                [-1, 0, 0, 0, 0],
                [0,  0, 1, 0, 0],
                [0,  0, 0, 0, 0],
                [0,  0, 0, 0, 0]
            ]).unsqueeze(0).unsqueeze(0)
            kernel_up1right2 = torch.FloatTensor([
                [0, 0, 0, 0,  0],
                [0, 0, 0, 0, -1],
                [0, 0, 1, 0,  0],
                [0, 0, 0, 0,  0],
                [0, 0, 0, 0,  0]
            ]).unsqueeze(0).unsqueeze(0)
            kernel_down2left1 = torch.FloatTensor([
                [0,  0, 0, 0, 0],
                [0,  0, 0, 0, 0],
                [0,  0, 1, 0, 0],
                [0,  0, 0, 0, 0],
                [0, -1, 0, 0, 0]
            ]).unsqueeze(0).unsqueeze(0)
            kernel_down2right1 = torch.FloatTensor([
                [0, 0, 0,  0, 0],
                [0, 0, 0,  0, 0],
                [0, 0, 1,  0, 0],
                [0, 0, 0,  0, 0],
                [0, 0, 0, -1, 0]
            ]).unsqueeze(0).unsqueeze(0)
            kernel_down1left2 = torch.FloatTensor([
                [ 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0],
                [ 0, 0, 1, 0, 0],
                [-1, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0]
            ]).unsqueeze(0).unsqueeze(0)
            kernel_down1right2 = torch.FloatTensor([
                [0, 0, 0, 0,  0],
                [0, 0, 0, 0,  0],
                [0, 0, 1, 0,  0],
                [0, 0, 0, 0, -1],
                [0, 0, 0, 0,  0]
            ]).unsqueeze(0).unsqueeze(0)
            
        self.weight_left  = nn.Parameter(data=kernel_left,  requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up    = nn.Parameter(data=kernel_up,    requires_grad=False)
        self.weight_down  = nn.Parameter(data=kernel_down,  requires_grad=False)
        if self.num_regions in [8, 16]:
            self.weight_upleft    = nn.Parameter(data=kernel_upleft,    requires_grad=False)
            self.weight_upright   = nn.Parameter(data=kernel_upright,   requires_grad=False)
            self.weight_downleft  = nn.Parameter(data=kernel_downleft,  requires_grad=False)
            self.weight_downright = nn.Parameter(data=kernel_downright, requires_grad=False)
        if self.num_regions in [16, 24]:
            self.weight_left2       = nn.Parameter(data=kernel_left2,       requires_grad=False)
            self.weight_right2      = nn.Parameter(data=kernel_right2,      requires_grad=False)
            self.weight_up2         = nn.Parameter(data=kernel_up2,         requires_grad=False)
            self.weight_down2       = nn.Parameter(data=kernel_down2,       requires_grad=False)
            self.weight_up2left2    = nn.Parameter(data=kernel_up2left2,    requires_grad=False)
            self.weight_up2right2   = nn.Parameter(data=kernel_up2right2,   requires_grad=False)
            self.weight_down2left2  = nn.Parameter(data=kernel_down2left2,  requires_grad=False)
            self.weight_down2right2 = nn.Parameter(data=kernel_down2right2, requires_grad=False)
        if self.num_regions in [24]:
            self.weight_up2left1    = nn.Parameter(data=kernel_up2left1,    requires_grad=False)
            self.weight_up2right1   = nn.Parameter(data=kernel_up2right1,   requires_grad=False)
            self.weight_up1left2    = nn.Parameter(data=kernel_up1left2,    requires_grad=False)
            self.weight_up1right2   = nn.Parameter(data=kernel_up1right2,   requires_grad=False)
            self.weight_down2left1  = nn.Parameter(data=kernel_down2left1,  requires_grad=False)
            self.weight_down2right1 = nn.Parameter(data=kernel_down2right1, requires_grad=False)
            self.weight_down1left2  = nn.Parameter(data=kernel_down1left2,  requires_grad=False)
            self.weight_down1right2 = nn.Parameter(data=kernel_down1right2, requires_grad=False)
        
        self.pool = nn.AvgPool2d(patch_size)  # Default 4
    
    # --- Callable & Context Manager ---
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate the loss between input and target.
        
        Args:
            input: Predicted image, formatted as a torch.Tensor of dimensions
                (B, C, H, W) and values ranging from 0.0 to 1.0.
            target: Target image, formatted as a torch.Tensor of dimensions
                (B, C, H, W) and values ranging from 0.0 to 1.0.
        
        Returns:
            Loss value, formatted according to the specified reduction method.
        """
        # Ensure weights are on the same device as input
        if self.weight_left.device != input.device:
            self.weight_left = self.weight_left.to(input.device)
        if self.weight_right.device != input.device:
            self.weight_right = self.weight_right.to(input.device)
        if self.weight_up.device != input.device:
            self.weight_up = self.weight_up.to(input.device)
        if self.weight_down.device != input.device:
            self.weight_down = self.weight_down.to(input.device)
        if self.num_regions in [8, 16]:
            if self.weight_upleft.device != input.device:
                self.weight_upleft = self.weight_upleft.to(input.device)
            if self.weight_upright.device != input.device:
                self.weight_upright = self.weight_upright.to(input.device)
            if self.weight_downleft.device != input.device:
                self.weight_downleft = self.weight_downleft.to(input.device)
            if self.weight_downright.device != input.device:
                self.weight_downright = self.weight_downright.to(input.device)
        if self.num_regions in [16, 24]:
            if self.weight_left2.device != input.device:
                self.weight_left2 = self.weight_left2.to(input.device)
            if self.weight_right2.device != input.device:
                self.weight_right2 = self.weight_right2.to(input.device)
            if self.weight_up2.device != input.device:
                self.weight_up2 = self.weight_up2.to(input.device)
            if self.weight_down2.device != input.device:
                self.weight_down2 = self.weight_down2.to(input.device)
            if self.weight_up2left2.device != input.device:
                self.weight_up2left2 = self.weight_up2left2.to(input.device)
            if self.weight_up2right2.device != input.device:
                self.weight_up2right2 = self.weight_up2right2.to(input.device)
            if self.weight_down2left2.device != input.device:
                self.weight_down2left2 = self.weight_down2left2.to(input.device)
            if self.weight_down2right2.device != input.device:
                self.weight_down2right2 = self.weight_down2right2.to(input.device)
        if self.num_regions == 24:
            if self.weight_up2left1.device != input.device:
                self.weight_up2left1 = self.weight_up2left1.to(input.device)
            if self.weight_up2right1.device != input.device:
                self.weight_up2right1 = self.weight_up2right1.to(input.device)
            if self.weight_up1left2.device != input.device:
                self.weight_up1left2 = self.weight_up1left2.to(input.device)
            if self.weight_up1right2.device != input.device:
                self.weight_up1right2 = self.weight_up1right2.to(input.device)
            if self.weight_down2left1.device != input.device:
                self.weight_down2left1 = self.weight_down2left1.to(input.device)
            if self.weight_down2right1.device != input.device:
                self.weight_down2right1 = self.weight_down2right1.to(input.device)
            if self.weight_down1left2.device != input.device:
                self.weight_down1left2 = self.weight_down1left2.to(input.device)
            if self.weight_down1right2.device != input.device:
                self.weight_down1right2 = self.weight_down1right2.to(input.device)
                
        # Compute mean across channels
        org_mean     = torch.mean(input,  1, keepdim=True)
        enhance_mean = torch.mean(target, 1, keepdim=True)
        
        # Apply average pooling
        org_pool     = self.pool(org_mean)
        enhance_pool = self.pool(enhance_mean)
        
        # Compute differences using convolutions
        d_org_left   = F.conv2d(org_pool, self.weight_left,  padding=1)
        d_org_right  = F.conv2d(org_pool, self.weight_right, padding=1)
        d_org_up     = F.conv2d(org_pool, self.weight_up,    padding=1)
        d_org_down   = F.conv2d(org_pool, self.weight_down,  padding=1)
        if self.num_regions in [8, 16]:
            d_org_upleft    = F.conv2d(org_pool, self.weight_upleft,    padding=1)
            d_org_upright   = F.conv2d(org_pool, self.weight_upright,   padding=1)
            d_org_downleft  = F.conv2d(org_pool, self.weight_downleft,  padding=1)
            d_org_downright = F.conv2d(org_pool, self.weight_downright, padding=1)
        if self.num_regions in [16, 24]:
            d_org_left2       = F.conv2d(org_pool, self.weight_left2,       padding=2)
            d_org_right2      = F.conv2d(org_pool, self.weight_right2,      padding=2)
            d_org_up2         = F.conv2d(org_pool, self.weight_up2,         padding=2)
            d_org_down2       = F.conv2d(org_pool, self.weight_down2,       padding=2)
            d_org_up2left2    = F.conv2d(org_pool, self.weight_up2left2,    padding=2)
            d_org_up2right2   = F.conv2d(org_pool, self.weight_up2right2,   padding=2)
            d_org_down2left2  = F.conv2d(org_pool, self.weight_down2left2,  padding=2)
            d_org_down2right2 = F.conv2d(org_pool, self.weight_down2right2, padding=2)
        if self.num_regions == 24:
            d_org_up2left1    = F.conv2d(org_pool, self.weight_up2left1,    padding=2)
            d_org_up2right1   = F.conv2d(org_pool, self.weight_up2right1,   padding=2)
            d_org_up1left2    = F.conv2d(org_pool, self.weight_up1left2,    padding=2)
            d_org_up1right2   = F.conv2d(org_pool, self.weight_up1right2,   padding=2)
            d_org_down2left1  = F.conv2d(org_pool, self.weight_down2left1,  padding=2)
            d_org_down2right1 = F.conv2d(org_pool, self.weight_down2right1, padding=2)
            d_org_down1left2  = F.conv2d(org_pool, self.weight_down1left2,  padding=2)
            d_org_down1right2 = F.conv2d(org_pool, self.weight_down1right2, padding=2)
        
        d_enhance_left  = F.conv2d(enhance_pool, self.weight_left,  padding=1)
        d_enhance_right = F.conv2d(enhance_pool, self.weight_right, padding=1)
        d_enhance_up    = F.conv2d(enhance_pool, self.weight_up,    padding=1)
        d_enhance_down  = F.conv2d(enhance_pool, self.weight_down,  padding=1)
        if self.num_regions in [8, 16]:
            d_enhance_upleft    = F.conv2d(enhance_pool, self.weight_upleft,    padding=1)
            d_enhance_upright   = F.conv2d(enhance_pool, self.weight_upright,   padding=1)
            d_enhance_downleft  = F.conv2d(enhance_pool, self.weight_downleft,  padding=1)
            d_enhance_downright = F.conv2d(enhance_pool, self.weight_downright, padding=1)
        if self.num_regions in [16, 24]:
            d_enhance_left2       = F.conv2d(enhance_pool, self.weight_left2,       padding=2)
            d_enhance_right2      = F.conv2d(enhance_pool, self.weight_right2,      padding=2)
            d_enhance_up2         = F.conv2d(enhance_pool, self.weight_up2,         padding=2)
            d_enhance_down2       = F.conv2d(enhance_pool, self.weight_down2,       padding=2)
            d_enhance_up2left2    = F.conv2d(enhance_pool, self.weight_up2left2,    padding=2)
            d_enhance_up2right2   = F.conv2d(enhance_pool, self.weight_up2right2,   padding=2)
            d_enhance_down2left2  = F.conv2d(enhance_pool, self.weight_down2left2,  padding=2)
            d_enhance_down2right2 = F.conv2d(enhance_pool, self.weight_down2right2, padding=2)
        if self.num_regions == 24:
            d_enhance_up2left1    = F.conv2d(enhance_pool, self.weight_up2left1,    padding=2)
            d_enhance_up2right1   = F.conv2d(enhance_pool, self.weight_up2right1,   padding=2)
            d_enhance_up1left2    = F.conv2d(enhance_pool, self.weight_up1left2,    padding=2)
            d_enhance_up1right2   = F.conv2d(enhance_pool, self.weight_up1right2,   padding=2)
            d_enhance_down2left1  = F.conv2d(enhance_pool, self.weight_down2left1,  padding=2)
            d_enhance_down2right1 = F.conv2d(enhance_pool, self.weight_down2right1, padding=2)
            d_enhance_down1left2  = F.conv2d(enhance_pool, self.weight_down1left2,  padding=2)
            d_enhance_down1right2 = F.conv2d(enhance_pool, self.weight_down1right2, padding=2)
        
        # Compute squared differences
        d_left  = torch.pow(d_org_left  - d_enhance_left,  2)
        d_right = torch.pow(d_org_right - d_enhance_right, 2)
        d_up    = torch.pow(d_org_up    - d_enhance_up,    2)
        d_down  = torch.pow(d_org_down  - d_enhance_down,  2)
        if self.num_regions in [8, 16]:
            d_upleft    = torch.pow(d_org_upleft    - d_enhance_upleft,    2)
            d_upright   = torch.pow(d_org_upright   - d_enhance_upright,   2)
            d_downleft  = torch.pow(d_org_downleft  - d_enhance_downleft,  2)
            d_downright = torch.pow(d_org_downright - d_enhance_downright, 2)
        if self.num_regions in [16, 24]:
            d_left2       = torch.pow(d_org_left2       - d_enhance_left2,       2)
            d_right2      = torch.pow(d_org_right2      - d_enhance_right2,      2)
            d_up2         = torch.pow(d_org_up2         - d_enhance_up2,         2)
            d_down2       = torch.pow(d_org_down2       - d_enhance_down2,       2)
            d_up2left2    = torch.pow(d_org_up2left2    - d_enhance_up2left2,    2)
            d_up2right2   = torch.pow(d_org_up2right2   - d_enhance_up2right2,   2)
            d_down2left2  = torch.pow(d_org_down2left2  - d_enhance_down2left2,  2)
            d_down2right2 = torch.pow(d_org_down2right2 - d_enhance_down2right2, 2)
        if self.num_regions == 24:
            d_up2left1    = torch.pow(d_org_up2left1    - d_enhance_up2left1,    2)
            d_up2right1   = torch.pow(d_org_up2right1   - d_enhance_up2right1,   2)
            d_up1left2    = torch.pow(d_org_up1left2    - d_enhance_up1left2,    2)
            d_up1right2   = torch.pow(d_org_up1right2   - d_enhance_up1right2,   2)
            d_down2left1  = torch.pow(d_org_down2left1  - d_enhance_down2left1,  2)
            d_down2right1 = torch.pow(d_org_down2right1 - d_enhance_down2right1, 2)
            d_down1left2  = torch.pow(d_org_down1left2  - d_enhance_down1left2,  2)
            d_down1right2 = torch.pow(d_org_down1right2 - d_enhance_down1right2, 2)
        
        # Aggregate loss
        loss = d_left + d_right + d_up + d_down
        if self.num_regions in [8, 16]:
            loss += d_upleft + d_upright + d_downleft + d_downright
        if self.num_regions in [16, 24]:
            loss += (d_left2 + d_right2 + d_up2 + d_down2 +
                     d_up2left2 + d_up2right2 + d_down2left2 + d_down2right2)
        if self.num_regions == 24:
            loss += (d_up2left1 + d_up2right1 + d_up1left2 + d_up1right2 +
                     d_down2left1 + d_down2right1 + d_down1left2 + d_down1right2)
        
        # Apply reduction and weighting
        loss = self.reduce(loss=loss)
        return loss


class TotalVariationLoss(BaseLoss):
    """A loss function for reducing noise by encouraging piecewise smoothness.
    
    Encourage spatial smoothness in the enhanced image by penalizing large
    intensity variations between neighboring pixels.

    References:
        - https://github.com/Li-Chongyi/Zero-DCE/blob/master/Zero-DCE_code/Myloss.py
    """
    
    # --- Lifecycle & Initialization ---
    def __init__(self, reduction: str = "mean"):
        """Initialize a new instance.
        
        Args:
            reduction: Reduction method to apply to the loss. Can be one of
                "none", "mean", or "sum". Defaults to "mean".
        """
        super().__init__(reduction=reduction)
    
    # --- Callable & Context Manager ---
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Calculate the loss for the input tensor.
        
        Args:
            input: Predicted image, formatted as a torch.Tensor of dimensions
                (B, C, H, W) and values ranging from 0.0 to 1.0.
        
        Returns:
            Loss value, formatted according to the specified reduction method.
        """
        x          = input
        b, c, h, w = x.size()
        count_h    = (x.size()[2]-1) * x.size()[3]
        count_w    =  x.size()[2] * (x.size()[3] - 1)
        h_tv       = torch.pow((x[:, :, 1:,  :] - x[:, :, :h - 1, :]), 2).sum()
        w_tv       = torch.pow((x[:, :,  :, 1:] - x[:, :, :, :w - 1]), 2).sum()
        loss       = 2 * (h_tv / count_h + w_tv / count_w) / b
        return loss


class EdgeLoss(BaseLoss):
    """A loss function for penalizing blurry boundaries.
    
    Preserve edge details in images by computing the Laplacian edge maps and
    penalizing differences between the input and target images.
    
    Attributes:
        kernel (torch.Tensor): Gaussian kernel for convolution.
        loss (CharbonnierLoss): Charbonnier loss function instance.
    """
    
    # --- Lifecycle & Initialization ---
    def __init__(self, reduction: str = "mean"):
        """Initialize a new instance.
        
        Args:
            reduction: Reduction method to apply to the loss. Can be one of
                "none", "mean", or "sum". Defaults to "mean".
        """
        super().__init__(reduction=reduction)
        k           = torch.Tensor([[0.05, 0.25, 0.4, 0.25, 0.05]])
        self.kernel = torch.matmul(k.t(), k).unsqueeze(0).repeat(3, 1, 1, 1)
        self.loss   = CharbonnierLoss()

    # --- Callable & Context Manager ---
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate the loss between input and target.
        
        Args:
            input: Predicted image, formatted as a torch.Tensor of dimensions
                (B, C, H, W) and values ranging from 0.0 to 1.0.
            target: Target image, formatted as a torch.Tensor of dimensions
                (B, C, H, W) and values ranging from 0.0 to 1.0.
                
        Returns:
            Loss value, formatted according to the specified reduction method.
        """
        edge1 = self._laplacian_kernel(input)
        edge2 = self._laplacian_kernel(target)
        diff  = edge1 - edge2
        loss  = torch.mean(torch.sqrt((diff * diff) + (self.eps * self.eps)))
        loss  = self.reduce(loss=loss)
        return loss

    def _gauss_conv(self, image: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian convolution to the input image.
        
        Args:
            image: An image, formatted as a torch.Tensor of dimensions
                (B, C, H, W) and values ranging from 0.0 to 1.0.
        
        Returns:
            Gaussian filtered image.
        """
        b, c, w, h  = self.kernel.shape
        self.kernel = self.kernel.to(image.device)
        image       = F.pad(image, (w // 2, h // 2, w // 2, h // 2), mode="replicate")
        # gauss       = F.conv2d(image, self.kernel, groups=b)  # Old code
        gauss       = F.conv2d(image, self.kernel, groups=c)  # Groups=c for channel-wise convolution
        return gauss
    
    def _laplacian_kernel(self, image: torch.Tensor) -> torch.Tensor:
        """Compute the Laplacian edge map using a Gaussian pyramid.
        
        Args:
            image: An image, formatted as a torch.Tensor of dimensions
                (B, C, H, W) and values ranging from 0.0 to 1.0.
        
        Returns:
            Laplacian edge map.
        """
        filtered   = self._gauss_conv(image)       # filter
        down       = filtered[:, :, ::2, ::2]     # downsample
        new_filter = torch.zeros_like(filtered)
        new_filter[:, :, ::2, ::2] = down * 4     # upsample
        filtered   = self._gauss_conv(new_filter)  # filter
        diff       = image - filtered
        return diff


# ==============================================================================
# GEOMETRY & CONTEXT-AWARE LOSSES
# ==============================================================================

class DepthAwareIlluminationLoss(BaseLoss):
    """A loss function for smoothing lighting while respecting 3D depth boundaries.
    
    Encourage smoothness in the illumination map while preserving depth
    discontinuities.
    
    Attributes:
        alpha (float): Weighting factor for depth influence.
    """
    
    # --- Lifecycle & Initialization ---
    def __init__(self, alpha: float = 1.0, reduction: str = "mean"):
        """Initialize a new instance.
        
        Args:
            alpha: Weighting factor for depth influence. Defaults to 1.0.
            reduction: Reduction method to apply to the loss. Can be one of
                "none", "mean", or "sum". Defaults to "mean".
        """
        super().__init__(reduction=reduction)
        self.alpha = alpha
    
    # --- Callable & Context Manager ---
    def forward(self, input: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """Calculate the loss between illumination map and depth map.
        
        Args:
            input: Illumination map, formatted as a torch.Tensor of dimensions
                (B, 1, H, W) and values ranging from 0.0 to 1.0.
            depth: Depth map, formatted as a torch.Tensor of dimensions
                (B, 1, H, W) and values ranging from 0.0 to 1.0.
                
        Returns:
            Loss value, formatted according to the specified reduction method.
        """
        # Calculate gradients of illumination map (L) in x and y directions
        L_dx = input[:, :, :, 1:] - input[:, :, :, :-1]
        L_dy = input[:, :, 1:, :] - input[:, :, :-1, :]
        
        # Calculate gradients of depth map (D) in x and y directions
        D_dx = depth[:, :, :, 1:] - depth[:, :, :, :-1]
        D_dy = depth[:, :, 1:, :] - depth[:, :, :-1, :]
        
        # Compute depth-weighted terms for x and y directions
        weight_dx = torch.exp(-self.alpha * torch.abs(D_dx))
        weight_dy = torch.exp(-self.alpha * torch.abs(D_dy))
        
        # Apply depth weights to illumination gradients and take the mean
        loss_dx = torch.mean(weight_dx * torch.abs(L_dx))
        loss_dy = torch.mean(weight_dy * torch.abs(L_dy))
        
        # Sum the losses from both directions
        loss = loss_dx + loss_dy
        loss = self.reduce(loss=loss)
        return loss


class StructureTextureDecompositionLoss(nn.Module):
    """A loss function for separating structural edges from fine details.
    
    Separates an image into structure and texture components using Gaussian
    blurring, and penalizes the texture component to encourage smoother textures
    in the enhanced image.
    
    Attributes:
        kernel_size (list[int]): Size of the Gaussian kernel for blurring.
        sigma (list[float]): Standard deviation for the Gaussian kernel.
    """
    
    # --- Lifecycle & Initialization ---
    def __init__(self, kernel_size: int = 3, sigma: float = 1.0):
        """Initialize a new instance.
        
        Args:
            kernel_size: Size of the Gaussian kernel for blurring. Defaults to 3.
            sigma: Standard deviation for the Gaussian kernel. Defaults to 1.0.
        """
        super().__init__()
        self.kernel_size = [kernel_size, kernel_size]
        self.sigma       = [sigma, sigma]

    # --- Callable & Context Manager ---
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Calculate the loss for the input tensor.
        
        Args:
            input: Predicted image, formatted as a torch.Tensor of dimensions
                (B, C, H, W) and values ranging from 0.0 to 1.0.
        
        Returns:
            Loss value, formatted according to the specified reduction method.
        """
        # Create a blurred version of the image to represent the "structure"
        structure = gaussian_blur(input, kernel_size=self.kernel_size, sigma=self.sigma)
        # The "texture" is the difference between the "input" and the "structure"
        texture   = input - structure
        # Penalize the L1 norm of the texture component
        return torch.mean(torch.abs(texture))

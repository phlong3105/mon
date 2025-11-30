#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module for feature fusion layers.

This module provides classes for various feature fusion techniques used in
neural networks, including Attentional Feature Fusion (AFF), Direct Add Fuse
(DAF), Multi-Scale Channel Attention Module (MS-CAM), and Iterative Attentional
Feature Fusion (iAFF).

References:
    - Paper: "Attentional Feature Fusion," WACV 2021.
    - Code: https://github.com/YimianDai/open-aff/tree/master/aff_pytorch
"""

__all__ = [
    "AFF",
    "DAF",
    "MS_CAM",
    "iAFF",
]

import torch
import torch.nn as nn


class DAF(nn.Module):
    """Direct Add Fuse (DAF) layer."""
    
    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """Forward pass of the DAF layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
            residual (torch.Tensor): Residual tensor of shape (B, C, H, W).
        
        Returns:
            torch.Tensor: Fused output tensor of shape (B, C, H, W).
        """
        return x + residual


class MS_CAM(nn.Module):
    """Multi-Scale Channel Attention Module (MS-CAM) layer.
    
    References:
        - Paper: "Attentional Feature Fusion," WACV 2021.
        - Code: https://github.com/YimianDai/open-aff/tree/master/aff_pytorch
    """

    def __init__(self, channels: int = 64, ratio: int = 4):
        """Initializes the MS-CAM layer.
        
        Args:
            channels (int): Number of input channels. Defaults to 64.
            ratio (int): Reduction ratio for the intermediate channels.
                Defaults to 4.
        """
        super().__init__()
        mid_channels   = int(channels // ratio)
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, mid_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the MS-CAM layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
            
        Returns:
            torch.Tensor: Output tensor after applying MS-CAM of shape (B, C, H, W).
        """
        x_l  = self.local_att(x)
        x_g  = self.global_att(x)
        x_lg = x_l + x_g
        w    = self.sigmoid(x_lg)
        return x * w
    

class AFF(nn.Module):
    """Attentional Feature Fusion (AFF) layer.
    
    References:
        - Paper: "Attentional Feature Fusion," WACV 2021.
        - Code: https://github.com/YimianDai/open-aff/tree/master/aff_pytorch
    """

    def __init__(self, channels: int = 64, ratio: int = 4):
        """Initializes the AFF layer.
        
        Args:
            channels (int): Number of input channels. Defaults to 64.
            ratio (int): Reduction ratio for the intermediate channels.
                Defaults to 4.
        """
        super().__init__()
        mid_channels   = int(channels // ratio)
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, mid_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """Forward pass of the AFF layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
            residual (torch.Tensor): Residual tensor of shape (B, C, H, W
            
        Returns:
            torch.Tensor: Fused output tensor of shape (B, C, H, W).
        """
        x_a  = x + residual
        x_l  = self.local_att(x_a)
        x_g  = self.global_att(x_a)
        x_lg = x_l + x_g
        w    = self.sigmoid(x_lg)
        x_o  = 2 * x * w + 2 * residual * (1 - w)
        return x_o


class iAFF(nn.Module):
    """Iterative Attentional Feature Fusion (iAFF) layer.
    
    References:
        - Paper: "Attentional Feature Fusion," WACV 2021.
        - Code: https://github.com/YimianDai/open-aff/tree/master/aff_pytorch
    """

    def __init__(self, channels: int = 64, ratio: int = 4):
        """Initializes the iAFF layer.
        
        Args:
            channels (int): Number of input channels. Defaults to 64.
            ratio (int): Reduction ratio for the intermediate channels.
                Defaults to 4.
        """
        super().__init__()
        mid_channels   = int(channels // ratio)
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, mid_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.local_att2 = nn.Sequential(
            nn.Conv2d(channels, mid_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.global_att2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, mid_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """Forward pass of the iAFF layer.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
            residual (torch.Tensor): Residual tensor of shape (B, C, H, W).
            
        Returns:
            torch.Tensor: Fused output tensor of shape (B, C, H, W).
        """
        x_a   = x + residual
        x_l1  = self.local_att(x_a)
        x_g1  = self.global_att(x_a)
        x_lg1 = x_l1 + x_g1
        w1    = self.sigmoid(x_lg1)
        x_i   = x * w1 + residual * (1 - w1)

        x_l2  = self.local_att2(x_i)
        x_g2  = self.global_att2(x_i)
        x_lg2 = x_l2 + x_g2
        w2    = self.sigmoid(x_lg2)
        x_o   = x * w2 + residual * (1 - w2)
        return x_o

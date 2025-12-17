#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A module for Convolutional Implicit Neural Representation (Conv-INR).

This module implements the Conv-INR architecture for representing multimodal visual
signals using convolutional layers.

References:
    - Paper: "Conv-INR: Convolutional Implicit Neural Representation for
      Multimodal Visual Signals," arXiv 2025.
"""

__all__ = [
    "ConvINR",
    "ConvINRLayer",
]

import torch
import torch.nn as nn


# ----- Layer -----
class ConvINRLayer(nn.Module):
    """A single layer of the Conv-INR architecture."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        """Initializes the ConvINRLayer.
        
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Size of the convolutional kernel. Defaults to 3.
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn   = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Conv INR layer.
        
        Args:
            input (torch.Tensor): Input tensor of shape (B, C_in, H, W).
            
        Returns:
            torch.Tensor: Output tensor of shape (B, C_out, H, W).
        """
        return self.relu(self.bn(self.conv(input)))


# ----- MLP -----
class ConvINR(nn.Module):
    """Convolutional Implicit Neural Representation (Conv-INR) model."""
    
    def __init__(
        self,
        in_channels  : int,
        out_channels : int,
        hidden_dim   : int = 32,
        hidden_layers: int = 10,
        kernel_size  : int = 3,
    ):
        """Initializes the Conv-INR model.
        
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            hidden_dim (int): Dimension of the hidden layers. Defaults to 32.
            hidden_layers (int): Number of hidden layers. Defaults to 10.
            kernel_size (int): Size of the convolutional kernel. Defaults to 3.
        """
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        
        # First layer
        self.net = []
        self.net.append(ConvINRLayer(in_channels, hidden_dim, kernel_size))
        # Hidden layers
        for i in range(hidden_layers):
            self.net.append(ConvINRLayer(hidden_dim, hidden_dim, kernel_size))
        # Final layer
        self.net.append(nn.Conv2d(hidden_dim, out_channels, kernel_size, padding=kernel_size // 2))
        
        self.net = nn.Sequential(*self.net)
        
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Conv-INR model.
        
        Args:
            coords (torch.Tensor): Input coordinates tensor of shape (B, C, H, W).
            
        Returns:
            torch.Tensor: Output tensor of shape (B, C_out, H, W).
        """
        b, c, h, w = coords.shape
        if c > self.in_channels:
            coords = coords.view(b, c, h, w).permute(0, 3, 1, 2)  # B x H x W x C -> B x C x H x W
        output = self.net(coords)
        if c > self.in_channels:
            output = output.permute(0, 2, 3, 1)  # B x C x H x W -> B x H x W x C
        return output

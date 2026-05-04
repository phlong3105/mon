#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implicit Neural Representations (INRs).

This module implements various Implicit Neural Representation (INR) models used
for modelling data like images, 3D shapes, or audio as continuous functions.
"""

from __future__ import annotations

__all__ = [
    "FFN",
    "Finer",
    "Finer_PP",
    "Siren",
    "create_coords",
    "create_depth_aware_patches",
    "create_patches",
    "interpolate_image",
    "pair_downsampler",
]

import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from mon.core import Size
from mon.nn.modules import FINERLinear, FourierPE, SineLinear


# ==============================================================================
# region IMPLICIT NEURAL REPRESENTATIONS
# ==============================================================================

# --- Positional Encoding Based (Standard MLPs) ---

class FFN(nn.Module):
    """Fourier Feature Network (FFN).

    References:
        - Code: https://github.com/liuzhen0212/FINER/blob/main/models.py
    """

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_dim: int,
        hidden_layers: int,
        B: float = 20.0,
    ):
        """Initialize a new instance.

        Args:
            in_features (int): Size of each input sample.
            out_features (int): Size of each output sample.
            hidden_dim (int): Number of hidden units in each hidden layer.
            hidden_layers (int): Number of hidden layers.
            B (float, optional): Fourier feature mapping scale factor.
                Defaults to 20.0.
        """
        super().__init__()

        self.encoding = FourierPE(mapping_size=in_features, B=B)

        # First layer
        net = []
        net.append(nn.Linear(int(self.encoding.out_features), hidden_dim))
        net.append(nn.ReLU(True))
        # Hidden layers
        for i in range(hidden_layers):
            net.append(nn.Linear(hidden_dim, hidden_dim))
            net.append(nn.ReLU(True))
        # Final layer
        final_linear = nn.Linear(hidden_dim, out_features)
        net.append(final_linear)

        self.net = nn.Sequential(*net)

    # --- Callable & Context Manager ---
    def forward(self, coords: Tensor) -> Tensor:
        """Forward the input through the network.

        Args:
            coords (Tensor): Input tensor of shape (..., in_features) and
                values ranging from 0.0 to 1.0.

        Returns:
            Tensor: Output tensor of shape (..., out_features) and values
                ranging from 0.0 to 1.0.
        """
        return self.net(self.encoding(coords))


# --- Periodic Activation Based (SIREN Variants) ---

class Siren(nn.Module):
    """SIREN MLP using sine activation functions.

    References:
        - Paper: "Implicit Neural Representations with Periodic Activation
          Functions," NeurIPS 2020.
        - Code: https://github.com/vsitzmann/siren
        - Code: https://github.com/liuzhen0212/FINER/blob/main/models.py
    """

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_dim: int,
        num_layers: int = 4,
        w0: float = 30.0,
        w: float = 30.0,
    ):
        """Initialize a new instance.

        Args:
            in_features (int): Size of each input sample.
            out_features (int): Size of each output sample.
            hidden_dim (int): Hidden channel dimensions.
            num_layers (int, optional): Number of layers in the MLP. Defaults to 4.
            w0 (float, optional): Frequency scaling factor for the first layer.
                Defaults to 30.0.
            w (float, optional): Frequency scaling factor for the hidden layers.
                Defaults to 30.0.
        """
        super().__init__()

        # Define network
        net = []
        # First layer
        net.append(SineLinear(in_features, hidden_dim, w0, is_first=True))
        # Hidden layers
        for i in range(num_layers - 2):
            net.append(SineLinear(in_features, hidden_dim, w, is_first=False))
        # Final layer
        final_linear = nn.Linear(hidden_dim, out_features)
        with torch.no_grad():
            final_linear.weight.uniform_(
                -np.sqrt(6.0 / hidden_dim) / w,
                 np.sqrt(6.0 / hidden_dim) / w
            )
        net.append(final_linear)

        self.net = nn.Sequential(*net)

    # --- Callable & Context Manager ---
    def forward(self, coords: Tensor) -> Tensor:
        """Forward the input through the network.

        Args:
            coords (Tensor): Input tensor of shape (..., in_features) and
                values ranging from 0.0 to 1.0.

        Returns:
            Tensor: Output tensor of shape (..., out_features) and values
                ranging from 0.0 to 1.0.
        """
        return self.net(coords)


class Finer(nn.Module):
    """FINER MLP.

    References:
        - Paper: "FINER: Flexible spectral-bias tuning in Implicit NEural
          Representation by Variable-periodic Activation Functions," CVPR 2024.
        - Code: https://github.com/liuzhen0212/FINER
    """

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_dim: int,
        num_layers: int = 4,
        w0: float = 30.0,
        w: float = 30.0,
        first_bias_scale: float | None = None,
        scale_req_grad: bool = False,
    ):
        """Initialize a new instance.

        Args:
            in_features (int): Size of each input sample.
            out_features (int): Size of each output sample.
            hidden_dim (int): Hidden channel dimensions.
            num_layers (int, optional): Number of layers in the MLP. Defaults to 4.
            w0 (float, optional): Frequency scaling factor for the first layer.
                Defaults to 30.0.
            w (float, optional): Frequency scaling factor for the hidden layers.
                Defaults to 30.0.
            first_bias_scale (float | None, optional): Bias scale for the first
                layer. Defaults to None.
            scale_req_grad (bool, optional): Scale requires gradient if True.
                Defaults to False.
        """
        super().__init__()

        # Define network
        net = []
        # First layer
        net.append(
            FINERLinear(
                in_features, hidden_dim, w0,
                first_bias_scale=first_bias_scale,
                scale_req_grad=scale_req_grad,
                is_first=True,
            ),
        )
        # Hidden layers
        for i in range(num_layers - 2):
            net.append(FINERLinear(hidden_dim, hidden_dim, w, scale_req_grad=scale_req_grad))
        # Final layer
        final_linear = nn.Linear(hidden_dim, out_features)
        with torch.no_grad():
            final_linear.weight.uniform_(
                -np.sqrt(6.0 / hidden_dim) / w,
                 np.sqrt(6.0 / hidden_dim) / w
            )
        net.append(final_linear)

        self.net = nn.Sequential(*net)

    # --- Callable & Context Manager ---
    def forward(self, coords: Tensor) -> Tensor:
        """Forward the input through the network.

        Args:
            coords (Tensor): Input tensor of shape (..., in_features) and
                values ranging from 0.0 to 1.0.

        Returns:
            Tensor: Output tensor of shape (..., out_features) and values
                ranging from 0.0 to 1.0.
        """
        return self.net(coords)


class Finer_PP(nn.Module):
    """FINER++ MLP.

    References:
        - Paper: "FINER++: Building a Family of Variable-periodic Functions for
          Activating Implicit Neural Representation," arXiv 2025.
        - Code: https://github.com/liuzhen0212/FINER
    """

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_dim: int,
        num_layers: int = 4,
        w0: float = 30.0,
        w: float = 30.0,
        first_bias_scale: float = 5,
        scale_req_grad: bool = False,
    ):
        """Initialize a new instance.

        Args:
            in_features (int): Size of each input sample.
            out_features (int): Size of each output sample.
            hidden_dim (int): Hidden channel dimensions.
            num_layers (int, optional): Number of layers in the MLP. Defaults to 4.
            w0 (float, optional): Frequency scaling factor for the first layer.
                Defaults to 30.0.
            w (float, optional): Frequency scaling factor for the hidden layers.
                Defaults to 30.0.
            first_bias_scale (float, optional): Bias scale for the first layer.
                Defaults to None.
            scale_req_grad (bool, optional): Scale requires gradient if True.
                Defaults to False.
        """
        super().__init__()
        self.out_features = out_features

        # Define network
        net = []
        # First layer
        net.append(
            FINERLinear(
                in_features, hidden_dim, w0,
                first_bias_scale=first_bias_scale,
                scale_req_grad=scale_req_grad,
                is_first=True,
            )
        )
        # Hidden layers
        for i in range(num_layers - 2):
            net.append(FINERLinear(hidden_dim, hidden_dim, w, scale_req_grad=scale_req_grad))
        # Final layer
        final_linear = nn.Linear(hidden_dim, out_features)
        with torch.no_grad():
            final_linear.weight.uniform_(
                -np.sqrt(6.0 / hidden_dim) / w,
                 np.sqrt(6.0 / hidden_dim) / w
            )
        net.append(final_linear)

        self.net = nn.Sequential(*net)

    # --- Callable & Context Manager ---
    def forward(self, coords: Tensor) -> Tensor:
        """Forward the input through the network.

        Args:
            coords (Tensor): Input tensor of shape (..., in_features) and
                values ranging from 0.0 to 1.0.

        Returns:
            Tensor: Output tensor of shape (..., out_features) and values
                ranging from 0.0 to 1.0.
        """
        output = self.net(coords)
        return output.view(-1, self.out_features)

# endregion


# ==============================================================================
# region UTILITIES
# ==============================================================================

# --- Coordinate Generation & Embedding ---

def create_coords(size: Size, device: torch.device) -> Tensor:
    """Create a normalized square coordinates grid.

    Args:
        size (Size): Size of the grid.
        device (torch.device): Device to place the coordinates on.

    Returns:
        Tensor: Coordinates tensor of shape (1, H, W, 2) and values ranging from
            -1.0 to 1.0.
    """
    size = Size.from_value(size)
    h, w = size.hw
    # TODO: Old code normalize from 0 to 1. Delete later
    # x_range = torch.linspace(0, 1, w, device=device)
    # y_range = torch.linspace(0, 1, h, device=device)
    x_range = torch.linspace(-1, 1, w, device=device)
    y_range = torch.linspace(-1, 1, h, device=device)
    y, x = torch.meshgrid(y_range, x_range, indexing="ij")
    # Stack to get (H, W, 2) then expand to (1, H, W, 2)
    coords = torch.stack([x, y], dim=-1).unsqueeze(0)
    return coords


# --- Spatial Context & Patch Extraction ---

def create_patches(image: Tensor, kernel_size: int = 7) -> Tensor:
    """Create a tensor where the channel contains patch information.

    Args:
        image (Tensor): Image, formatted as a tensor of shape (1, C, H, W) and
            pixel values ranging from 0.0 to 1.0.
        kernel_size (int, optional): Size of square patches. Defaults to 7.

    Returns:
        Tensor: A tensor of shape (1, H', W', K^2) where H' and W' are the
            height and width after patch extraction, and K is the ``kernel_size``.

    Raises:
        ValueError: If the input ``image`` does not have 4 dimensions.
    """
    if image.ndim != 4:
        raise ValueError(
            f"Expected 'image' to be a 4D tensor, but got: {image.ndim}D."
        )

    b, c, h, w = image.shape
    k = kernel_size
    device = image.device

    kernel = torch.zeros((k ** 2, c, k, k)).to(device)
    for i in range(k):
        for j in range(k):
            kernel[i + j * k, :, i, j] = 1

    pad = nn.ReflectionPad2d(k // 2)
    image_padded = pad(image)
    patches = F.conv2d(image_padded, kernel, padding=0).squeeze(0)
    return torch.movedim(patches, 0, -1)


def create_depth_aware_patches(
    image: Tensor,
    depth: Tensor,
    kernel_size: int = 7,
    alpha: float = 8.3
) -> Tensor:
    """Create depth-aware patches for the given image and depth map.

    Args:
        image (Tensor): Image, formatted as a Tensor of shape
            (1, C, H, W) and values ranging from 0.0 to 1.0.
        depth (Tensor): Depth map, formatted as a Tensor of shape
            (1, 1, H, W) and values ranging from 0.0 to 1.0.
        kernel_size (int, optional): Size of square patches. Defaults to 7.
        alpha (float, optional): Exponential decay factor for depth-aware
            weighting. Defaults to 8.3.

    Returns:
        Tensor: A tensor of shape (1, H', W', K^2) where H' and W' are the
            height and width after patch extraction, and K is the ``kernel_size``.
    """
    b, c, h, w = image.shape
    k = kernel_size
    device = image.device

    kernel = torch.zeros((k ** 2, c, k, k)).to(device)
    for i in range(k):
        for j in range(k):
            kernel[i + j * k, 0, i, j] = 1

    pad = nn.ReflectionPad2d(kernel_size // 2)
    image_padded = pad(image)
    image_patches = F.conv2d(image_padded, kernel, padding=0).squeeze(0)
    depth_padded = pad(depth)
    depth_patches = F.conv2d(depth_padded, kernel, padding=0).squeeze(0)

    # Compute center index in patch
    center_idx = (k ** 2) // 2
    depth_center = depth_patches[center_idx, :, :].unsqueeze(0).repeat(k ** 2, 1, 1)

    # FD = exp(-alpha * |depth_center - depth_neighbor|)
    depth_diff = torch.abs(depth_center - depth_patches)
    fd = torch.exp(-alpha * depth_diff)  # Shape for multiplication

    # Weight the image patches and normalize
    patches = image_patches * fd
    weights_sum = fd.sum(dim=0, keepdim=True) + 1e-6  # Avoid division by zero
    patches = patches / weights_sum

    return torch.movedim(patches, 0, -1)


# --- Multi-scale & Sampling Operations ---

def pair_downsampler(image: Tensor) -> tuple[Tensor, Tensor]:
    """Downsample the image into two sub-images using learned filters.

    Args:
        image (Tensor): Image tensor of shape (1, C, H, W) and pixel values
            ranging from 0.0 to 1.0.

    Returns:
        tuple[Tensor, Tensor]: Two downsampled images, each is a tensor of
            shape (1, C, H/2, W/2) and values ranging from 0.0 to 1.0.
    """
    c = image.shape[1]
    device = image.device
    filter1 = torch.FloatTensor([[[[0, 0.5], [0.5, 0]]]]).to(device)
    filter1 = filter1.repeat(c, 1, 1, 1)
    filter2 = torch.FloatTensor([[[[0.5, 0], [0, 0.5]]]]).to(device)
    filter2 = filter2.repeat(c, 1, 1, 1)
    output1 = F.conv2d(image, filter1, stride=2, groups=c)
    output2 = F.conv2d(image, filter2, stride=2, groups=c)
    return output1, output2


def interpolate_image(image: Tensor, size: int) -> Tensor:
    """Resize the image to the specified size.

    Args:
        image (Tensor): Image tensor of shape (B, C, H, W) and values ranging
            from 0.0 to 1.0.
        size (int): Desired output size.

    Returns:
        Tensor: Resized image tensor of shape (B, C, size, size) and values
            ranging from 0.0 to 1.0.
    """
    return F.interpolate(image, size=(size, size), mode="bicubic")

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implicit Neural Representations (INRs).

This module implements various Implicit Neural Representation (INR) models used
for modelling data like images, 3D shapes, or audio as continuous functions.
"""

from __future__ import annotations

__all__ = [
    "FFN",
    "FINER",
    "FINER_PP",
    "GAUSS",
    "PosEncodingMLP",
    "SIREN",
    "WIRE",
    "create_coords",
    "create_depth_aware_patches",
    "create_noisy_coords",
    "create_patches",
    "ff_embedding",
    "interpolate_image",
    "pair_downsampler",
]

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mon.core import image as I
from ...comp import (
    ComplexGaborLayer,
    FINERLinear,
    GaussLinear,
    PosEncodingFourier,
    PosEncodingNeRF,
    SineLinear,
)


# ==============================================================================
# IMPLICIT NEURAL REPRESENTATIONS (INRs)
# ==============================================================================

# --- Positional Encoding Based (Standard MLPs) ---

class FFN(nn.Module):
    """Fourier Feature Network (FFN).

    References:
        - Code: https://github.com/liuzhen0212/FINER/blob/main/models.py

    Attributes:
        encoding (PosEncodingFourier): Positional encoding layer.
        net (torch.nn.Sequential): The MLP network.
    """

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        in_features  : int,
        out_features : int,
        hidden_dim   : int,
        hidden_layers: int,
        B            : float = 20.0,
        bias         : bool  = True,
    ):
        """Initialize a new instance.

        Args:
            in_features: Size of each input sample.
            out_features: Size of each output sample.
            hidden_dim: Number of hidden units in each hidden layer.
            hidden_layers: Number of hidden layers.
            B: Standard deviation of the Gaussian distribution used to sample
                the projection matrix. If set to None, no projection is applied.
                Defaults to 20.0.
            bias: If True, adds a learnable bias to the linear layers.
                Defaults to True.
        """
        super().__init__()
        self.encoding = PosEncodingFourier(in_features=in_features, B=B)

        # First layer
        net = []
        net.append(nn.Linear(int(self.encoding.out_features), hidden_dim, bias=bias))
        net.append(nn.ReLU(True))
        # Hidden layers
        for i in range(hidden_layers):
            net.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
            net.append(nn.ReLU(True))
        # Final layer
        final_linear = nn.Linear(hidden_dim, out_features, bias=bias)
        net.append(final_linear)

        self.net = nn.Sequential(*net)

    # --- Callable & Context Manager ---
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Forward the input through the network.

        Args:
            coords: Input tensor of shape (..., in_features) and values ranging
                from 0.0 to 1.0.

        Returns:
            Output tensor of shape (..., out_features) and values ranging
            from 0.0 to 1.0.
        """
        return self.net(self.encoding(coords))


class PosEncodingMLP(nn.Module):
    """Positional Encoding (PE) MLP.

    References:
        - Code: https://github.com/liuzhen0212/FINER/blob/main/models.py

    Attributes:
        encoding (PosEncodingNeRF): Positional encoding layer.
        net (torch.nn.Sequential): The MLP network.
    """

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        in_features    : int,
        out_features   : int,
        hidden_dim     : int,
        hidden_layers  : int,
        num_frequencies: int  = 10,
        bias           : bool = True,
    ):
        """Initialize a new instance.

        Args:
            in_features: Size of each input sample.
            out_features: Size of each output sample.
            hidden_dim: Number of hidden units in each hidden layer.
            hidden_layers: Number of hidden layers.
            num_frequencies: Number of frequency bands for positional encoding.
                Defaults to 10.
            bias: If True, adds a learnable bias to the linear layers.
                Defaults to True.
        """
        super().__init__()
        self.encoding = PosEncodingNeRF(in_features=in_features, num_frequencies=num_frequencies)

        # First layer
        net = []
        net.append(nn.Linear(int(self.encoding.out_features), hidden_dim, bias=bias))
        net.append(nn.ReLU(True))
        # Hidden layers
        for i in range(hidden_layers):
            net.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
            net.append(nn.ReLU(True))
        # Final layer
        final_linear = nn.Linear(hidden_dim, out_features, bias=bias)
        net.append(final_linear)

        self.net = nn.Sequential(*net)

    # --- Callable & Context Manager ---
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Forward the input through the network.

        Args:
            coords: Input tensor of shape (..., in_features) and values ranging
                from 0.0 to 1.0.

        Returns:
            Output tensor of shape (..., out_features) and values ranging
            from 0.0 to 1.0.
        """
        return self.net(self.encoding(coords))


# --- Periodic Activation Based (SIREN Variants) ---

class SIREN(nn.Module):
    """SIREN MLP using sine activation functions.

    References:
        - Paper: "Implicit Neural Representations with Periodic Activation Functions,"
          NeurIPS 2020.
        - Code: https://github.com/vsitzmann/siren
        - Code: https://github.com/liuzhen0212/FINER/blob/main/models.py

    Attributes:
        net (torch.nn.Sequential): The MLP network.
    """

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        in_features   : int,
        out_features  : int,
        hidden_dim    : int,
        hidden_layers : int,
        first_omega_0 : float = 30.0,
        hidden_omega_0: float = 30.0,
        bias          : bool  = True,
    ):
        """Initialize a new instance.

        Args:
            in_features: Size of each input sample.
            out_features: Size of each output sample.
            hidden_dim: Number of hidden units in each hidden layer.
            hidden_layers: Number of hidden layers.
            first_omega_0: Frequency scaling factor for the first layer.
                Defaults to 30.0.
            hidden_omega_0: Frequency scaling factor for the hidden layers.
                Defaults to 30.0.
            bias: If True, adds a learnable bias to the linear layers.
        """
        super().__init__()
        # First layer
        net = []
        net.append(
            SineLinear(
                in_features  = in_features,
                out_features = hidden_dim,
                bias         = bias,
                is_first     = True,
                omega_0      = first_omega_0
            )
        )
        # Hidden layers
        for i in range(hidden_layers):
            net.append(
                SineLinear(
                    in_features  = hidden_dim,
                    out_features = hidden_dim,
                    bias         = bias,
                    is_first     = False,
                    omega_0      = hidden_omega_0
                )
            )
        # Final layer
        final_linear = nn.Linear(hidden_dim, out_features, bias=bias)
        with torch.no_grad():
            final_linear.weight.uniform_(
                -np.sqrt(6.0 / hidden_dim) / hidden_omega_0,
                 np.sqrt(6.0 / hidden_dim) / hidden_omega_0
            )
        net.append(final_linear)

        self.net = nn.Sequential(*net)

    # --- Callable & Context Manager ---
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Forward the input through the network.

        Args:
            coords: Input tensor of shape (..., in_features) and values ranging
                from 0.0 to 1.0.

        Returns:
            Output tensor of shape (..., out_features) and values ranging
            from 0.0 to 1.0.
        """
        return self.net(coords)


class FINER(nn.Module):
    """FINER MLP.

    References:
        - Paper: "FINER: Flexible spectral-bias tuning in Implicit NEural
          Representation by Variable-periodic Activation Functions," CVPR 2024.
        - Code: https://github.com/liuzhen0212/FINER

    Attributes:
        net (torch.nn.Sequential): The MLP network.
    """

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        in_features     : int,
        out_features    : int,
        hidden_dim      : int,
        hidden_layers   : int,
        first_omega_0   : float = 30.0,
        hidden_omega_0  : float = 30.0,
        first_bias_scale: float | None = None,
        scale_req_grad  : bool  = False,
        bias            : bool  = True,
    ):
        """Initialize a new instance.

        Args:
            in_features: Size of each input sample.
            out_features: Size of each output sample.
            hidden_dim: Hidden channel dimensions.
            hidden_layers: Number of hidden layers.
            first_omega_0: Frequency scaling factor for the first layer.
                Defaults to 30.0.
            hidden_omega_0: Frequency scaling factor for the hidden layers.
                Defaults to 30.0.
            first_bias_scale: Bias scale for the first layer as float or None.
                Defaults to None.
            bias: If set to False, the layer will not learn an additive bias.
                Defaults to True.
            scale_req_grad: Scale requires gradient if True. Defaults to False.
        """
        super().__init__()

        # First layer
        net = []
        net.append(
            FINERLinear(
                in_features      = in_features,
                out_features     = hidden_dim,
                bias             = bias,
                is_first         = True,
                omega_0          = first_omega_0,
                first_bias_scale = first_bias_scale,
                scale_req_grad   = scale_req_grad
            )
        )
        # Hidden layers
        for i in range(hidden_layers):
            net.append(
                FINERLinear(
                    in_features    = hidden_dim,
                    out_features   = hidden_dim,
                    bias           = bias,
                    omega_0        = hidden_omega_0,
                    scale_req_grad = scale_req_grad
                )
            )
        # Final layer
        final_linear = nn.Linear(hidden_dim, out_features, bias=bias)
        with torch.no_grad():
            final_linear.weight.uniform_(
                -np.sqrt(6.0 / hidden_dim) / hidden_omega_0,
                 np.sqrt(6.0 / hidden_dim) / hidden_omega_0
            )
        net.append(final_linear)

        self.net = nn.Sequential(*net)

    # --- Callable & Context Manager ---
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Forward the input through the network.

        Args:
            coords: Input tensor of shape (..., in_features) and values ranging
                from 0.0 to 1.0.

        Returns:
            Output tensor of shape (..., out_features) and values ranging
            from 0.0 to 1.0.
        """
        return self.net(coords)


class FINER_PP(nn.Module):
    """FINER++ MLP.

    References:
        - Paper: "FINER++: Building a Family of Variable-periodic Functions for
          Activating Implicit Neural Representation," arXiv 2025.
        - Code: https://github.com/liuzhen0212/FINER

    Attributes:
        out_features (int): Size of each output sample.
        net (torch.nn.Sequential): The MLP network.
    """

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        in_features     : int,
        out_features    : int,
        hidden_dim      : int,
        hidden_layers   : int,
        first_omega_0   : float = 30.0,
        hidden_omega_0  : float = 30.0,
        first_bias_scale: float = 5,
        scale_req_grad  : bool  = False,
        bias            : bool  = True,
    ):
        """Initialize a new instance.

        Args:
            in_features: Size of each input sample.
            out_features: Size of each output sample.
            hidden_dim: Hidden channel dimensions.
            hidden_layers: Number of hidden layers.
            first_omega_0: Frequency scaling factor for the first layer.
                Defaults to 30.0.
            hidden_omega_0: Frequency scaling factor for the hidden layers.
                Defaults to 30.0.
            first_bias_scale: Bias scale for the first layer as float or None.
                Defaults to 5.
            scale_req_grad: Scale requires gradient if True. Defaults to False.
            bias: If set to False, the layer will not learn an additive bias.
                Defaults to True.
        """
        super().__init__()
        self.out_features = out_features

        # First layer
        net = []
        net.append(
            FINERLinear(
                in_features      = in_features,
                out_features     = hidden_dim,
                bias             = bias,
                is_first         = True,
                omega_0          = first_omega_0,
                first_bias_scale = first_bias_scale,
                scale_req_grad   = scale_req_grad
            )
        )
        # Hidden layers
        for i in range(hidden_layers):
            net.append(
                FINERLinear(
                    in_features    = hidden_dim,
                    out_features   = hidden_dim,
                    bias           = bias,
                    omega_0        = hidden_omega_0,
                    scale_req_grad = scale_req_grad
                )
            )
        # Final layer
        final_linear = nn.Linear(hidden_dim, out_features, bias=bias)
        with torch.no_grad():
            final_linear.weight.uniform_(
                -np.sqrt(6.0 / hidden_dim) / hidden_omega_0,
                 np.sqrt(6.0 / hidden_dim) / hidden_omega_0
            )
        net.append(final_linear)

        self.net = nn.Sequential(*net)

    # --- Callable & Context Manager ---
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Forward the input through the network.

        Args:
            coords: Input tensor of shape (..., in_features) and values ranging
                from 0.0 to 1.0.

        Returns:
            Output tensor of shape (..., out_features) and values ranging
            from 0.0 to 1.0.
        """
        output = self.net(coords)
        return output.view(-1, self.out_features)


# --- Radial & Wavelet Based ---

class GAUSS(nn.Module):
    """Gaussian MLP.

    References:
        - Code: https://github.com/liuzhen0212/FINER/blob/main/models.py

    Attributes:
        net (torch.nn.Sequential): The MLP network.
    """

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        in_features  : int,
        out_features : int,
        hidden_dim   : int,
        hidden_layers: int,
        scale        : float = 30.0,
        bias         : bool  = True,
    ):
        """Initialize a new instance.

        Args:
            in_features: Size of each input sample.
            out_features: Size of each output sample.
            hidden_dim: Number of hidden units in each hidden layer.
            hidden_layers: Number of hidden layers.
            scale: Gaussian scale factor. Defaults to 30.0.
            bias: If True, adds a learnable bias to the linear layers.
                Defaults to True.
        """
        super().__init__()

        # First layer
        net = []
        net.append(GaussLinear(in_features, hidden_dim, bias, scale=scale))
        # Hidden layers
        for i in range(hidden_layers):
            net.append(GaussLinear(hidden_dim, hidden_dim, bias, scale=scale))
        # Final layer
        final_linear = nn.Linear(hidden_dim, out_features, bias=bias)
        net.append(final_linear)

        self.net = nn.Sequential(*net)

    # --- Callable & Context Manager ---
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Forward the input through the network.

        Args:
            coords: Input tensor of shape (..., in_features) and values ranging
                from 0.0 to 1.0.

        Returns:
            Output tensor of shape (..., out_features) and values ranging
            from 0.0 to 1.0.
        """
        return self.net(coords)


class WIRE(nn.Module):
    """WIRE MLP with Gabor wavelet activations.

    References:
        - Paper: "WIRE: Wavelet Implicit Neural Representations," CVPR 2023.
        - Code: https://github.com/vishwa91/wire
        - Code: https://github.com/liuzhen0212/FINER/blob/main/models.py

    Attributes:
        nonlin (type): The nonlinearity class used.
        complex (bool): Whether the network uses complex numbers.
        wavelet (str): The type of wavelet used.
        pos_encode (bool): Whether positional encoding is used.
        net (torch.nn.Sequential): The MLP network.
    """

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        in_features   : int,
        out_features  : int,
        hidden_dim    : int,
        hidden_layers : int   = 4,
        first_omega_0 : float = 10.0,
        hidden_omega_0: float = 10.0,
        scale         : float = 10.0,
        bias          : bool  = True,
    ):
        """Initialize a new instance.

        Args:
            in_features: Size of each input sample.
            out_features: Size of each output sample.
            hidden_dim: Size of each hidden layer.
            hidden_layers: Number of hidden layers. Defaults to 4.
            first_omega_0: Frequency scaling factor for the first layer.
                Defaults to 10.0.
            hidden_omega_0: Frequency scaling factor for hidden layers.
                Defaults to 10.0.
            scale: Scaling of Gabor Gaussian term. Defaults to 10.0.
            bias: If False, the layers will not learn an additive bias.
                Defaults to True.
        """
        super().__init__()
        # All results in the paper were with the default complex 'gabor' nonlinearity
        self.nonlin  = ComplexGaborLayer

        # Since complex numbers are two real numbers, reduce the number of hidden parameters by 2
        hidden_dim   = int(hidden_dim / np.sqrt(2))
        dtype        = torch.cfloat
        self.complex = True
        self.wavelet = "gabor"

        # Legacy parameter
        self.pos_encode = False

        # First layer
        net = []
        net.append(
            self.nonlin(
                in_features  = in_features,
                out_features = hidden_dim,
                bias         = bias,
                is_first     = True,
                omega_0      = first_omega_0,
                sigma_0      = scale,
                trainable    = False
            )
        )
        # Hidden layers
        for i in range(hidden_layers):
            net.append(
                self.nonlin(
                    in_features  = hidden_dim,
                    out_features = hidden_dim,
                    bias         = bias,
                    is_first     = False,
                    omega_0      = hidden_omega_0,
                    sigma_0      = scale
                )
            )
        # Final layer
        final_linear = nn.Linear(hidden_dim, out_features, bias=bias, dtype=dtype)
        net.append(final_linear)

        self.net = nn.Sequential(*net)

    # --- Callable & Context Manager ---
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Forward the input through the network.

        Args:
            coords: Input tensor of shape (..., in_features) and values ranging
                from 0.0 to 1.0.

        Returns:
            Output tensor of shape (..., out_features) and values ranging
            from 0.0 to 1.0.
        """
        output = self.net(coords)
        if self.wavelet == "gabor":
            return output.real
        return output


# ==============================================================================
# UTILITIES
# ==============================================================================

# --- Coordinate Generation & Embedding ---

def create_coords(size: int | tuple[int, ...], device: torch.device) -> torch.Tensor:
    """Create a normalized square coordinates grid.

    Args:
        size: The size of the grid.
        device: The device to place the tensor on.

    Returns:
        A tensor of shape (1, size, size, 2) and values ranging from -1.0 to 1.0.
    """
    h, w    = I.imgsz(size)
    # TODO: Old code normalize from 0 to 1. Delete later
    # x_range = torch.linspace(0, 1, w, device=device)
    # y_range = torch.linspace(0, 1, h, device=device)
    x_range = torch.linspace(-1, 1, w, device=device)
    y_range = torch.linspace(-1, 1, h, device=device)
    y, x    = torch.meshgrid(y_range, x_range, indexing="ij")
    # Stack to get (H, W, 2) then expand to (1, H, W, 2)
    coords  = torch.stack([x, y], dim=-1).unsqueeze(0)
    return coords


def create_noisy_coords(size: int, sigma: float = 0.5, lamda: float = 1.0) -> torch.Tensor:
    """Create a normalized square coordinates grid with Gaussian noise.

    Args:
        size: The size of the grid.
        sigma: Standard deviation of the Gaussian noise. Defaults to 0.5.
        lamda: Lambda parameter for Poisson noise. Defaults to 1.0

    Returns:
        A tensor of shape (size, size, 2) and values ranging from 0.0 to 1.0.
    """
    h, w   = size, size
    coords = np.dstack(np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w)))
    coords = torch.from_numpy(coords).float()

    # Add Gaussian noise
    gaussian_noise = torch.normal(mean=0.0, std=sigma, size=coords.shape).to(coords.device)
    noisy_coords   = coords + gaussian_noise

    # Add Poisson noise
    poisson_noise  = torch.poisson(torch.full(coords.shape, lamda)).to(coords.device) - lamda
    noisy_coords   = noisy_coords + poisson_noise * 0.1  # Scale noise

    # Clip to ensure coordinates stay within [0, 1]
    noisy_coords = torch.clamp(noisy_coords, 0.0, 1.0)

    return noisy_coords


def ff_embedding(p: torch.Tensor, B: torch.Tensor | None = None) -> torch.Tensor:
    """Apply Fourier feature embedding to input tensor.

    Args:
        p: Input tensor of shape (..., D) and values ranging from 0.0 to 1.0.
        B: Frequency matrix of shape (F, D). Default to None means no embedding.

    Returns:
        Embedded tensor of shape (..., 2 * F) if B is provided, otherwise
        returns the original tensor p.
    """
    if B is None:
        return p
    else:
        x_proj    = (2 * np.pi * p) @ B.T
        embedding = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return embedding


# --- Spatial Context & Patch Extraction ---

def create_patches(image: torch.Tensor, kernel_size: int = 7) -> torch.Tensor:
    """Create a tensor where the channel contains patch information.

    Args:
        image: Image, formatted as a torch.Tensor of shape (1, C, H, W)
            and pixel values ranging from 0.0 to 1.0.
        kernel_size: Size of square patches. Defaults to 7.

    Returns:
        A tensor of shape (1, H', W', K^2) where H' and W' are the height
        and width after patch extraction, and K is the kernel_size.

    Raises:
        ValueError: If the input ``image`` does not have 4 dimensions.
    """
    if image.ndim != 4:
        raise ValueError(f"Expected 'image' to be a 4D tensor, but got {image.ndim}D.")

    b, c, h, w = image.shape
    kernel     = torch.zeros((kernel_size ** 2, c, kernel_size, kernel_size)).to(image.device)
    for i in range(kernel_size):
        for j in range(kernel_size):
            kernel[i + j * kernel_size, :, i, j] = 1

    pad          = nn.ReflectionPad2d(kernel_size // 2)
    image_padded = pad(image)
    patches      = F.conv2d(image_padded, kernel, padding=0).squeeze(0)
    return torch.movedim(patches, 0, -1)


def create_depth_aware_patches(
    image      : torch.Tensor,
    depth      : torch.Tensor,
    kernel_size: int   = 7,
    alpha      : float = 8.3
) -> torch.Tensor:
    """Create depth-aware patches for the given image and depth map.

    Args:
        image: Image, formatted as a torch.Tensor of shape (1, C, H, W)
            and pixel values ranging from 0.0 to 1.0.
        depth: Depth, formatted as a torch.Tensor of shape (1, 1, H, W)
            and pixel values ranging from 0.0 to 1.0.
        kernel_size: Size of square patches. Defaults to 7.
        alpha: Depth sensitivity parameter. Defaults to 8.3.

    Returns:
        A tensor of shape (1, H', W', K^2) where H' and W' are the height
        and width after patch extraction, and K is the kernel_size.
    """
    b, c, h, w = image.shape
    kernel = torch.zeros((kernel_size ** 2, c, kernel_size, kernel_size)).to(image.device)
    for i in range(kernel_size):
        for j in range(kernel_size):
            kernel[i + j * kernel_size, 0, i, j] = 1

    pad           = nn.ReflectionPad2d(kernel_size // 2)
    image_padded  = pad(image)
    image_patches = F.conv2d(image_padded, kernel, padding=0).squeeze(0)
    depth_padded  = pad(depth)
    depth_patches = F.conv2d(depth_padded, kernel, padding=0).squeeze(0)

    # Compute center index in patch
    center_idx   = (kernel_size ** 2) // 2
    depth_center = depth_patches[center_idx, :, :].unsqueeze(0).repeat(kernel_size ** 2, 1, 1)

    # FD = exp(-alpha * |depth_center - depth_neighbor|)
    depth_diff = torch.abs(depth_center - depth_patches)
    fd         = torch.exp(-alpha * depth_diff)  # Shape for multiplication

    # Weight the image patches and normalize
    patches     = image_patches * fd
    weights_sum = fd.sum(dim=0, keepdim=True) + 1e-6  # Avoid division by zero
    patches     = patches / weights_sum

    return torch.movedim(patches, 0, -1)


# --- Multi-scale & Sampling Operations ---

def pair_downsampler(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Downsample the image into two sub-images using learned filters.

    Args:
        image: Image, formatted as a torch.Tensor of shape (1, C, H, W)
            and pixel values ranging from 0.0 to 1.0.

    Returns:
        Two downsampled images, formatted as a torch.Tensor of shape
        (B, C, H/2, W/2) and pixel values ranging from 0.0 to 1.0.
    """
    c       = image.shape[1]
    filter1 = torch.FloatTensor([[[[0, 0.5], [0.5, 0]]]]).to(image.device)
    filter1 = filter1.repeat(c, 1, 1, 1)
    filter2 = torch.FloatTensor([[[[0.5, 0], [0, 0.5]]]]).to(image.device)
    filter2 = filter2.repeat(c, 1, 1, 1)
    output1 = F.conv2d(image, filter1, stride=2, groups=c)
    output2 = F.conv2d(image, filter2, stride=2, groups=c)
    return output1, output2


def interpolate_image(image: torch.Tensor, size: int) -> torch.Tensor:
    """Resize the image to the specified size.

    Args:
        image: Image, formatted as a torch.Tensor of shape (1, C, H, W)
            and pixel values ranging from 0.0 to 1.0.
        size: The target size for both height and width.

    Returns:
        Resized image, formatted as a torch.Tensor of shape (B, C, size, size)
        and pixel values ranging from 0.0 to 1.0.
    """
    # return F.interpolate(image, size=(down_size, down_size), mode="bicubic")
    return F.interpolate(image, size=(size, size), mode="area")


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Modules.

This module provides various layers, blocks, and modules.
"""

from __future__ import annotations

__all__ = [
    "Conv2dTime",
    "Decoder",
    "DenoiseNet",
    "Encoder",
    "EncoderTime",
    "EnhanceFunction",
    "EnhanceFunctionTime",
    "ODEBlock",
]

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torchdiffeq import odeint_adjoint

from .loss import L_tv
from .utils import weights_init


# ==============================================================================
# region MODULES
# ==============================================================================

# --- Denoise ---

class DenoiseNet(nn.Module):

    # --- Lifecycle & Initialization ---
    def __init__(self, in_channels: int = 3, hidden_dim: int = 48):
        """Initialize a new instance.

        Args:
            in_channels (int, optional): Number of input channels. Defaults to 3.
            hidden_dim (int, optional): Number of hidden channels. Defaults to 48.
        """
        super().__init__()
        # Assign attributes
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.out_channels = in_channels

        self.conv1 = nn.Conv2d(self.in_channels, self.hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(self.hidden_dim, self.out_channels, kernel_size=1)
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    # --- Callable & Context Manager ---
    def forward(self, x: Tensor) -> Tensor:
        """Forward the input through the network.

        Args:
            x (Tensor): Input tensor of shape (B, C, H, W) and values ranging
                from 0.0 to 1.0.

        Returns:
            Tensor: Output tensor of shape (B, C, H, W) and values ranging
                from 0.0 to 1.0.
        """
        y = self.act(self.conv1(x))
        y = self.act(self.conv2(y))
        y = self.conv3(y)
        return y


# --- Vanilla ---

class Encoder(nn.Module):

    # --- Lifecycle & Initialization ---
    def __init__(self, in_channels: int, hidden_dim: int = 32):
        """Initialize a new instance.

        Args:
            in_channels (int): Number of input channels.
            hidden_dim (int, optional): Number of hidden channels. Defaults to 32.
        """
        super().__init__()
        # Assign attributes
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.out_channels = hidden_dim

        # Define network
        self.conv_1 = nn.Conv2d(self.in_channels, self.hidden_dim, 3, 1, padding=1, padding_mode="reflect")
        self.conv_3_1 = nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, 1, padding=3//2, padding_mode="reflect")
        self.conv_5_1 = nn.Conv2d(self.hidden_dim, self.hidden_dim, 5, 1, padding=5//2, padding_mode="reflect")
        self.conv_3_2 = nn.Conv2d(self.hidden_dim * 2, self.hidden_dim * 2, 3, 1, padding=3//2, padding_mode="reflect")
        self.conv_5_2 = nn.Conv2d(self.hidden_dim * 2, self.hidden_dim * 2, 5, 1, padding=5//2, padding_mode="reflect")
        self.confusion = nn.Conv2d(self.hidden_dim * 4, self.hidden_dim, 1, 1, padding=0)
        self.act = nn.ReLU(inplace=True)
        self.norm_1 = nn.GroupNorm(1, self.hidden_dim)
        self.norm_2 = nn.GroupNorm(1, self.hidden_dim * 2)

        self.apply(weights_init)

    # --- Callable & Context Manager ---
    def forward(self, x: Tensor) -> Tensor:
        """Forward the input through the network.

        Args:
            x (Tensor): Input tensor of shape (B, C, H, W) and values ranging
                from 0.0 to 1.0.

        Returns:
            Tensor: Output tensor of shape (B, hidden_dim, H, W) and values
                ranging from 0.0 to 1.0.
        """
        x_1 = self.act(self.norm_1(self.conv_1(x)))
        x_3_1 = self.act(self.norm_1(self.conv_3_1(x_1)))
        x_5_1 = self.act(self.norm_1(self.conv_5_1(x_3_1)))
        x_2 = torch.cat([x_3_1, x_5_1], dim=1)
        x_3_2 = self.act(self.norm_2(self.conv_3_2(x_2)))
        x_5_2 = self.act(self.norm_2(self.conv_5_2(x_3_2)))
        x_3 = torch.cat([x_3_2, x_5_2], dim=1)
        y = self.act(self.norm_1(self.confusion(x_3)))
        return y


class Decoder(nn.Module):

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_dim: int = 32,
    ):
        """Initialize a new instance.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            hidden_dim (int, optional): Number of hidden channels. Defaults to 32.
        """
        super().__init__()
        # Assign attributes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_dim = hidden_dim

        # Define network
        self.linear_1 = nn.Linear(self.in_channels, self.hidden_dim)
        self.linear_2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.linear_3 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.linear_4 = nn.Linear(self.hidden_dim, self.out_channels)
        self.act = nn.ReLU(inplace=True)

    # --- Callable & Context Manager ---
    def forward(self, feat: Tensor, coords: Tensor) -> Tensor:
        """Forward the input through the network.

        Args:
            feat (Tensor): Input feature tensor of shape (B, N, hidden_dim) and
                values ranging from 0.0 to 1.0.
            coords (Tensor): Input coordinate tensor of shape (B, N, 2) and
                values ranging from 0.0 to 1.0.

        Returns:
            Tensor: Output tensor of shape (B, N, out_channels) and values
                ranging from 0.0 to 1.0.
        """
        x = torch.cat([feat, coords], dim=-1)
        y = self.act(self.linear_1(x))
        y = self.act(self.linear_2(y))
        y = self.act(self.linear_3(y))
        y = F.tanh(self.linear_4(y))
        return y


class EnhanceFunction(nn.Module):
    """A module for enhancing the input image."""

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_dim: int = 32,
        imgsz: int = 512,
        chunk_size: int = 100000,
        num_iter: int = 8,
        *args, **kwargs
    ):
        """Initialize a new instance.

        Args:
            in_channels (int, optional): Number of input channels.
                Defaults to 3.
            out_channels (int, optional): Number of output channels.
                Defaults to 3.
            hidden_dim (int, optional): Hidden dimension. Defaults to 32.
            imgsz (int, optional): Downsample the input image to this size for
                encoding. Defaults to 512.
            chunk_size (int): Number of pixels to process at once. Defaults to 100,000.
            num_iter (int, optional): Number of iterations for curve estimation.
                Defaults to 8.
        """
        super().__init__()
        # Assign attributes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_dim = hidden_dim
        self.imgsz = imgsz
        self.chunk_size = chunk_size
        self.num_iter = num_iter

        # Define network
        # Denoising Module
        self.denoise = DenoiseNet(in_channels=self.in_channels)
        # Encoder
        self.encode = Encoder(in_channels=self.in_channels * 2 + 1, hidden_dim=self.hidden_dim)
        # Implicit Refiner (Siren/Continuous MLP)
        # Input: Features (32) + Coordinates (2) = 34
        self.decode = Decoder(
            in_channels=self.hidden_dim + 2,
            out_channels=self.out_channels * self.num_iter,
            hidden_dim=self.hidden_dim,
        )

    # --- Callable & Context Manager ---
    def forward(self, image: Tensor, depth: Tensor | None = None) -> dict:
        """Forward the input through the network.

        Args:
            image (Tensor): Image tensor of shape (B, 3, H, W) and values
                ranging from 0.0 to 1.0.
            depth (Tensor, optional): Depth tensor of shape (B, 1, H, W) and
                values ranging from 0.0 to 1.0. Defaults to None.
        """
        # 1. Pre-process
        x = image
        d = depth
        b, c, h, w = x.shape
        size = (self.imgsz, self.imgsz)

        # We downsample the input to 512x512 so the CNN doesn't cause an OOM error
        if (h, w) != size:
            x = F.interpolate(x, size=size, mode="bilinear", align_corners=True)
            d = F.interpolate(d, size=size, mode="bilinear", align_corners=True) if d is not None else None

        # 2. Denoise
        l_denoise = self.denoise_loss(x)
        p_x = x - self.denoise(x)

        # 3. Fusion
        if d is not None:
            x_in = torch.cat([x, p_x, d], dim=1)
        else:
            x_in = torch.cat([x, p_x], dim=1)

        # 4. Encode
        feat = self.encode(x_in)

        # 5. Predict curve parameters
        if (h, w) == size:
            A = self.predict_curve_map(feat, self.imgsz, self.imgsz)
        else:
            A = self.predict_curve_map_chunk(feat, h, w)

        # 6. Enhance
        y = self.enhance(image, A)

        # 7. Return final and intermediate results for debugging
        outputs = {
            "enhanced": y,
            "denoised": p_x,
            "A": A,
            "l_denoise": l_denoise,
        }
        return outputs

    # --- Denoise ---
    def denoise_loss(self, noisy_image: Tensor) -> Tensor:
        """Calculate the ZS-N2N denoising loss."""
        mse = nn.MSELoss()

        noisy1, noisy2 = self.pair_downsampler(noisy_image)
        pred1 = noisy1 - self.denoise(noisy1)
        pred2 = noisy2 - self.denoise(noisy2)
        loss_res = 0.5 * (mse(noisy1, pred2) + mse(noisy2, pred1))

        noisy_denoised = noisy_image - self.denoise(noisy_image)
        denoised1, denoised2 = self.pair_downsampler(noisy_denoised)
        loss_cons = 0.5 * (mse(pred1, denoised1) + mse(pred2, denoised2))
        loss = loss_res + loss_cons
        return loss

    def add_noise(self, x: Tensor, noise_level: float) -> Tensor:
        """Add noise to the image."""
        noisy = x + torch.normal(0, noise_level / 255, x.shape)
        noisy = torch.clamp(noisy, 0, 1)
        noisy = noisy.to(x.device)
        return noisy

    def pair_downsampler(self, image: Tensor) -> tuple[Tensor, Tensor]:
        """Downsample an image tensor into a pair to half resolution.

        References:
            - Code: https://colab.research.google.com/drive/1i82nyizTdszyHkaHBuKPbWnTzao8HF9b?usp=sharing

        Args:
            image (Tensor): Image tensor of shape (B, C, H, W) and values ranging
                from 0.0 to 1.0.

        Returns:
            tuple[Tensor, Tensor]: Downsampled images of shape (B, C, H/2, W/2).

        Raises:
            TypeError: If ``image`` is not a 4D torch.Tensor.

        Notes:
            Averages diagonal pixels in non-overlapping patches:
                -------------      -------------
                | A1 | B1 | A2 | B2 |      | A1+D1/2 | A2+D2/2 |
                | C1 | D1 | C2 | D2 |      | A3+D3/2 | A4+D4/2 |
                -------------  =>  -------------
                | A3 | B3 | A4 | B4 |      | B1+C1/2 | B2+C2/2 |
                | C3 | D3 | C4 | D4 |      | B3+C3/2 | B4+C4/2 |
                -------------      -------------
        """
        b, c, h, w  = image.shape
        device, dtype = image.device, image.dtype

        # Define kernels: filter_ad picks (top-left, bottom-right), filter_bc picks (top-right, bottom-left)
        # We use .repeat(c, 1, 1, 1) for channel-wise (depthwise) convolution
        kernel_ad = torch.tensor([[[[0.5, 0.0], [0.0, 0.5]]]], device=device, dtype=dtype)
        kernel_ad = kernel_ad.repeat(c, 1, 1, 1)
        kernel_bc = torch.tensor([[[[0.0, 0.5], [0.5, 0.0]]]], device=device, dtype=dtype)
        kernel_bc = kernel_bc.repeat(c, 1, 1, 1)

        # Stride=2 ensures non-overlapping 2x2 patches
        out_ad = F.conv2d(image, kernel_ad, stride=2, groups=c)
        out_bc = F.conv2d(image, kernel_bc, stride=2, groups=c)
        return out_ad, out_bc

    # --- Curve Map ---
    def predict_curve_map(self, feat: Tensor, h: int, w: int) -> Tensor:
        b = feat.shape[0]
        device = feat.device

        # We map the massive target resolution to the [-1, 1] continuous space.
        h_coords = torch.linspace(-1, 1, steps=h, device=device)
        w_coords = torch.linspace(-1, 1, steps=w, device=device)
        grid_h, grid_w = torch.meshgrid(h_coords, w_coords, indexing="ij")
        coords = torch.stack([grid_w, grid_h], dim=-1).view(1, -1, 2).repeat(b, 1, 1)  # [B, H*W, 2]

        sampled_feat = F.grid_sample(feat, coords.unsqueeze(1), mode="bilinear", align_corners=True)
        sampled_feat = sampled_feat.squeeze(2).permute(0, 2, 1)  # [B, N, 32]

        A = self.decode(sampled_feat, coords)
        A = A.view(b, h, w, 3 * self.num_iter).permute(0, 3, 1, 2)

        return A

    def predict_curve_map_chunk(self, feat: Tensor, h: int, w: int) -> Tensor:
        chunk_size = self.chunk_size
        b = feat.shape[0]
        device = feat.device

        # We map the massive target resolution to the [-1, 1] continuous space.
        h_coords = torch.linspace(-1, 1, steps=h, device=device)
        w_coords = torch.linspace(-1, 1, steps=w, device=device)
        grid_h, grid_w = torch.meshgrid(h_coords, w_coords, indexing="ij")
        coords = torch.stack([grid_w, grid_h], dim=-1).view(b, -1, 2)  # [B, H*W, 2]

        total_points = h * w
        A_list = []

        # 1 Chunked MLP Inference
        # Process the points in batches of `chunk_size` to cap VRAM usage.
        for i in range(0, total_points, chunk_size):
            coords_chunk = coords[:, i:i+chunk_size, :]  # [B, chunk, 2]

            # Sample from the 512x512 feature map at the exact target coordinates
            sampled_feat = F.grid_sample(feat, coords_chunk.unsqueeze(1), mode="bilinear", align_corners=True)
            sampled_feat = sampled_feat.squeeze(2).permute(0, 2, 1)  # [B, chunk, 32]

            # Predict the curve parameters for this chunk
            A_chunk = self.decode(sampled_feat, coords_chunk)
            A_list.append(A_chunk)

        # 2. Reconstruct the spatial curve parameter map
        A_flat = torch.cat(A_list, dim=1)  # [B, H*W, 24]
        A = A_flat.view(b, h, w, 3 * self.num_iter).permute(0, 3, 1, 2)  # [B, 24, H, W]

        return A

    # --- Enhance ---
    def enhance(self, image: Tensor, A: Tensor) -> Tensor:
        """Apply the iterative enhancement."""
        y = image
        for i in range(self.num_iter):
            A_i = A[:, i*3:(i+1)*3, :, :]
            y = y + A_i * (torch.pow(y, 2) - y)
        return y


# --- ODE ---

class Conv2dTime(nn.Conv2d):
    """2D convolutional layer that takes in the time step as an additional input.
    """

    # --- Lifecycle & Initialization ---
    def __init__(self, in_channels: int, *args, **kwargs):
        """Initialize a new instance.

        Args:
            in_channels (int): Number of channels in the input image (excluding
                the time channel).
        """
        super().__init__(in_channels + 1, *args, **kwargs)

    # --- Callable & Context Manager ---
    def forward(self, t: Tensor, x: Tensor) -> Tensor:
        """Forward the input through the network.

        Args:
            t (Tensor): Time step tensor of shape (B,) or a scalar.
            x (Tensor): Input image tensor of shape (B, C, H, W) and values
                ranging from 0.0 to 1.0.
        """
        t_img = torch.ones_like(x[:, :1, :, :]) * t  # (B, 1, H, W)
        t_and_x = torch.cat([t_img, x], 1)  # (B, C + 1, H, W)
        return super(Conv2dTime, self).forward(t_and_x)


class EncoderTime(nn.Module):

    # --- Lifecycle & Initialization ---
    def __init__(self, in_channels: int, hidden_dim: int = 32):
        """Initialize a new instance.

        Args:
            in_channels (int): Number of input channels.
            hidden_dim (int, optional): Number of hidden channels. Defaults to 32.
        """
        super().__init__()
        # Assign attributes
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.out_channels = hidden_dim

        # Define network
        self.conv_1 = Conv2dTime(self.in_channels, self.hidden_dim, 3, 1, padding=1, padding_mode="reflect")
        self.conv_3_1 = Conv2dTime(self.hidden_dim, self.hidden_dim, 3, 1, padding=3//2, padding_mode="reflect")
        self.conv_5_1 = Conv2dTime(self.hidden_dim, self.hidden_dim, 5, 1, padding=5//2, padding_mode="reflect")
        self.conv_3_2 = Conv2dTime(self.hidden_dim * 2, self.hidden_dim * 2, 3, 1, padding=3//2, padding_mode="reflect")
        self.conv_5_2 = Conv2dTime(self.hidden_dim * 2, self.hidden_dim * 2, 5, 1, padding=5//2, padding_mode="reflect")
        self.confusion = Conv2dTime(self.hidden_dim * 4, self.hidden_dim, 1, 1, padding=0)
        self.act = nn.ReLU(inplace=True)
        self.norm_1 = nn.GroupNorm(1, self.hidden_dim)
        self.norm_2 = nn.GroupNorm(1, self.hidden_dim * 2)

        self.apply(weights_init)

    # --- Callable & Context Manager ---
    def forward(self, t: Tensor, x: Tensor) -> Tensor:
        """Forward the input through the network.

        Args:
            t (Tensor): Time step tensor of shape (B,) or a scalar.
            x (Tensor): Input tensor of shape (B, C, H, W) and values ranging
                from 0.0 to 1.0.

        Returns:
            Tensor: Output tensor of shape (B, hidden_dim, H, W) and values
                ranging from 0.0 to 1.0.
        """
        x_1 = self.act(self.norm_1(self.conv_1(t, x)))
        x_3_1 = self.act(self.norm_1(self.conv_3_1(t, x_1)))
        x_5_1 = self.act(self.norm_1(self.conv_5_1(t, x_3_1)))
        x_2 = torch.cat([x_3_1, x_5_1], dim=1)
        x_3_2 = self.act(self.norm_2(self.conv_3_2(t, x_2)))
        x_5_2 = self.act(self.norm_2(self.conv_5_2(t, x_3_2)))
        x_3 = torch.cat([x_3_2, x_5_2], dim=1)
        y = self.act(self.norm_1(self.confusion(t, x_3)))
        return y


class EnhanceFunctionTime(nn.Module):
    """A module for enhancing the input image with time conditioning."""

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_dim: int = 32,
        imgsz: int = 512,
        chunk_size: int = 100000,
        *args, **kwargs
    ):
        """Initialize a new instance.

        Args:
            in_channels (int, optional): Number of input channels.
                Defaults to 3.
            out_channels (int, optional): Number of output channels.
                Defaults to 3.
            hidden_dim (int, optional): Hidden dimension. Defaults to 32.
            imgsz (int, optional): Downsample the input image to this size for
                encoding. Defaults to 512.
            chunk_size (int): Number of pixels to process at once. Defaults to 100,000.
        """
        super().__init__()
        # Assign attributes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_dim = hidden_dim
        self.imgsz = imgsz
        self.chunk_size = chunk_size

        # Define network
        # Denoising Module
        self.denoise = DenoiseNet(in_channels=self.in_channels)
        # Encoder
        self.encode = EncoderTime(in_channels=self.in_channels * 2 + 1, hidden_dim=self.hidden_dim)
        # Implicit Refiner (Siren/Continuous MLP)
        # Input: Features (32) + Coordinates (2) = 34
        self.decode = Decoder(
            in_channels=self.hidden_dim + 2,
            out_channels=self.out_channels,
            hidden_dim=self.hidden_dim,
        )

        # Allocate resources
        self.nfe = 0
        self.pred_t = []
        self.last_A = None
        self.tv_loss = L_tv()

    # --- Callable & Context Manager ---
    def forward(self, t: Tensor, x: Tensor) -> Tensor:
        """Forward the input through the network.

        Args:
            t (Tensor): Time step tensor of shape (B,) or a scalar.
            x (Tensor): Input tensor of shape (B, C, H, W) and values ranging
                from 0.0 to 1.0.
        """
        # 1. Pre-process
        self.nfe += 1

        image = _x = x[:, :3 , :, :]  # Image
        depth = _d = x[:, 3:4, :, :]  # Depth map
        b, c, h, w = x.shape
        size = (self.imgsz, self.imgsz)

        # We downsample the input to 512x512 so the CNN doesn't cause an OOM error
        if (h, w) != size:
            _x = F.interpolate(_x, size=size, mode="bilinear", align_corners=True)
            _d = F.interpolate(_d, size=size, mode="bilinear", align_corners=True) if _d is not None else None

        # 2. Denoise
        l_denoise = self.denoise_loss(_x)
        p_x = _x - self.denoise(_x)

        # 3. Fusion
        if _d is not None:
            _in = torch.cat([_x, p_x, _d], dim=1)
        else:
            _in = torch.cat([_x, p_x], dim=1)

        # 4. Encode
        feat = self.encode(t, _in)

        # 5. Predict curve parameters
        if (h, w) == size:
            A = self.predict_curve_map(feat, self.imgsz, self.imgsz)
        else:
            A = self.predict_curve_map_chunk(feat, h, w)

        # 6. Enhance
        y = A * (torch.pow(image, 2) - image)

        # 7. Return final and intermediate results for debugging
        self.last_A = A
        self.pred_t.append(t.item())

        # Since ODE solvers typically expect the output to be the same shape as
        # the input, we concatenate the intermediate results along the channel
        # dimension for debugging purposes. The final output will still be `y`,
        # which is the enhanced image.
        # l_tv = torch.ones_like(A) * self.tv_loss(A, depth)
        l_denoise = torch.ones_like(A) * l_denoise

        # Debug
        # print(self.nfe)

        outputs = torch.cat([y, depth, l_denoise], dim=1)
        return outputs

    # --- Denoise ---
    def denoise_loss(self, noisy_image: Tensor) -> Tensor:
        """Calculate the ZS-N2N denoising loss."""
        mse = nn.MSELoss()

        noisy1, noisy2 = self.pair_downsampler(noisy_image)
        pred1 = noisy1 - self.denoise(noisy1)
        pred2 = noisy2 - self.denoise(noisy2)
        loss_res = 0.5 * (mse(noisy1, pred2) + mse(noisy2, pred1))

        noisy_denoised = noisy_image - self.denoise(noisy_image)
        denoised1, denoised2 = self.pair_downsampler(noisy_denoised)
        loss_cons = 0.5 * (mse(pred1, denoised1) + mse(pred2, denoised2))
        loss = loss_res + loss_cons
        return loss

    def add_noise(self, x: Tensor, noise_level: float) -> Tensor:
        """Add noise to the image."""
        noisy = x + torch.normal(0, noise_level / 255, x.shape)
        noisy = torch.clamp(noisy, 0, 1)
        noisy = noisy.to(x.device)
        return noisy

    def pair_downsampler(self, image: Tensor) -> tuple[Tensor, Tensor]:
        """Downsample an image tensor into a pair to half resolution.

        References:
            - Code: https://colab.research.google.com/drive/1i82nyizTdszyHkaHBuKPbWnTzao8HF9b?usp=sharing

        Args:
            image (Tensor): Image tensor of shape (B, C, H, W) and values ranging
                from 0.0 to 1.0.

        Returns:
            tuple[Tensor, Tensor]: Downsampled images of shape (B, C, H/2, W/2).

        Raises:
            TypeError: If ``image`` is not a 4D torch.Tensor.

        Notes:
            Averages diagonal pixels in non-overlapping patches:
                -------------      -------------
                | A1 | B1 | A2 | B2 |      | A1+D1/2 | A2+D2/2 |
                | C1 | D1 | C2 | D2 |      | A3+D3/2 | A4+D4/2 |
                -------------  =>  -------------
                | A3 | B3 | A4 | B4 |      | B1+C1/2 | B2+C2/2 |
                | C3 | D3 | C4 | D4 |      | B3+C3/2 | B4+C4/2 |
                -------------      -------------
        """
        b, c, h, w  = image.shape
        device, dtype = image.device, image.dtype

        # Define kernels: filter_ad picks (top-left, bottom-right), filter_bc picks (top-right, bottom-left)
        # We use .repeat(c, 1, 1, 1) for channel-wise (depthwise) convolution
        kernel_ad = torch.tensor([[[[0.5, 0.0], [0.0, 0.5]]]], device=device, dtype=dtype)
        kernel_ad = kernel_ad.repeat(c, 1, 1, 1)
        kernel_bc = torch.tensor([[[[0.0, 0.5], [0.5, 0.0]]]], device=device, dtype=dtype)
        kernel_bc = kernel_bc.repeat(c, 1, 1, 1)

        # Stride=2 ensures non-overlapping 2x2 patches
        out_ad = F.conv2d(image, kernel_ad, stride=2, groups=c)
        out_bc = F.conv2d(image, kernel_bc, stride=2, groups=c)
        return out_ad, out_bc

    # --- Curve Map ---
    def predict_curve_map(self, feat: Tensor, h: int, w: int) -> Tensor:
        b = feat.shape[0]
        device = feat.device

        # We map the massive target resolution to the [-1, 1] continuous space.
        h_coords = torch.linspace(-1, 1, steps=h, device=device)
        w_coords = torch.linspace(-1, 1, steps=w, device=device)
        grid_h, grid_w = torch.meshgrid(h_coords, w_coords, indexing="ij")
        coords = torch.stack([grid_w, grid_h], dim=-1).view(1, -1, 2).repeat(b, 1, 1)  # [B, H*W, 2]

        sampled_feat = F.grid_sample(feat, coords.unsqueeze(1), mode="bilinear", align_corners=True)
        sampled_feat = sampled_feat.squeeze(2).permute(0, 2, 1)  # [B, N, 32]

        A = self.decode(sampled_feat, coords)
        A = A.view(b, h, w, 3).permute(0, 3, 1, 2)

        return A

    def predict_curve_map_chunk(self, feat: Tensor, h: int, w: int) -> Tensor:
        chunk_size = self.chunk_size
        b = feat.shape[0]
        device = feat.device

        # We map the massive target resolution to the [-1, 1] continuous space.
        h_coords = torch.linspace(-1, 1, steps=h, device=device)
        w_coords = torch.linspace(-1, 1, steps=w, device=device)
        grid_h, grid_w = torch.meshgrid(h_coords, w_coords, indexing="ij")
        coords = torch.stack([grid_w, grid_h], dim=-1).view(b, -1, 2)  # [B, H*W, 2]

        total_points = h * w
        A_list = []

        # 1 Chunked MLP Inference
        # Process the points in batches of `chunk_size` to cap VRAM usage.
        for i in range(0, total_points, chunk_size):
            coords_chunk = coords[:, i:i+chunk_size, :]  # [B, chunk, 2]

            # Sample from the 512x512 feature map at the exact target coordinates
            sampled_feat = F.grid_sample(feat, coords_chunk.unsqueeze(1), mode="bilinear", align_corners=True)
            sampled_feat = sampled_feat.squeeze(2).permute(0, 2, 1)  # [B, chunk, 32]

            # Predict the curve parameters for this chunk
            A_chunk = self.decode(sampled_feat, coords_chunk)
            A_list.append(A_chunk)

        # 2. Reconstruct the spatial curve parameter map
        A_flat = torch.cat(A_list, dim=1)  # [B, H*W, 24]
        A = A_flat.view(b, h, w, 3).permute(0, 3, 1, 2)  # [B, 24, H, W]

        return A


class ODEBlock(nn.Module):

    max_num_steps = 100  # 30 # 50 # 100 # 1000

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        ode_func: nn.Module,
        tol: float = 1e-3,
        adjoint: bool = True,
        *args, **kwargs
    ):
        """Initialize a new instance.

        Args:
            ode_func (nn.Module): The ODE function defining the dynamics.
            tol (float, optional): Tolerance for the ODE solver. Defaults to 1e-3.
            adjoint (bool, optional): Whether to use the adjoint method for
                backpropagation. Defaults to True.
        """
        super().__init__()
        # Assign attributes
        self.ode_func = ode_func
        self.tol = tol
        self.adjoint = adjoint

    # --- Callable & Context Manager ---
    def forward(self, x: Tensor, eval_time: Tensor | None = None) -> Tensor:
        """Forward the input through the ODE block.

        Args:
            x (Tensor): Input tensor of shape (B, C, H, W) and values ranging
                from 0.0 to 1.0.
            eval_time (Tensor, optional): Time steps at which to evaluate the
                ODE solution. If None, defaults to [0, 1]. Defaults to None.

        Returns:
            Tensor: Output tensor of shape (B, C, H, W) and values ranging from
                0.0 to 1.0.
        """
        if eval_time is None:
            t = torch.tensor([0, 1]).float().type_as(x)
        else:
            t = eval_time

        self.ode_func.nfe = 0
        x_aug = x

        return odeint_adjoint(
            func=self.ode_func,
            y0=x_aug,
            t=t,
            rtol=self.tol,
            atol=self.tol,
            method="dopri5", # "dopri5", "euler", "rk4"
            options={"max_num_steps": self.max_num_steps}
        )

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion

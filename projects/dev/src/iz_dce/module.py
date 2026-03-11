#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Modules.

This module provides various layers, blocks, and modules.
"""

from __future__ import annotations

__all__ = [
    "EnhanceFunction",
    "EnhanceFunctionTime",
    "ODEBlock",
]

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torchdiffeq import odeint_adjoint

from mon.models.restore.zs_n2n.module import ImprovedDenoiseNetwork
from mon.nn import (
    CharbonnierLoss,
    Conv2dTime,
    EdgePreservingLoss,
    FourierPE,
    LinearTime,
    SineLinear,
    SineLinearTime,
)
from mon.ops import anscombe, inverse_anscombe
from .loss import L_tv
from .utils import weights_init


# ==============================================================================
# region MODULES
# ==============================================================================

# --- Denoising ---

class DenoiseNet(nn.Module):
    """A simple CNN for estimating the noise in the input image."""

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        in_channels: int = 3,
        hidden_dim: int = 48,
        use_anscombe: bool = False,
    ):
        """Initialize a new instance.

        Args:
            in_channels (int, optional): Number of input channels. Defaults to 3.
            hidden_dim (int, optional): Number of hidden channels. Defaults to 48.
            use_anscombe (bool, optional): Whether to apply the Anscombe
                transform to the input before denoising. Defaults to False.
        """
        super().__init__()
        # Assign attributes
        self.use_anscombe = use_anscombe

        # Define network
        # self.model = DenoiseNetwork(in_channels=in_channels, hidden_dim=hidden_dim)
        self.model = ImprovedDenoiseNetwork(in_channels=in_channels, hidden_dim=hidden_dim)

    # --- Callable & Context Manager ---
    def forward(self, x: Tensor) -> tuple[Tensor | None, Tensor, Tensor]:
        """Forward the input through the network.

        Args:
            x (Tensor): Input tensor of shape (B, C, H, W) and values ranging
                from 0.0 to 1.0.

        Returns:
            tuple[Tensor | None, Tensor]: A tuple containing:
                - loss (Tensor | None): The denoising loss if in training mode,
                    otherwise None.
                - noise (Tensor): The predicted noise tensor of shape (B, C, H, W)
                    and values ranging from 0.0 to 1.0.
                - y (Tensor): The denoised image tensor of shape (B, C, H, W)
                    and values ranging from 0.0 to 1.0.
        """
        # 1. Add noise to the input image.
        # x = self.add_noise(x, 40)

        # 2. Apply Anscombe transform if enabled.
        # This stabilizes the variance of Poisson noise, making it more
        # Gaussian-like and easier for the CNN to learn.
        if self.use_anscombe:
            x = anscombe(x)

        # 3. Forward pass to the underlying denoising network.
        # loss = self.loss_n2n(x)
        loss = self.loss_p2n(x)
        noise = self.model(x)
        y = x - noise

        # 4. Apply inverse Anscombe transform if enabled.
        if self.use_anscombe:
            y = inverse_anscombe(y)

        y = torch.clamp(y, 0.0, 1.0)

        return loss, noise, y

    # --- Denoise Loss ---
    def loss_n2n(self, noisy_image: Tensor) -> Tensor:
        """Calculate the ZS-N2N denoising loss."""
        # L = nn.MSELoss()  # Vanilla loss function
        # L = nn.SmoothL1Loss()  # Improved loss function
        # L = CharbonnierLoss(eps=1e-5)
        L = EdgePreservingLoss()

        # Residual loss
        noisy1, noisy2 = self.pair_downsampler(noisy_image)
        pred1 = noisy1 - self.model(noisy1)
        pred2 = noisy2 - self.model(noisy2)
        loss_res = 0.5 * (L(noisy1, pred2) + L(noisy2, pred1))

        # Consistency loss
        noisy_denoised = noisy_image - self.model(noisy_image)
        denoised1, denoised2 = self.pair_downsampler(noisy_denoised)
        loss_cons = 0.5 * (L(pred1, denoised1) + L(pred2, denoised2))

        # Total loss
        loss = loss_res + loss_cons

        return loss

    def loss_p2n(self, noisy_image: Tensor) -> Tensor:
        """Calculates the Positive2Negative consistency loss."""
        # L = nn.MSELoss()  # Vanilla loss function
        # L = nn.SmoothL1Loss()  # Improved loss function
        # L = CharbonnierLoss(eps=1e-5)
        L = EdgePreservingLoss()

        # 1. Initial Full-Resolution Forward Pass
        # We get the predicted noise and the predicted clean image
        predicted_noise = self.model(noisy_image)
        predicted_clean = noisy_image - predicted_noise

        # 2. Re-noised Data Construction (RDC)
        # We detach the clean image and noise so gradients don't flow in a circle
        clean_detached = predicted_clean.detach()
        noise_detached = predicted_noise.detach()

        # To create a new noisy image, we apply a random spatial flip to the
        # predicted noise.
        # This breaks the spatial correlation of the sensor noise while keeping
        # its statistical distribution.
        if torch.rand(1) > 0.5:
            shuffled_noise = torch.flip(noise_detached, dims=[2]) # Flip vertically
        else:
            shuffled_noise = torch.flip(noise_detached, dims=[3]) # Flip horizontally

        # Construct the synthetic noisy image
        renoised_image = clean_detached + shuffled_noise

        # 3. Denoised Consistency Supervision (DCS)
        # Pass the synthetic noisy image through the network again
        predicted_noise_from_synthetic = self.model(renoised_image)
        predicted_clean_from_synthetic = renoised_image - predicted_noise_from_synthetic

        # 4. Calculate the Consistency Loss
        # The network should predict the exact same clean image, regardless of
        # how the noise was shuffled
        loss_cons = L(predicted_clean_from_synthetic, clean_detached)

        # To prevent the network from just predicting a flat gray image,
        # we add a small regularization term to ensure the predicted noise isn't zero
        loss_reg = torch.mean(torch.abs(predicted_clean_from_synthetic - noisy_image))

        # The final P2N loss
        loss = loss_cons + (0.1 * loss_reg)

        return loss

    # --- Utilities ---
    def add_noise(self, x: Tensor, noise_level: float) -> Tensor:
        """Add noise to the image."""
        noisy = x + torch.normal(0, noise_level / 255, x.shape).to(x.device)
        noisy = torch.clamp(noisy, 0, 1)
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


# --- Encoders ---

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
        self.out_channels = hidden_dim

        # Define network
        self.conv_1 = nn.Conv2d(in_channels, hidden_dim, 3, 1, padding=1, padding_mode="reflect")
        self.conv_3_1 = nn.Conv2d(hidden_dim, hidden_dim, 3, 1, padding=3//2, padding_mode="reflect")
        self.conv_5_1 = nn.Conv2d(hidden_dim, hidden_dim, 5, 1, padding=5//2, padding_mode="reflect")
        self.conv_3_2 = nn.Conv2d(hidden_dim * 2, hidden_dim * 2, 3, 1, padding=3//2, padding_mode="reflect")
        self.conv_5_2 = nn.Conv2d(hidden_dim * 2, hidden_dim * 2, 5, 1, padding=5//2, padding_mode="reflect")
        self.confusion = nn.Conv2d(hidden_dim * 4, hidden_dim, 1, 1, padding=0)
        self.act = nn.ReLU(inplace=True)
        self.norm_1 = nn.GroupNorm(1, hidden_dim)
        self.norm_2 = nn.GroupNorm(1, hidden_dim * 2)

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


# --- Decoders ---

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

        # Define network
        self.linear_1 = nn.Linear(in_channels, hidden_dim)
        self.linear_2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear_3 = nn.Linear(hidden_dim, hidden_dim)
        self.linear_4 = nn.Linear(hidden_dim, out_channels)
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


class DecoderINR(nn.Module):

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_dim: int = 32,
        pos_encode: bool = False,
        mapping_size: int = 256,
        B: float = 20.0,
    ):
        """Initialize a new instance.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            hidden_dim (int, optional): Number of hidden channels. Defaults to 32.
            pos_encode (bool, optional): Whether to use positional encoding.
                Defaults to False.
            mapping_size (int, optional): Size of Fourier feature mapping.
                Defaults to 256.
            B (float, optional): Fourier feature scaling factor. Defaults to 20.0.
        """
        super().__init__()
        # Assign attributes
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Define network
        if pos_encode:
            self.ff = FourierPE(mapping_size=mapping_size, B=B)
            coords_dim = self.ff.out_features
        else:
            self.ff = None
            coords_dim = 2

        patch_layers = [
            SineLinear(in_channels, hidden_dim, is_first=True),
            SineLinear(hidden_dim, hidden_dim // 2),
        ]
        spatial_layers = [
            SineLinear(coords_dim, hidden_dim, is_first=True),
            SineLinear(hidden_dim, hidden_dim // 2),
        ]
        output_layers = [
            SineLinear(hidden_dim, hidden_dim),
            SineLinear(hidden_dim, out_channels, is_last=True),
        ]

        self.patch_net = nn.Sequential(*patch_layers)
        self.spatial_net = nn.Sequential(*spatial_layers)
        self.output_net = nn.Sequential(*output_layers)

        weight_decay = [0.1, 0.0001, 0.001]
        self.params = []
        self.params += [{"params": self.spatial_net.parameters(), "weight_decay": weight_decay[0]}]
        self.params += [{"params": self.patch_net.parameters(), "weight_decay": weight_decay[1]}]
        self.params += [{"params": self.output_net.parameters(),"weight_decay": weight_decay[2]}]

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
        coords = self.ff(coords) if self.ff is not None else coords
        patch = self.patch_net(feat)
        coords = self.spatial_net(coords)
        A = F.tanh(self.output_net(torch.cat([patch, coords], dim=-1)))
        return A


# --- Main Network ---

class EnhanceFunction(nn.Module):
    """A module for enhancing the input image."""

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 32,
        num_iter: int = 8,
        imgsz: int = 256,
        chunk_size: int = 100000,
        use_depth: bool = False,
        use_anscombe: bool = False,
        *args, **kwargs
    ):
        """Initialize a new instance.

        Args:
            in_channels (int, optional): Number of input channels.
                Defaults to 3.
            hidden_dim (int, optional): Hidden dimension. Defaults to 32.
            imgsz (int, optional): Downsample the input image to this size for
                encoding. Defaults to 256.
            num_iter (int, optional): Number of iterations to apply the iterative
                enhancement. Defaults to 8.
            chunk_size (int): Number of pixels to process at once.
                Defaults to 100,000.
            use_depth (bool, optional): Whether to use depth as an additional
                input channel. Defaults to False.
            use_anscombe (bool, optional): Whether to apply the Anscombe
                transform to the input before denoising. Defaults to False.
        """
        super().__init__()
        # Assign attributes
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.imgsz = imgsz
        self.chunk_size = chunk_size
        self.num_iter = num_iter

        # Define network
        # Denoising module
        self.denoise = DenoiseNet(in_channels, use_anscombe=use_anscombe)
        # Encoder
        enc_in_channels = in_channels * 2 + 1 if use_depth else in_channels * 2
        self.encode = Encoder(enc_in_channels, hidden_dim)
        # Implicit decoder (Siren/Continuous MLP)
        # self.decode = Decoder(hidden_dim + 2, self.out_channels, hidden_dim)  # Input: features (32) + coordinates (2) = 34
        self.decode = DecoderINR(hidden_dim, self.out_channels, hidden_dim, True, imgsz)

    # --- Callable & Context Manager ---
    def forward(
        self,
        image: Tensor,
        depth: Tensor | None = None,
        save_debug: bool = False
    ) -> dict:
        """Forward the input through the network.

        Args:
            image (Tensor): Image tensor of shape (B, 3, H, W) and values
                ranging from 0.0 to 1.0.
            depth (Tensor, optional): Depth tensor of shape (B, 1, H, W) and
                values ranging from 0.0 to 1.0. Defaults to None.
            save_debug (bool, optional): Whether to save intermediate results for
                debugging. Defaults to False.
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
        l_denoise, noise, p_x = self.denoise(x)

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
        outputs = { "enhanced": y }
        if self.training or save_debug:
            outputs |= {
                "curve_map": A,
                "denoised": p_x,
                "noise_map": noise,
                "l_denoise": l_denoise,
            }
        return outputs

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
        A = A_flat.view(b, h, w, 3).permute(0, 3, 1, 2)  # [B, 3, H, W]

        return A

    # --- Enhance ---
    def enhance(self, image: Tensor, A: Tensor) -> Tensor:
        """Apply the iterative enhancement."""
        y = image
        c = self.out_channels
        for i in range(self.num_iter):
            if A.shape[1] == c * self.num_iter:
                A_i = A[:, i * c:(i + 1) * c, :, :]
            else:
                A_i = A
            y = y + A_i * (torch.pow(y, 2) - y)
        return y

# endregion


# ==============================================================================
# region ODE MODULES
# ==============================================================================

# --- Encoders ---

class EncoderTime(nn.Module):
    """A time-conditioned encoder module that takes in the time step as an
    additional input.
    """

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
        self.out_channels = hidden_dim

        # Define network
        self.conv_1 = Conv2dTime(in_channels, hidden_dim, 3, 1, padding=1, padding_mode="reflect")
        self.conv_3_1 = Conv2dTime(hidden_dim, hidden_dim, 3, 1, padding=3//2, padding_mode="reflect")
        self.conv_5_1 = Conv2dTime(hidden_dim, hidden_dim, 5, 1, padding=5//2, padding_mode="reflect")
        self.conv_3_2 = Conv2dTime(hidden_dim * 2, hidden_dim * 2, 3, 1, padding=3//2, padding_mode="reflect")
        self.conv_5_2 = Conv2dTime(hidden_dim * 2, hidden_dim * 2, 5, 1, padding=5//2, padding_mode="reflect")
        self.confusion = Conv2dTime(hidden_dim * 4, hidden_dim, 1, 1, padding=0)
        self.act = nn.ReLU(inplace=True)
        self.norm_1 = nn.GroupNorm(1, hidden_dim)
        self.norm_2 = nn.GroupNorm(1, hidden_dim * 2)

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


# --- Decoders ---

class DecoderTime(nn.Module):
    """A time-conditioned decoder module that takes in the time step as an
    additional input.
    """

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

        # Define network
        self.linear_1 = LinearTime(in_channels, hidden_dim)
        self.linear_2 = LinearTime(hidden_dim, hidden_dim)
        self.linear_3 = LinearTime(hidden_dim, hidden_dim)
        self.linear_4 = LinearTime(hidden_dim, out_channels)
        self.act = nn.ReLU(inplace=True)

    # --- Callable & Context Manager ---
    def forward(self, t: Tensor, feat: Tensor, coords: Tensor) -> Tensor:
        """Forward the input through the network.

        Args:
            t (Tensor): Time step tensor of shape (B,) or a scalar.
            feat (Tensor): Input feature tensor of shape (B, N, hidden_dim) and
                values ranging from 0.0 to 1.0.
            coords (Tensor): Input coordinate tensor of shape (B, N, 2) and
                values ranging from 0.0 to 1.0.

        Returns:
            Tensor: Output tensor of shape (B, N, out_channels) and values
                ranging from 0.0 to 1.0.
        """
        x = torch.cat([feat, coords], dim=-1)
        y = self.act(self.linear_1(t, x))
        y = self.act(self.linear_2(t, y))
        y = self.act(self.linear_3(t, y))
        y = F.tanh(self.linear_4(t, y))
        return y


class DecoderINRTime(nn.Module):

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_dim: int = 32,
        pos_encode: bool = False,
        mapping_size: int = 256,
        B: float = 20.0,
    ):
        """Initialize a new instance.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            hidden_dim (int, optional): Number of hidden channels. Defaults to 32.
            pos_encode (bool, optional): Whether to use positional encoding.
                Defaults to False.
            mapping_size (int, optional): Size of Fourier feature mapping.
                Defaults to 256.
            B (float, optional): Fourier feature scaling factor. Defaults to 20.0.
        """
        super().__init__()
        # Assign attributes
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Define network
        if pos_encode:
            self.ff = FourierPE(mapping_size=mapping_size, B=B)
            coords_dim = self.ff.out_features
        else:
            self.ff = None
            coords_dim = 2

        self.patch = SineLinearTime(in_channels, hidden_dim // 2, is_first=True)
        # self.patch1 = SineLinearTime(in_channels, hidden_dim, is_first=True)
        # self.patch2 = SineLinearTime(hidden_dim, hidden_dim // 2)

        self.spatial = SineLinearTime(coords_dim, hidden_dim // 2, is_first=True)
        # self.spatial1 = SineLinearTime(coords_dim, hidden_dim, is_first=True)
        # self.spatial2 = SineLinearTime(hidden_dim, hidden_dim // 2)

        self.output1 = SineLinearTime(hidden_dim, hidden_dim)
        self.output2 = SineLinearTime(hidden_dim, out_channels, is_last=True)

        weight_decay = [0.1, 0.0001, 0.001]
        self.params = []
        self.params += [{"params": self.spatial.parameters(), "weight_decay": weight_decay[0]}]
        # self.params += [{"params": self.spatial1.parameters(), "weight_decay": weight_decay[0]}]
        # self.params += [{"params": self.spatial2.parameters(), "weight_decay": weight_decay[0]}]
        self.params += [{"params": self.patch.parameters(), "weight_decay": weight_decay[1]}]
        # self.params += [{"params": self.patch1.parameters(), "weight_decay": weight_decay[1]}]
        # self.params += [{"params": self.patch2.parameters(), "weight_decay": weight_decay[1]}]
        self.params += [{"params": self.output1.parameters(),"weight_decay": weight_decay[2]}]
        self.params += [{"params": self.output2.parameters(),"weight_decay": weight_decay[2]}]

    # --- Callable & Context Manager ---
    def forward(self, t: Tensor, feat: Tensor, coords: Tensor) -> Tensor:
        """Forward the input through the network.

        Args:
            t (Tensor): Time step tensor of shape (B,) or a scalar.
            feat (Tensor): Input feature tensor of shape (B, N, hidden_dim) and
                values ranging from 0.0 to 1.0.
            coords (Tensor): Input coordinate tensor of shape (B, N, 2) and
                values ranging from 0.0 to 1.0.

        Returns:
            Tensor: Output tensor of shape (B, N, out_channels) and values
                ranging from 0.0 to 1.0.
        """
        coords = self.ff(coords) if self.ff is not None else coords
        patch = self.patch(t, feat)
        # patch = self.patch1(t, feat)
        # patch = self.patch2(t, patch)
        coords = self.spatial(t, coords)
        # coords = self.spatial1(t, coords)
        # coords = self.spatial2(t, coords)
        concat = torch.cat([patch, coords], dim=-1)
        output = self.output1(t, concat)
        output = self.output2(t, output)
        A = F.tanh(output)
        return A


# --- Main Network ---

class EnhanceFunctionTime(nn.Module):
    """A module for enhancing the input image with time conditioning."""

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 32,
        imgsz: int = 128,
        chunk_size: int = 100000,
        use_depth: bool = False,
        use_anscombe: bool = False,
        *args, **kwargs
    ):
        """Initialize a new instance.

        Args:
            in_channels (int, optional): Number of input channels.
                Defaults to 3.
            hidden_dim (int, optional): Hidden dimension. Defaults to 32.
            imgsz (int, optional): Downsample the input image to this size for
                encoding. Defaults to 128.
            chunk_size (int): Number of pixels to process at once.
                Defaults to 100,000.
            use_depth (bool, optional): Whether to use depth as an additional
                input channel. Defaults to False.
            use_anscombe (bool, optional): Whether to apply the Anscombe
                transform to the input before denoising. Defaults to False.
        """
        super().__init__()
        # Assign attributes
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.imgsz = imgsz
        self.chunk_size = chunk_size
        self.use_depth = use_depth

        # Define network
        # Denoising module
        self.denoise = DenoiseNet(in_channels, use_anscombe=use_anscombe)
        # Encoder
        enc_in_channels = in_channels * 2 + 1 if use_depth else in_channels * 2
        self.encode = EncoderTime(enc_in_channels, hidden_dim)
        # Implicit refiner (Siren/Continuous MLP)
        # self.decode = DecoderTime(hidden_dim + 2, self.out_channels, hidden_dim)  # Input: features (32) + coordinates (2) = 34
        self.decode = DecoderINRTime(hidden_dim, self.out_channels, hidden_dim, True, imgsz)

        # Allocate resources
        self.nfe = 0
        self.pred_t = []
        self.last_curve_map = None
        self.last_noise_map = None
        self.last_denoised = None
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
        depth = _d = x[:, 3:4, :, :] if self.use_depth else None  # Depth map
        b, c, h, w = x.shape
        size = (self.imgsz, self.imgsz)

        # We downsample the input to 512x512 so the CNN doesn't cause an OOM error
        if (h, w) != size:
            _x = F.interpolate(_x, size=size, mode="bilinear", align_corners=True)
            _d = F.interpolate(_d, size=size, mode="bilinear", align_corners=True) if _d is not None else None

        # 2. Denoise
        l_denoise, noise, p_x = self.denoise(_x)

        # 3. Fusion
        if _d is not None:
            _x_in = torch.cat([_x, p_x, _d], dim=1)
        else:
            _x_in = torch.cat([_x, p_x], dim=1)

        # 4. Encode
        feat = self.encode(t, _x_in)

        # 5. Predict curve parameters
        if (h, w) == size:
            A = self.predict_curve_map(t, feat, self.imgsz, self.imgsz)
        else:
            A = self.predict_curve_map_chunk(t, feat, h, w)

        # 6. Enhance
        y = A * (torch.pow(image, 2) - image)

        # 7. Return final and intermediate results for debugging
        self.last_curve_map = A
        self.last_noise_map = noise
        self.last_denoised = p_x
        self.pred_t.append(t.item())

        # Since ODE solvers typically expect the output to be the same shape as
        # the input, we concatenate the intermediate results along the channel
        # dimension for debugging purposes. The final output will still be `y`,
        # which is the enhanced image.
        depth = depth if self.use_depth else torch.zeros(b, 1, h, w, device=x.device)
        l_tv = torch.ones_like(A) * self.tv_loss(A, depth)
        l_denoise = torch.ones_like(A) * l_denoise
        outputs = torch.cat([y, depth, l_tv, l_denoise], dim=1)
        return outputs

    # --- Curve Map ---
    def predict_curve_map(self, t: Tensor, feat: Tensor, h: int, w: int) -> Tensor:
        b = feat.shape[0]
        device = feat.device

        # We map the massive target resolution to the [-1, 1] continuous space.
        h_coords = torch.linspace(-1, 1, steps=h, device=device)
        w_coords = torch.linspace(-1, 1, steps=w, device=device)
        grid_h, grid_w = torch.meshgrid(h_coords, w_coords, indexing="ij")
        coords = torch.stack([grid_w, grid_h], dim=-1).view(1, -1, 2).repeat(b, 1, 1)  # [B, H*W, 2]

        sampled_feat = F.grid_sample(feat, coords.unsqueeze(1), mode="bilinear", align_corners=True)
        sampled_feat = sampled_feat.squeeze(2).permute(0, 2, 1)  # [B, N, 32]

        A = self.decode(t, sampled_feat, coords)
        A = A.view(b, h, w, 3).permute(0, 3, 1, 2)

        return A

    def predict_curve_map_chunk(self, t: Tensor, feat: Tensor, h: int, w: int) -> Tensor:
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

        # 1. Chunked MLP Inference
        # Process the points in batches of `chunk_size` to cap VRAM usage.
        for i in range(0, total_points, chunk_size):
            coords_chunk = coords[:, i:i+chunk_size, :]  # [B, chunk, 2]

            # Sample from the 512x512 feature map at the exact target coordinates
            sampled_feat = F.grid_sample(feat, coords_chunk.unsqueeze(1), mode="bilinear", align_corners=True)
            sampled_feat = sampled_feat.squeeze(2).permute(0, 2, 1)  # [B, chunk, 32]

            # Predict the curve parameters for this chunk
            A_chunk = self.decode(t, sampled_feat, coords_chunk)
            A_list.append(A_chunk)

        # 2. Reconstruct the spatial curve parameter map
        A_flat = torch.cat(A_list, dim=1)  # [B, H*W, 24]
        A = A_flat.view(b, h, w, 3).permute(0, 3, 1, 2)  # [B, 24, H, W]

        return A


class ODEBlock(nn.Module):

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        ode_func: nn.Module,
        tol: float = 1e-5,
        adjoint: bool = True,
        ode_options: dict | None = None,
        *args, **kwargs
    ):
        """Initialize a new instance.

        Args:
            ode_func (nn.Module): The ODE function defining the dynamics.
            tol (float, optional): Tolerance for the solver. Defaults to 1e-5.
            adjoint (bool, optional): Whether to use the adjoint method for
                backpropagation. Defaults to True.
            ode_options (dict, optional): Additional options to pass to the ODE
                solver. Defaults to None.
        """
        super().__init__()
        # Assign attributes
        self.ode_func = ode_func
        self.tol = tol
        self.adjoint = adjoint
        self.ode_options = ode_options or {}

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
            method="dopri5",  # "dopri5", "euler", "rk4"
            options=self.ode_options,
        )

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion

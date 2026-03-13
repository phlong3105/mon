#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Modules.

This module provides various layers, blocks, and modules.
"""

from __future__ import annotations

__all__ = [
    "Decoder",
    "DecoderSIREN",
    "Denoiser",
    "Encoder",
    "EnhancementCurveODE",
]

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from mon.nn import EdgePreservingLoss, FourierPE, SineLinear
from mon.ops import anscombe, inverse_anscombe
from .utils import weights_init


# ==============================================================================
# region DENOISE
# ==============================================================================

class DenoiseNetwork(nn.Module):

    # --- Lifecycle & Initialization ---
    def __init__(self, in_channels: int = 3, hidden_dim: int = 48):
        """Initialize a new instance.

        Args:
            in_channels (int, optional): Number of input channels. Defaults to 3.
            hidden_dim (int, optional): Number of hidden channels. Defaults to 48.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim, in_channels, kernel_size=1)
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    # --- Callable & Context Manager ---
    def forward(self, x: Tensor) -> Tensor:
        """Forward the input through the network.

        Args:
            x (Tensor): Input tensor of shape (B, C, H, W) and values ranging
                from 0.0 to 1.0.

        Returns:
            Tensor: Predicted noise tensor of shape (B, C, H, W) and values
                ranging from 0.0 to 1.0.
        """
        y = self.act(self.conv1(x))
        y = self.act(self.conv2(y))
        noise = self.conv3(y)
        return noise


class GlobalContextBlock(nn.Module):
    """Global Context Self-Attention Mechanism.

    Enhance global semantic information before the convolutions.
    """

    # --- Lifecycle & Initialization ---
    def __init__(self, in_channels: int = 3):
        """Initialize a new instance.

        Args:
            in_channels (int, optional): Number of input channels. Defaults to 3.
        """
        super().__init__()
        # Spatial pooling branch to compute attention matrix
        self.context_conv = nn.Conv2d(in_channels, 1, kernel_size=1)

        # Transform branch (Conv2d -> LayerNorm -> ReLU -> Conv2d)
        self.transform = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.LayerNorm([in_channels, 1, 1]), # Normalizes across channels
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=1)
        )

    # --- Callable & Context Manager ---
    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape

        # 1. Calculate similarity/attention matrix
        context = self.context_conv(x).view(b, 1, h * w)
        context = F.softmax(context, dim=-1)

        # 2. Weight the average of the features
        x_reshaped = x.view(b, c, h * w)
        context_out = torch.bmm(
            x_reshaped, context.transpose(1, 2)
        ).view(b, c, 1, 1)

        # 3. Transform and add back to the original input via skip connection
        transform_out = self.transform(context_out)
        return x + transform_out


class ChannelAttentionModule(nn.Module):
    """Channel Attention Mechanism.

    Extract dependencies between channels using 1D convolution.
    """

    # --- Lifecycle & Initialization ---
    def __init__(self, kernel_size: int = 3):
        """Initialize a new instance.

        Args:
            kernel_size (int, optional): Kernel size for the 1D convolution.
                Defaults to 3.
        """
        super().__init__()
        # Adaptive average pooling compresses each channel to one dimension
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=False
        )
        self.sigmoid = nn.Sigmoid()

    # --- Callable & Context Manager ---
    def forward(self, x: Tensor) -> Tensor:
        b, c, _, _ = x.shape
        y = self.avg_pool(x)                   # (B, C, 1, 1)
        y = y.squeeze(-1).transpose(-1, -2)    # (B, 1, C) for Conv1D
        y = self.conv(y)                       # (B, 1, C)
        y = y.transpose(-1, -2).unsqueeze(-1)  # (B, C, 1, 1)

        # Multiply weights with the corresponding elements of the feature map
        return x * self.sigmoid(y)


class ImprovedDenoiseNetwork(nn.Module):

    # --- Lifecycle & Initialization ---
    def __init__(self, in_channels: int = 3, hidden_dim: int = 48):
        """Initialize a new instance.

        Args:
            in_channels (int, optional): Number of input channels. Defaults to 3.
            hidden_dim (int, optional): Number of hidden channels. Defaults to 48.
        """
        super().__init__()

        # 1. Global Context Module
        self.global_context = GlobalContextBlock(in_channels=in_channels)

        # 2. Noise Fitting Convolutions
        # Increases channels from 3 to 48 with 3x3 kernel
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.lrelu1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # Uses dilated convolution (dilation=2) to increase receptive field to 5x5
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=2, dilation=2)
        self.lrelu2 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # Reduces channels back from 48 to 3 with 1x1 kernel
        self.conv3 = nn.Conv2d(hidden_dim, in_channels, kernel_size=1)

        # 3. Channel Attention Module
        self.channel_attention = ChannelAttentionModule()

    # --- Callable & Context Manager ---
    def forward(self, x: Tensor) -> Tensor:
        """Forward the input through the network.

        Args:
            x (Tensor): Input tensor of shape (B, C, H, W) and values ranging
                from 0.0 to 1.0.

        Returns:
            Tensor: Predicted noise tensor of shape (B, C, H, W) and values
                ranging from 0.0 to 1.0.
        """
        # Step 1: Global Context
        y = self.global_context(x)

        # Step 2: Convolutional mapping
        y = self.lrelu1(self.conv1(y))
        y = self.lrelu2(self.conv2(y))
        y = self.conv3(y)

        # Step 3: Channel Attention
        noise = self.channel_attention(y)

        # Note: The network fits the noise parameter f_θ(y).
        # To get the denoised image during inference, you subtract this from
        # the input: x = y - f_θ(y)
        return noise


class Denoiser(nn.Module):
    """A simple CNN for estimating the noise in the input image."""

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        in_channels: int = 3,
        hidden_dim: int = 48,
        noise_level: float | None = None,
        use_anscombe: bool = False,
    ):
        """Initialize a new instance.

        Args:
            in_channels (int, optional): Number of input channels. Defaults to 3.
            hidden_dim (int, optional): Number of hidden channels. Defaults to 48.
            noise_level (float | None, optional): The noise level to add to the
                input. If None, no noise is added. Defaults to None.
            use_anscombe (bool, optional): Whether to apply the Anscombe
                transform to the input before denoising. Defaults to False.
        """
        super().__init__()
        # Assign attributes
        self.use_anscombe = use_anscombe
        self.noise_level = noise_level

        # Define network
        # self.model = DenoiseNetwork(in_channels=in_channels, hidden_dim=hidden_dim)
        self.model = ImprovedDenoiseNetwork(in_channels=in_channels, hidden_dim=hidden_dim)

    # --- Callable & Context Manager ---
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Forward the input through the network.

        Args:
            x (Tensor): Input tensor of shape (B, C, H, W) and values ranging
                from 0.0 to 1.0.

        Returns:
            tuple[Tensor, Tensor, Tensor]: A tuple containing:
                - loss (Tensor): The denoising loss if in training mode, otherwise None.
                - noise (Tensor): The predicted noise tensor of shape (B, C, H, W)
                    and values ranging from 0.0 to 1.0.
                - y (Tensor): The denoised image tensor of shape (B, C, H, W)
                    and values ranging from 0.0 to 1.0.
        """
        # 1. Add noise to the input image.
        if self.noise_level is not None:
            x = self.add_noise(x, 40)

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

# endregion


# ==============================================================================
# region MODULES
# ==============================================================================

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


class DecoderSIREN(nn.Module):

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


# --- Curve ---

class EnhancementCurveODE(nn.Module):

    # --- Lifecycle & Initialization ---
    def __init__(self, A: Tensor):
        super().__init__()
        self.A = A

    # --- Callable & Context Manager ---
    def forward(self, t: Tensor, y: Tensor) -> Tensor:
        """Forward the input through the network.

        Args:
            t (Tensor): Time tensor.
            y (Tensor): Input tensor of shape (B, C, H, W) and values ranging
                from 0.0 to 1.0.

        Returns:
            Tensor: Output tensor of shape (B, C, H, W) and values ranging
                from 0.0 to 1.0.
        """
        y = torch.clamp(y, 0.0, 1.0)
        # Swapped to y * (1 - y) so A learns positive values!
        dy_dt = self.A * y * (1.0 - y)
        return dy_dt

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Modules.

This module provides various layers, blocks, and modules for the DCC-Net model.
"""

from __future__ import annotations

__all__ = [
    "C_Net",
    "G_Net",
    "R_Net",
]

import torch
from torch import nn, Tensor


# ==============================================================================
# region MODULES
# ==============================================================================

# --- Layers ---

class BasicConv(nn.Module):

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        bias: bool = True,
        norm: bool = True,
        activation: bool = True,
        transpose: bool = False
    ):
        """Initialize a new instance."""
        super().__init__()
        # Define layers
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias
                )
            )
        else:
            layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias
                )
            )
        if norm:
            layers.append(nn.InstanceNorm2d(out_channels))
        if activation:
            layers.append(nn.GELU())

        self.main = nn.Sequential(*layers)

    # --- Callable & Context Manager ---
    def forward(self, x: Tensor) -> Tensor:
        return self.main(x)


class RB(nn.Module):

    # --- Lifecycle & Initialization ---
    def __init__(self, channels: int):
        """Initialize a new instance."""
        super().__init__()
        # Define layers
        self.layer_1 = BasicConv(channels, channels, 3, 1)
        self.layer_2 = BasicConv(channels, channels, 3, 1)

    # --- Callable & Context Manager ---
    def forward(self, x: Tensor) -> Tensor:
        y = self.layer_1(x)
        y = self.layer_2(y)
        return y + x


class Downscale(nn.Module):

    # --- Lifecycle & Initialization ---
    def __init__(self, in_channels: int):
        """Initialize a new instance."""
        super().__init__()
        self.main = BasicConv(in_channels, in_channels * 2, 3, 2)

    # --- Callable & Context Manager ---
    def forward(self, x: Tensor) -> Tensor:
        return self.main(x)


class Upscale(nn.Module):

    # --- Lifecycle & Initialization ---
    def __init__(self, in_channels: int):
        """Initialize a new instance."""
        super().__init__()
        # Define layers
        self.main = BasicConv(
            in_channels=in_channels,
            out_channels=in_channels // 2,
            kernel_size=4,
            stride=2,
            activation=True,
            transpose=True,
        )

    # --- Callable & Context Manager ---
    def forward(self, x: Tensor) -> Tensor:
        return self.main(x)


# --- Blocks ---

class CMA(nn.Module):

    # --- Lifecycle & Initialization ---
    def __init__(self, in_channels: int, out_channels: int):
        """Initialize a new instance."""
        super().__init__()
        # Define layers
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest")
        )

    # --- Callable & Context Manager ---
    def forward(self, color: Tensor, gray: Tensor) -> tuple[Tensor, Tensor]:
        # color: color features
        # gray: gray image features

        # L1 distance
        channels = color.shape[1]
        sim_mat_l1 = -torch.abs(gray - color)  # <0  (b,c,h,w)
        sim_mat_l1 = torch.sum(sim_mat_l1, dim=1, keepdim=True)  # (b,1,h,w)
        sim_mat_l1 = torch.sigmoid(sim_mat_l1)  # (0, 0.5) (b,1,h,w)
        sim_mat_l1 = sim_mat_l1.repeat(1, channels, 1, 1)
        sim_mat_l1 = 2 * sim_mat_l1  # (0, 1)

        # Cos distance
        sim_mat_cos = gray * color  # >0 (b,c,h,w)
        sim_mat_cos = torch.sum(sim_mat_cos, dim=1, keepdim=True)  # (b,1,h,w)
        sim_mat_cos = torch.tanh(sim_mat_cos)  # (0, 1) (b,1,h,w)
        sim_mat_cos = sim_mat_cos.repeat(1, channels, 1, 1)  # (0, 1)

        # similarity matrix
        sim_mat = sim_mat_l1 * sim_mat_cos  # (0, 1)

        # Color embedding
        x_color = gray + color * sim_mat

        # color features upsample
        color_up = self.conv(color)

        return x_color, color_up


class PCE(nn.Module):
    """Pyramid color embedding."""

    # --- Lifecycle & Initialization ---
    def __init__(self, num_channels: int = 64):
        """Initialize a new instance."""
        super().__init__()
        # Define layers
        self.cma_3 = CMA(128, 64)
        self.cma_2 = CMA(64, 32)
        self.cma_1 = CMA(32, 16)

    # --- Callable & Context Manager ---
    def forward(self, color: Tensor, shortcuts: list[Tensor]) -> list[Tensor]:
        x3_color, c3 = self.cma_3(color, shortcuts[2])
        x2_color, c2 = self.cma_2(c3, shortcuts[1])
        x1_color, c1 = self.cma_1(c2, shortcuts[0])
        return [x1_color, x2_color, x3_color]


# --- Modules ---

class R_Net(nn.Module):

    # --- Lifecycle & Initialization ---
    def __init__(self, depth: tuple[int, ...] = (2, 2, 2, 2)):
        super().__init__()
        # Assign attributes
        base_channel = 32

        # Define layers
        # Encoder
        self.Encoder = nn.ModuleList([
            BasicConv(base_channel, base_channel, 3, 1),
            nn.Sequential(*[RB(base_channel) for _ in range(depth[0])]),
            Downscale(base_channel),
            BasicConv(base_channel * 2, base_channel * 2, 3, 1),
            nn.Sequential(*[RB(base_channel * 2) for _ in range(depth[1])]),
            Downscale(base_channel * 2),
            BasicConv(base_channel * 4, base_channel * 4, 3, 1),
            nn.Sequential(*[RB(base_channel * 4) for _ in range(depth[2])]),
            Downscale(base_channel * 4),
        ])

        # Middle
        self.middle = nn.Sequential(*[RB(base_channel * 8) for _ in range(depth[3])])

        # Decoder
        self.Decoder = nn.ModuleList([
            Upscale(base_channel * 8),
            BasicConv(base_channel * 8, base_channel * 4, 3, 1),
            nn.Sequential(*[RB(base_channel * 4) for _ in range(depth[2])]),
            Upscale(base_channel * 4),
            BasicConv(base_channel * 4, base_channel * 2, 3, 1),
            nn.Sequential(*[RB(base_channel * 2) for _ in range(depth[1])]),
            Upscale(base_channel * 2),
            BasicConv(base_channel * 2, base_channel, 3, 1),
            nn.Sequential(*[RB(base_channel) for _ in range(depth[0])]),
        ])

        # Conv
        self.conv_first = BasicConv(4, base_channel, 3, 1)
        self.conv_last = nn.Conv2d(base_channel, 3, 3, 1, 1)
        self.pce = PCE()

    # --- Callable & Context Manager ---
    def forward(self, image: Tensor, gray: Tensor, color_feature: Tensor) -> Tensor:
        x = torch.cat([image, gray], 1)
        x = self.conv_first(x)
        x, shortcuts = self.encoder(x)
        x = self.middle(x)
        shortcuts = self.pce(color_feature, shortcuts)
        x = self.decoder(x, shortcuts)
        x = self.conv_last(x)
        image_color = (torch.tanh(x) + 1) / 2
        return image_color

    def encoder(self, x: Tensor) -> tuple[Tensor, list[Tensor]]:
        shortcuts = []
        for i in range(len(self.Encoder)):
            x = self.Encoder[i](x)
            if (i + 2) % 3 == 0:
                shortcuts.append(x)
        return x, shortcuts

    def decoder(self, x: Tensor, shortcuts: list[Tensor]) -> Tensor:
        for i in range(len(self.Decoder)):
            if (i + 2) % 3 == 0:
                index = len(shortcuts) - (i//3 + 1)
                x = torch.cat([x, shortcuts[index]], 1)
            x = self.Decoder[i](x)
        return x


class C_Net(nn.Module):

    # --- Lifecycle & Initialization ---
    def __init__(self, d_hist, depth: tuple[int, ...] = (2, 2, 2)):
        super().__init__()
        # Assign attributes
        base_channel = 32

        # Define layers
        # Encoder
        self.Encoder = nn.ModuleList([
            BasicConv(base_channel, base_channel, 3, 1),
            nn.Sequential(*[RB(base_channel) for _ in range(depth[0])]),
            Downscale(base_channel),
            BasicConv(base_channel * 2, base_channel * 2, 3, 1),
            nn.Sequential(*[RB(base_channel * 2) for _ in range(depth[1])]),
            Downscale(base_channel * 2),
            BasicConv(base_channel * 4, base_channel * 4, 3, 1),
            nn.Sequential(*[RB(base_channel * 4) for _ in range(depth[2])]),
        ])

        self.conv_first = BasicConv(3, base_channel, 3, 1)

        # Color hist
        self.conv_color = BasicConv(base_channel * 4, 256 * 3, 3, 1)
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, d_hist)
        self.softmax = nn.Softmax(dim=2)
        self.d_hist = d_hist

    # --- Callable & Context Manager ---
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self.conv_first(x)
        x, _ = self.encoder(x)
        color_hist = self.color_forward(x)
        return color_hist, x

    def encoder(self, x: Tensor) -> tuple[Tensor, list[Tensor]]:
        shortcuts = []
        for i in range(len(self.Encoder)):
            x = self.Encoder[i](x)
            if (i + 2) % 3 == 0:
                shortcuts.append(x)
        return x, shortcuts

    def color_forward(self, x: Tensor) -> Tensor:
        x = self.conv_color(x)
        x = self.pooling(x)
        x = torch.reshape(x, (-1, 3, 256))
        color_hist = self.softmax(self.fc(x))
        return color_hist


class G_Net(nn.Module):

    # --- Lifecycle & Initialization ---
    def __init__(self, depth: tuple[int, ...] = (2, 2, 2, 2)):
        super().__init__()
        # Assign attributes
        base_channel = 32

        # Define layers
        # Encoder
        self.Encoder = nn.ModuleList([
            BasicConv(base_channel, base_channel, 3, 1),
            nn.Sequential(*[RB(base_channel) for _ in range(depth[0])]),
            Downscale(base_channel),
            BasicConv(base_channel * 2, base_channel * 2, 3, 1),
            nn.Sequential(*[RB(base_channel * 2) for _ in range(depth[1])]),
            Downscale(base_channel * 2),
            BasicConv(base_channel * 4, base_channel * 4, 3, 1),
            nn.Sequential(*[RB(base_channel * 4) for _ in range(depth[2])]),
            Downscale(base_channel * 4),
        ])

        # Middle
        self.middle = nn.Sequential(*[RB(base_channel*8) for _ in range(depth[3])])

        # Decoder
        self.Decoder = nn.ModuleList([
            Upscale(base_channel * 8),
            BasicConv(base_channel * 8, base_channel * 4, 3, 1),
            nn.Sequential(*[RB(base_channel * 4) for _ in range(depth[2])]),
            Upscale(base_channel * 4),
            BasicConv(base_channel * 4, base_channel * 2, 3, 1),
            nn.Sequential(*[RB(base_channel * 2) for _ in range(depth[1])]),
            Upscale(base_channel * 2),
            BasicConv(base_channel * 2, base_channel, 3, 1),
            nn.Sequential(*[RB(base_channel) for _ in range(depth[0])]),
        ])

        # Conv
        self.conv_first = BasicConv(3, base_channel, 3, 1)
        self.conv_last = nn.Conv2d(base_channel, 1, 3, 1, 1)

    # --- Callable & Context Manager ---
    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_first(x)
        x, shortcuts = self.encoder(x)
        x =  self.middle(x)
        x = self.decoder(x, shortcuts)
        x = self.conv_last(x)
        gray = (torch.tanh(x) + 1) / 2
        return gray

    def encoder(self, x: Tensor) -> tuple[Tensor, list[Tensor]]:
        shortcuts = []
        for i in range(len(self.Encoder)):
            x = self.Encoder[i](x)
            if (i + 2) % 3 == 0:
                shortcuts.append(x)
        return x, shortcuts

    def decoder(self, x: Tensor, shortcuts: list[Tensor]) -> Tensor:
        for i in range(len(self.Decoder)):
            if (i + 2) % 3 == 0:
                index = len(shortcuts) - (i//3 + 1)
                x = torch.cat([x, shortcuts[index]], 1)
            x = self.Decoder[i](x)
        return x

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion

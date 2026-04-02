#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Modules.

This module provides various layers, blocks, and modules for the SGZ model.
"""

from __future__ import annotations

__all__ = [
    "DSC",
    "FPN",
    "TC",
    "resnet101",
    "resnet50",
]

import torch
import torch.utils.model_zoo as model_zoo
from torch import nn, Tensor
from torch.nn import functional as F


# ==============================================================================
# region MODULES
# ==============================================================================

# --- ResNet ---

resnet_model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
}


class Bottleneck(nn.Module):

    expansion: int = 4

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        in_channels: int,
        channels: int,
        stride: int = 1,
        downsample: nn.Module | None = None
    ):
        super().__init__()
        self.downsample = downsample
        self.stride = stride

        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        self.conv3 = nn.Conv2d(channels, channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)

    # --- Callable & Context Manager ---
    def forward(self, x: Tensor) -> Tensor:
        residual = x

        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)

        y = self.conv2(y)
        y = self.bn2(y)
        y = self.relu(y)

        y = self.conv3(y)
        y = self.bn3(y)

        if self.downsample is not None:
            residual = self.downsample(x)

        y += residual
        y = self.relu(y)

        return y


class ResNet(nn.Module):

    # --- Lifecycle & Initialization ---
    def __init__(self, block, layers):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, planes, stride, downsample))
        self.in_channels = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_channels, planes))

        return nn.Sequential(*layers)

    # --- Callable & Context Manager ---
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        return c2, c3, c4, c5


def resnet50(pretrained: bool = False, *args, **kwargs):
    """Constructs a ResNet-50 model."""
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(resnet_model_urls["resnet50"]), strict=False)
    return model


def resnet101(pretrained: bool = False, *args, **kwargs):
    """Constructs a ResNet-101 model."""
    model = ResNet(Bottleneck, [3, 4, 23, 3])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(resnet_model_urls["resnet101"]), strict=False)
    return model


# --- FPN (Segmentation with ResNet backbone) ---

class FPNModule(nn.Module):

    # --- Lifecycle & Initialization ---
    def __init__(self, num_classes: int):
        super().__init__()
        # Top layer
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels

        # Smooth layers
        self.smooth1_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth4_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth1_2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.smooth2_2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.smooth3_2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.smooth4_2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

        # Classify layers
        self.classify = nn.Conv2d(128 * 4, num_classes, kernel_size=3, stride=1, padding=1)

    # --- Callable & Context Manager ---
    def forward(self, c2: Tensor, c3: Tensor, c4: Tensor, c5: Tensor) -> Tensor:
        # Top-down
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p2 = self._upsample_add(p3, self.latlayer3(c2))
        # Smooth
        p5 = self.smooth1_2(self.smooth1_1(p5))
        p4 = self.smooth2_2(self.smooth2_1(p4))
        p3 = self.smooth3_2(self.smooth3_1(p3))
        p2 = self.smooth4_2(self.smooth4_1(p2))
        # Classify
        output = self.classify(self._concatenate(p5, p4, p3, p2))

        return output

    def _concatenate(self, p5: Tensor, p4: Tensor, p3: Tensor, p2: Tensor) -> Tensor:
        _, _, h, w = p2.size()
        p5 = F.upsample(p5, size=(h, w), mode="bilinear")
        p4 = F.upsample(p4, size=(h, w), mode="bilinear")
        p3 = F.upsample(p3, size=(h, w), mode="bilinear")
        return torch.cat([p5, p4, p3, p2], dim=1)

    def _upsample_add(self, x: Tensor, y: Tensor) -> Tensor:
        """Upsample and add two feature maps.

        Args:
            x: (Variable) top feature map to be upsampled.
            y: (Variable) lateral feature map.

        Returns:
            (Variable) added feature map.

        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        """
        _, _, h, w = y.size()
        return F.upsample(x, size=(h, w), mode="bilinear") + y


class FPN(nn.Module):

    # --- Lifecycle & Initialization ---
    def __init__(self, num_classes: int):
        super().__init__()
        # ResNet backbone
        self.resnet = resnet50(pretrained=True)

        # FPN module
        self.fpn = FPNModule(num_classes)

        # Initialize weights
        for m in self.fpn.children():
            nn.init.normal_(m.weight, mean=0, std=0.01)
            nn.init.constant_(m.bias, 0)

    # --- Callable & Context Manager ---
    def forward(self, x: Tensor) -> Tensor:
        # Top-down
        c2, c3, c4, c5 = self.resnet.forward(x)
        return self.fpn.forward(c2, c3, c4, c5)


# --- SGZ Modules ---

class TC(nn.Module):
    """Traditional Convolution."""

    # --- Lifecycle & Initialization ---
    def __init__(self, in_channels: int, out_channels: int):
        """Initialize a new instance.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    # --- Callable & Context Manager ---
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x (Tensor): Input tensor of shape (B, C_in, H, W) and values
                ranging from 0.0 to 1.0.

        Returns:
            Tensor: Output tensor of shape (B, C_out, H, W) and values ranging
                from 0.0 to 1.0.
        """
        return self.conv(x)


class DSC(nn.Module):
    """Depthwise Separable Convolution."""

    # --- Lifecycle & Initialization ---
    def __init__(self, in_channels: int, out_channels: int):
        """Initialize a new instance."""
        super().__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_channels,
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
        )

    # --- Callable & Context Manager ---
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x (Tensor): Input tensor of shape (B, C_in, H, W) and values
                ranging from 0.0 to 1.0.

        Returns:
            Tensor: Output tensor of shape (B, C_out, H, W) and values ranging
                from 0.0 to 1.0.
        """
        return self.point_conv(self.depth_conv(x))

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion

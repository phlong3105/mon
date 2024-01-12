#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements Zero-DCE and Zero-DCE++ models."""

from __future__ import annotations

__all__ = [
    "ZeroDCE",
    "ZeroDCEPP",
]

import torch

from mon.globals import MODELS
from mon.vision import core, nn
from mon.vision.enhance.llie import base
from mon.vision.nn import functional as F

console      = core.console
_current_dir = core.Path(__file__).absolute().parent


# region Layer

class PixelwiseHigherOrderLECurve(nn.MergingLayerParsingMixin, nn.Module):
    """Pixelwise Light-Enhancement Curve is a higher-order curves that can be
    applied iteratively to enable more versatile adjustment to cope with
    challenging low-light conditions:
        LE_{n}(x) = LE_{n−1}(x) + A_{n}(x) * LE_{n−1}(x)(1 − LE_{n−1}(x)),
        
        where `A` is a parameter map with the same size as the given image, and
        `n` is the number of iterations, which controls the curvature.
    
    This module is designed to go with:
        - Zero-DCE (estimate 3 * n curve parameter maps)
        - Zero-DCE++, Zero-DCE-Tiny (estimate 3 curve parameter maps)
    
    Args:
        n: Number of iterations.
    """
    
    def __init__(self, n: int):
        super().__init__()
        self.n = n
    
    def forward(self, input: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        # Split
        y = input[0]  # Trainable curve parameters learned from the previous layer
        x = input[1]  # Original input image
        
        # Prepare curve parameter
        _, c1, _, _ = x.shape  # Should be 3
        _, c2, _, _ = y.shape  # Should be 3 * n
        single_map = True
        
        if c2 == c1 * self.n:
            single_map = False
            y = torch.split(y, c1, dim=1)
        elif c2 == 3:
            pass
        else:
            raise ValueError(
                f"Curve parameter maps 'c2' must be '3' or '3 * {self.n}'. "
                f"But got: {c2}."
            )
        
        # Estimate curve parameter
        for i in range(self.n):
            y_i = y if single_map else y[i]
            x   = x + y_i * (torch.pow(x, 2) - x)
        
        y = list(y) if isinstance(y, tuple) else y
        y = torch.cat(y, dim=1) if isinstance(y, list) else y
        return y, x
    
# endregion


# region Loss

class CombinedLoss01(nn.Loss):
    """Loss = SpatialConsistencyLoss
              + ExposureControlLoss
              + ColorConstancyLoss
              + IlluminationSmoothnessLoss
    """
    
    def __init__(
        self,
        spa_weight    : float = 1.0,
        exp_patch_size: int   = 16,
        exp_mean_val  : float = 0.6,
        exp_weight    : float = 10.0,
        col_weight    : float = 5.0,
        tv_weight     : float = 200.0,
        reduction     : str   = "mean",
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.spa_weight = spa_weight
        self.exp_weight = exp_weight
        self.col_weight = col_weight
        self.tv_weight  = tv_weight
        
        self.loss_spa = nn.SpatialConsistencyLoss(reduction=reduction)
        self.loss_exp = nn.ExposureControlLoss(
            reduction  = reduction,
            patch_size = exp_patch_size,
            mean_val   = exp_mean_val,
        )
        self.loss_col = nn.ColorConstancyLoss(reduction=reduction)
        self.loss_tv  = nn.IlluminationSmoothnessLoss(reduction=reduction)
    
    def __str__(self) -> str:
        return f"combined_loss"
    
    def forward(
        self,
        input : torch.Tensor | list[torch.Tensor],
        target: list[torch.Tensor],
        **_
    ) -> torch.Tensor:
        if isinstance(target, list | tuple):
            a       = target[-2]
            enhance = target[-1]
        else:
            raise TypeError()
        loss_spa = self.loss_spa(input=enhance, target=input)
        loss_exp = self.loss_exp(input=enhance)
        loss_col = self.loss_col(input=enhance)
        loss_tv  = self.loss_tv(input=a)
        loss     = self.spa_weight * loss_spa \
                   + self.exp_weight * loss_exp \
                   + self.col_weight * loss_col \
                   + self.tv_weight * loss_tv
        return loss


class CombinedLoss02(nn.Loss):
    """Loss = SpatialConsistencyLoss
              + ExposureControlLoss
              + ColorConstancyLoss
              + IlluminationSmoothnessLoss
              + ChannelConsistencyLoss
    """
    
    def __init__(
        self,
        spa_weight    : float = 1.0,
        exp_patch_size: int   = 16,
        exp_mean_val  : float = 0.6,
        exp_weight    : float = 10.0,
        col_weight    : float = 5.0,
        tv_weight     : float = 1600.0,
        channel_weight: float = 5.0,
        reduction     : str   = "mean",
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.spa_weight     = spa_weight
        self.exp_weight     = exp_weight
        self.col_weight     = col_weight
        self.tv_weight      = tv_weight
        self.channel_weight = channel_weight
        
        self.loss_spa     = nn.SpatialConsistencyLoss(reduction=reduction)
        self.loss_exp     = nn.ExposureControlLoss(
            reduction  = reduction,
            patch_size = exp_patch_size,
            mean_val   = exp_mean_val,
        )
        self.loss_col     = nn.ColorConstancyLoss(reduction=reduction)
        self.loss_tv      = nn.IlluminationSmoothnessLoss(reduction=reduction)
        self.loss_channel = nn.ChannelConsistencyLoss(reduction=reduction)
    
    def __str__(self) -> str:
        return f"combined_loss"
    
    def forward(
        self,
        input : torch.Tensor | list[torch.Tensor],
        target: list[torch.Tensor],
        **_
    ) -> torch.Tensor:
        if isinstance(target, list | tuple):
            a       = target[-2]
            enhance = target[-1]
        else:
            raise TypeError()
        loss_spa     = self.loss_spa(input=enhance, target=input)
        loss_exp     = self.loss_exp(input=enhance)
        loss_col     = self.loss_col(input=enhance)
        loss_tv      = self.loss_tv(input=a)
        loss_channel = self.loss_channel(input=enhance, target=input)
        loss = self.spa_weight * loss_spa \
               + self.exp_weight * loss_exp \
               + self.col_weight * loss_col \
               + self.tv_weight * loss_tv \
               + self.channel_weight * loss_channel
        return loss
    
# endregion


# region Zero-DCE

@MODELS.register(name="zero-dce")
@MODELS.register(name="zerodce")
class ZeroDCE(base.LowLightImageEnhancementModel):
    """Zero-DCE (Zero-Reference Deep Curve Estimation) model.

    References:
        `<https://github.com/Li-Chongyi/Zero-DCE>`__

    See Also: :class:`mon.vision.enhance.llie.base.LowLightImageEnhancementModel`
    """
    
    zoo = {
        "zerodce-sice": {
            "path"        : "best.pth",
            "num_channels": 32,
        },
    }

    def __init__(
        self,
        num_channels: int = 32,
        num_iters   : int = 8,
        *args, **kwargs
    ):
        super().__init__(loss=CombinedLoss01(), *args, **kwargs)
        assert num_iters <= 8
        self.num_channels = num_channels
        self.num_iters    = num_iters

        self.relu     = nn.ReLU(inplace=True)
        self.e_conv1  = nn.Conv2d(3,          self.num_channels, 3, 1, 1, bias=True)
        self.e_conv2  = nn.Conv2d(self.num_channels,     self.num_channels, 3, 1, 1, bias=True)
        self.e_conv3  = nn.Conv2d(self.num_channels,     self.num_channels, 3, 1, 1, bias=True)
        self.e_conv4  = nn.Conv2d(self.num_channels,     self.num_channels, 3, 1, 1, bias=True)
        self.e_conv5  = nn.Conv2d(self.num_channels * 2, self.num_channels, 3, 1, 1, bias=True)
        self.e_conv6  = nn.Conv2d(self.num_channels * 2, self.num_channels, 3, 1, 1, bias=True)
        self.e_conv7  = nn.Conv2d(self.num_channels * 2, 24,    3, 1, 1, bias=True)
        self.maxpool  = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.apply(self.init_weights)

    def init_weights(self, m: nn.Module):
        """Initialize model's weights."""
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            if hasattr(m, "conv"):
                m.conv.weight.data.normal_(0.0, 0.02)
            else:
                m.weight.data.normal_(0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    def forward_loss(
        self,
        input : torch.Tensor,
        target: torch.Tensor | None,
        *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass with loss value. Loss function may need more arguments
        beside the ground-truth and prediction values. For calculating the
        metrics, we only need the final predictions and ground-truth.

        Args:
            input: An input of shape :math:`[N, C, H, W]`.
            target: A ground-truth of shape :math:`[N, C, H, W]`. Default: ``None``.
            
        Return:
            Predictions and loss value.
        """
        pred = self.forward(input=input, *args, **kwargs)
        loss = self.loss(input, pred) if self.loss else None
        return pred[-1], loss

    def forward_once(
        self,
        input    : torch.Tensor,
        profile  : bool = False,
        out_index: int = -1,
        *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass once. Implement the logic for a single forward pass.

        Args:
            input: An input of shape :math:`[N, C, H, W]`.
            profile: Measure processing time. Default: ``False``.
            out_index: Return specific layer's output from :param:`out_index`.
                Default: ``-1`` means the last layer.

        Return:
            Predictions.
        """
        x   = input
        x1  = self.relu(self.e_conv1(x))
        x2  = self.relu(self.e_conv2(x1))
        x3  = self.relu(self.e_conv3(x2))
        x4  = self.relu(self.e_conv4(x3))
        x5  = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6  = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        x_r = torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))

        x_rs = torch.split(x_r, 3, dim=1)
        y    = x
        for i in range(0, self.num_iters):
            y = y + x_rs[i] * (torch.pow(y, 2) - y)

        return x_r, y

# endregion


# region Zero-DCE++

@MODELS.register(name="zero-dce++")
@MODELS.register(name="zerodce++")
class ZeroDCEPP(base.LowLightImageEnhancementModel):
    """Zero-DCE++ (Zero-Reference Deep Curve Estimation) model.

    References:
        `<https://github.com/Li-Chongyi/Zero-DCE_extension>`__

    See Also: :class:`mon.vision.enhance.llie.base.LowLightImageEnhancementModel`
    """

    zoo = {
        "zerodce-sice" : {
            "path"        : "best.pth",
            "num_channels": 32,
            "map": {
                "depth_conv": "dw_conv",
                "point_conv": "pw_conv",
            }
        },
    }

    def __init__(
        self,
        num_channels: int   = 32,
        num_iters   : int   = 8,
        scale_factor: float = 1.0,
        *args, **kwargs
    ):
        super().__init__(loss=CombinedLoss01(), *args, **kwargs)
        assert num_iters <= 8
        self.num_channels = num_channels
        self.num_iters    = num_iters
        self.scale_factor = scale_factor

        self.relu     = nn.ReLU(inplace=True)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=self.scale_factor)
        self.e_conv1  = nn.DSConv2d(in_channels=3,                     out_channels=self.num_channels, kernel_size=3, dw_stride=1, dw_padding=1, groups=1)
        self.e_conv2  = nn.DSConv2d(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, dw_stride=1, dw_padding=1, groups=1)
        self.e_conv3  = nn.DSConv2d(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, dw_stride=1, dw_padding=1, groups=1)
        self.e_conv4  = nn.DSConv2d(in_channels=self.num_channels,     out_channels=self.num_channels, kernel_size=3, dw_stride=1, dw_padding=1, groups=1)
        self.e_conv5  = nn.DSConv2d(in_channels=self.num_channels * 2, out_channels=self.num_channels, kernel_size=3, dw_stride=1, dw_padding=1, groups=1)
        self.e_conv6  = nn.DSConv2d(in_channels=self.num_channels * 2, out_channels=self.num_channels, kernel_size=3, dw_stride=1, dw_padding=1, groups=1)
        self.e_conv7  = nn.DSConv2d(in_channels=self.num_channels * 2, out_channels=3,                 kernel_size=3, dw_stride=1, dw_padding=1, groups=1)
        self.apply(self.init_weights)

    def init_weights(self, m: torch.nn.Module):
        """Initialize model's weights."""
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            if hasattr(m, "conv"):
                m.conv.weight.data.normal_(0.0, 0.02)
            elif hasattr(m, "dw_conv"):
                m.dw_conv.weight.data.normal_(0.0, 0.02)
            elif hasattr(m, "pw_conv"):
                m.pw_conv.weight.data.normal_(0.0, 0.02)
            else:
                m.weight.data.normal_(0.0, 0.02)

    def forward_loss(
        self,
        input : torch.Tensor,
        target: torch.Tensor | None,
        *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass with loss value. Loss function may need more arguments
        beside the ground-truth and prediction values. For calculating the
        metrics, we only need the final predictions and ground-truth.

        Args:
            input: An input of shape :math:`[N, C, H, W]`.
            target: A ground-truth of shape :math:`[N, C, H, W]`. Default: ``None``.
            
        Return:
            Predictions and loss value.
        """
        pred = self.forward(input=input, *args, **kwargs)
        loss = self.loss(input, pred) if self.loss else None
        return pred[-1], loss

    def forward_once(
        self,
        input    : torch.Tensor,
        profile  : bool = False,
        out_index: int = -1,
        *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass once. Implement the logic for a single forward pass.

        Args:
            input: An input of shape :math:`[N, C, H, W]`.
            profile: Measure processing time. Default: ``False``.
            out_index: Return specific layer's output from :param:`out_index`.
                Default: ``-1`` means the last layer.

        Return:
            Predictions.
        """
        x = input
        if self.scale_factor == 1:
            x_down = x
        else:
            x_down = F.interpolate(x, scale_factor=1 / self.scale_factor, mode="bilinear")

        x1  = self.relu(self.e_conv1(x_down))
        x2  = self.relu(self.e_conv2(x1))
        x3  = self.relu(self.e_conv3(x2))
        x4  = self.relu(self.e_conv4(x3))
        x5  = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6  = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        x_r = torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
        if self.scale_factor == 1:
            x_r = x_r
        else:
            x_r = self.upsample(x_r)

        y = x
        for i in range(0, self.num_iters):
            y = y + x_r * (torch.pow(y, 2) - y)
        return x_r, y

# endregion

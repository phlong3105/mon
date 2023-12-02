#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements Zero-DCE, Zero-DCE++, and Zero-DCE-Tiny models."""

from __future__ import annotations

__all__ = [
    "DCE",
    "PixelwiseHigherOrderLECurve",
    "ZeroDCE",
    "ZeroDCEPP",
    "ZeroDCEPPVanilla",
    "ZeroDCETiny",
    "ZeroDCEVanilla",
]

from typing import Any

import torch

from mon.globals import LAYERS, MODELS
from mon.nn.typing import _size_2_t
from mon.vision import core, nn
from mon.vision.enhance.llie import base
from mon.vision.nn import functional as F

console      = core.console
_current_dir = core.Path(__file__).absolute().parent


# region Layer

@LAYERS.register()
class DCE(nn.ConvLayerParsingMixin, nn.Module):
    
    def __init__(
        self,
        in_channels : int       = 3,
        out_channels: int       = 24,
        mid_channels: int       = 32,
        kernel_size : _size_2_t = 3,
        stride      : _size_2_t = 1,
        padding     : _size_2_t = 1,
        dilation    : _size_2_t = 1,
        groups      : int       = 1,
        bias        : bool      = True,
        padding_mode: str       = "zeros",
        device      : Any       = None,
        dtype       : Any       = None,
    ):
        super().__init__()
        self.relu  = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            in_channels  = in_channels,
            out_channels = mid_channels,
            kernel_size  = kernel_size,
            stride       = stride,
            padding      = padding,
            dilation     = dilation,
            groups       = groups,
            bias         = bias,
            padding_mode = padding_mode,
            device       = device,
            dtype        = dtype,
        )
        self.conv2 = nn.Conv2d(
            in_channels  = mid_channels,
            out_channels = mid_channels,
            kernel_size  = kernel_size,
            stride       = stride,
            padding      = padding,
            dilation     = dilation,
            groups       = groups,
            bias         = bias,
            padding_mode = padding_mode,
            device       = device,
            dtype        = dtype,
        )
        self.conv3 = nn.Conv2d(
            in_channels  = mid_channels,
            out_channels = mid_channels,
            kernel_size  = kernel_size,
            stride       = stride,
            padding      = padding,
            dilation     = dilation,
            groups       = groups,
            bias         = bias,
            padding_mode = padding_mode,
            device       = device,
            dtype        = dtype,
        )
        self.conv4 = nn.Conv2d(
            in_channels  = mid_channels,
            out_channels = mid_channels,
            kernel_size  = kernel_size,
            stride       = stride,
            padding      = padding,
            dilation     = dilation,
            groups       = groups,
            bias         = bias,
            padding_mode = padding_mode,
            device       = device,
            dtype        = dtype,
        )
        self.conv5 = nn.Conv2d(
            in_channels  = mid_channels * 2,
            out_channels = mid_channels,
            kernel_size  = kernel_size,
            stride       = stride,
            padding      = padding,
            dilation     = dilation,
            groups       = groups,
            bias         = bias,
            padding_mode = padding_mode,
            device       = device,
            dtype        = dtype,
        )
        self.conv6 = nn.Conv2d(
            in_channels  = mid_channels * 2,
            out_channels = mid_channels,
            kernel_size  = kernel_size,
            stride       = stride,
            padding      = padding,
            dilation     = dilation,
            groups       = groups,
            bias         = bias,
            padding_mode = padding_mode,
            device       = device,
            dtype        = dtype,
        )
        self.conv7 = nn.Conv2d(
            in_channels  = mid_channels * 2,
            out_channels = out_channels,
            kernel_size  = kernel_size,
            stride       = stride,
            padding      = padding,
            dilation     = dilation,
            groups       = groups,
            bias         = bias,
            padding_mode = padding_mode,
            device       = device,
            dtype        = dtype,
        )
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x  = input
        y1 = self.relu(self.conv1(x))
        y2 = self.relu(self.conv2(y1))
        y3 = self.relu(self.conv3(y2))
        y4 = self.relu(self.conv4(y3))
        y5 = self.relu(self.conv5(torch.cat([y3, y4], dim=1)))
        y6 = self.relu(self.conv6(torch.cat([y2, y5], dim=1)))
        y  = torch.tanh(self.conv7(torch.cat([y1, y6], dim=1)))
        return y


@LAYERS.register()
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

@MODELS.register(name="zerodce")
class ZeroDCE(base.LowLightImageEnhancementModel):
    """Zero-DCE (Zero-Reference Deep Curve Estimation) model.
    
    See Also: :class:`mon.vision.enhance.llie.base.LowLightImageEnhancementModel`
    """
    
    configs     = {}
    zoo         = {
        "zerodce-lol": {
            "name"       : "lol",
            "path"       : "",
            "file_name"  : "zerodce-lol.pth",
            "num_classes": None,
        },
        "zerodce-sice": {
            "name"       : "sice",
            "path"       : "",
            "file_name"  : "zerodce-sice.pth",
            "num_classes": None,
        },
    }
    map_weights = {
        "backbone": {
            "1.weight" : "e_conv1.weight",
            "1.bias"   : "e_conv1.bias",
            "3.weight" : "e_conv2.weight",
            "3.bias"   : "e_conv2.bias",
            "5.weight" : "e_conv3.weight",
            "5.bias"   : "e_conv3.bias",
            "7.weight" : "e_conv4.weight",
            "7.bias"   : "e_conv4.bias",
            "10.weight": "e_conv5.weight",
            "10.bias"  : "e_conv5.bias",
            "13.weight": "e_conv6.weight",
            "13.bias"  : "e_conv6.bias",
            "16.weight": "e_conv7.weight",
            "16.bias"  : "e_conv7.bias",
        },
        "head"    : {},
    }
    
    def __init__(
        self,
        config: Any = "zerodce.yaml",
        loss  : Any = CombinedLoss01(),
        *args, **kwargs
    ):
        super().__init__(
            config = config,
            loss   = loss,
            *args, **kwargs
        )
    
    def init_weights(self, m: torch.nn.Module):
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
    
    def load_weights(self):
        """Load weights. It only loads the intersection layers of matching keys
        and shapes between the current model and weights.
        """
        if isinstance(self.weights, dict) \
            and self.weights["name"] in ["sice"]:
            state_dict = nn.load_state_dict_from_path(
                model_dir=self.zoo_dir, **self.weights
            )
            state_dict       = state_dict["params"]
            model_state_dict = self.model.state_dict()
            """
            for k in self.model.state_dict().keys():
                print(f"\"{k}\": ")
            for k in state_dict.keys():
                print(f"\"{k}\"")
            """
            for k, v in self.map_weights["backbone"].items():
                model_state_dict[k] = state_dict[v]
            if self.weights["num_classes"] == self.num_classes:
                for k, v in self.map_weights["head"].items():
                    model_state_dict[k] = state_dict[v]
            self.model.load_state_dict(model_state_dict)
        else:
            super().load_weights()
    
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


@MODELS.register(name="zerodce-vanilla")
class ZeroDCEVanilla(nn.Module):
    """Original implementation of Zero-DCE.
    
    References:
        `<https://github.com/Li-Chongyi/Zero-DCE>`__
    """
    
    def __init__(self):
        super().__init__()
        number_f   = 32
        self.relu  = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(3,            number_f, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(number_f,     number_f, 3, 1, 1, bias=True)
        self.conv3 = nn.Conv2d(number_f,     number_f, 3, 1, 1, bias=True)
        self.conv4 = nn.Conv2d(number_f,     number_f, 3, 1, 1, bias=True)
        self.conv5 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
        self.conv6 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
        self.conv7 = nn.Conv2d(number_f * 2, 24,       3, 1, 1, bias=True)
    
    def enhance(self, x: torch.Tensor, x_r: torch.Tensor) -> torch.Tensor:
        x = x + x_r * (torch.pow(x, 2) - x)
        x = x + x_r * (torch.pow(x, 2) - x)
        x = x + x_r * (torch.pow(x, 2) - x)
        x = x + x_r * (torch.pow(x, 2) - x)
        x = x + x_r * (torch.pow(x, 2) - x)
        x = x + x_r * (torch.pow(x, 2) - x)
        x = x + x_r * (torch.pow(x, 2) - x)
        x = x + x_r * (torch.pow(x, 2) - x)
        return x
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x   = input
        x1  = self.relu(self.conv1(x))
        x2  = self.relu(self.conv2(x1))
        x3  = self.relu(self.conv3(x2))
        x4  = self.relu(self.conv4(x3))
        x5  = self.relu(self.conv5(torch.cat([x3, x4], 1)))
        x6  = self.relu(self.conv6(torch.cat([x2, x5], 1)))
        x_r = torch.tanh(self.conv7(torch.cat([x1, x6], 1)))
        r1, r2, r3, r4, r5, r6, r7, r8 = torch.split(x_r, 3, dim=1)
        x = x + r1 * (torch.pow(x, 2) - x)
        x = x + r2 * (torch.pow(x, 2) - x)
        x = x + r3 * (torch.pow(x, 2) - x)
        x = x + r4 * (torch.pow(x, 2) - x)
        x = x + r5 * (torch.pow(x, 2) - x)
        x = x + r6 * (torch.pow(x, 2) - x)
        x = x + r7 * (torch.pow(x, 2) - x)
        x = x + r8 * (torch.pow(x, 2) - x)
        r = torch.cat([r1, r2, r3, r4, r5, r6, r7, r8], 1)
        return x

# endregion


# region Zero-DCE++

@MODELS.register(name="zerodce++")
class ZeroDCEPP(base.LowLightImageEnhancementModel):
    """Zero-DCE++ (Zero-Reference Deep Curve Estimation) model.
    
    See Also: :class:`mon.vision.enhance.llie.base.LowLightImageEnhancementModel`
    """
    
    configs     = {}
    zoo         = {
        "zerodce++-lol" : {
            "name"       : "lol",
            "path"       : "",
            "file_name"  : "zerodce++-lol.pth",
            "num_classes": None,
        },
        "zerodce++-sice": {
            "name"       : "sice",
            "path"       : "",
            "file_name"  : "zerodce++-sice.pth",
            "num_classes": None,
        },
    }
    map_weights = {
        "backbone": {
            "2.dw_conv.weight" : "e_conv1.depth_conv.weight",
            "2.dw_conv.bias"   : "e_conv1.depth_conv.bias",
            "2.pw_conv.weight" : "e_conv1.point_conv.weight",
            "2.pw_conv.bias"   : "e_conv1.point_conv.bias",
            "4.dw_conv.weight" : "e_conv2.depth_conv.weight",
            "4.dw_conv.bias"   : "e_conv2.depth_conv.bias",
            "4.pw_conv.weight" : "e_conv2.point_conv.weight",
            "4.pw_conv.bias"   : "e_conv2.point_conv.bias",
            "6.dw_conv.weight" : "e_conv3.depth_conv.weight",
            "6.dw_conv.bias"   : "e_conv3.depth_conv.bias",
            "6.pw_conv.weight" : "e_conv3.point_conv.weight",
            "6.pw_conv.bias"   : "e_conv3.point_conv.bias",
            "8.dw_conv.weight" : "e_conv4.depth_conv.weight",
            "8.dw_conv.bias"   : "e_conv4.depth_conv.bias",
            "8.pw_conv.weight" : "e_conv4.point_conv.weight",
            "8.pw_conv.bias"   : "e_conv4.point_conv.bias",
            "11.dw_conv.weight": "e_conv5.depth_conv.weight",
            "11.dw_conv.bias"  : "e_conv5.depth_conv.bias",
            "11.pw_conv.weight": "e_conv5.point_conv.weight",
            "11.pw_conv.bias"  : "e_conv5.point_conv.bias",
            "14.dw_conv.weight": "e_conv6.depth_conv.weight",
            "14.dw_conv.bias"  : "e_conv6.depth_conv.bias",
            "14.pw_conv.weight": "e_conv6.point_conv.weight",
            "14.pw_conv.bias"  : "e_conv6.point_conv.bias",
            "17.dw_conv.weight": "e_conv7.depth_conv.weight",
            "17.dw_conv.bias"  : "e_conv7.depth_conv.bias",
            "17.pw_conv.weight": "e_conv7.point_conv.weight",
            "17.pw_conv.bias"  : "e_conv7.point_conv.bias",
        },
        "head"    : {},
    }
    
    def __init__(
        self,
        config: Any = "zerodce++.yaml",
        loss  : Any = CombinedLoss01(),
        *args, **kwargs
    ):
        super().__init__(
            config = config,
            loss   = loss,
            *args, **kwargs
        )
    
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
    
    def load_weights(self):
        """Load weights. It only loads the intersection layers of matching keys
        and shapes between the current model and weights.
        """
        if isinstance(self.weights, dict) \
            and self.weights["name"] in ["sice"]:
            state_dict = nn.load_state_dict_from_path(
                model_dir=self.zoo_dir, **self.weights
            )
            state_dict       = state_dict["params"]
            model_state_dict = self.model.state_dict()
            """
            for k in self.model.state_dict().keys():
                print(f"\"{k}\": ")
            for k in state_dict.keys():
                print(f"\"{k}\"")
            """
            for k, v in self.map_weights["backbone"].items():
                model_state_dict[k] = state_dict[v]
            if self.weights["num_classes"] == self.num_classes:
                for k, v in self.map_weights["head"].items():
                    model_state_dict[k] = state_dict[v]
            self.model.load_state_dict(model_state_dict)
        else:
            super().load_weights()
    
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


@MODELS.register(name="zerodce++-vanilla")
class ZeroDCEPPVanilla(nn.Module):
    """Original implementation of ZeroDCE++.
    
    References:
        `<https://github.com/Li-Chongyi/Zero-DCE_extension>`__
    """
    
    def __init__(self, scale_factor: float = 1.0):
        super().__init__()
        number_f          = 32
        self.relu         = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor
        self.upsample     = nn.UpsamplingBilinear2d(scale_factor=self.scale_factor)
        self.conv1        = nn.DSConv2d(in_channels=3,            out_channels=number_f, kernel_size=3, dw_stride=1, dw_padding=1, groups=1)
        self.conv2        = nn.DSConv2d(in_channels=number_f,     out_channels=number_f, kernel_size=3, dw_stride=1, dw_padding=1, groups=1)
        self.conv3        = nn.DSConv2d(in_channels=number_f,     out_channels=number_f, kernel_size=3, dw_stride=1, dw_padding=1, groups=1)
        self.conv4        = nn.DSConv2d(in_channels=number_f,     out_channels=number_f, kernel_size=3, dw_stride=1, dw_padding=1, groups=1)
        self.conv5        = nn.DSConv2d(in_channels=number_f * 2, out_channels=number_f, kernel_size=3, dw_stride=1, dw_padding=1, groups=1)
        self.conv6        = nn.DSConv2d(in_channels=number_f * 2, out_channels=number_f, kernel_size=3, dw_stride=1, dw_padding=1, groups=1)
        self.conv7        = nn.DSConv2d(in_channels=number_f * 2, out_channels=3,        kernel_size=3, dw_stride=1, dw_padding=1, groups=1)
    
    def enhance(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        x = x + a * (torch.pow(x, 2) - x)
        x = x + a * (torch.pow(x, 2) - x)
        x = x + a * (torch.pow(x, 2) - x)
        x = x + a * (torch.pow(x, 2) - x)
        x = x + a * (torch.pow(x, 2) - x)
        x = x + a * (torch.pow(x, 2) - x)
        x = x + a * (torch.pow(x, 2) - x)
        x = x + a * (torch.pow(x, 2) - x)
        return x
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        if self.scale_factor == 1:
            x_down = x
        else:
            x_down = F.interpolate(x, scale_factor=1 / self.scale_factor, mode="bilinear")
        
        x1  = self.relu(self.conv1(x_down))
        x2  = self.relu(self.conv2(x1))
        x3  = self.relu(self.conv3(x2))
        x4  = self.relu(self.conv4(x3))
        x5  = self.relu(self.conv5(torch.cat([x3, x4], 1)))
        x6  = self.relu(self.conv6(torch.cat([x2, x5], 1)))
        a = F.tanh(self.conv7(torch.cat([x1, x6], 1)))
        if self.scale_factor == 1:
            a = a
        else:
            a = self.upsample(a)
        x = self.enhance(x, a)
        return x

# endregion


# region Zero-DCE-Tiny

@MODELS.register(name="zerodce-tiny")
class ZeroDCETiny(base.LowLightImageEnhancementModel):
    """Zero-DCE-Tiny (Zero-Reference Deep Curve Estimation) model.
    
    See Also: :class:`mon.vision.enhance.llie.base.LowLightImageEnhancementModel`
    """
    
    configs     = {}
    zoo         = {}
    map_weights = {}
    
    def __init__(
        self,
        config: Any = "zerodce-tiny.yaml",
        loss  : Any = CombinedLoss02(),
        *args, **kwargs
    ):
        super().__init__(
            config = config,
            loss   = loss,
            *args, **kwargs
        )
    
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

# endregion

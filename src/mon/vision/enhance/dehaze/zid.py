#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements ZID models."""

from __future__ import annotations

__all__ = [
    "ZID",
]

from typing import Any, Literal

import numpy as np
import torch

import mon
from mon.globals import MODELS
from mon.vision import core, nn
from mon.vision.enhance.dehaze import base

math         = core.math
console      = core.console
_current_dir = core.Path(__file__).absolute().parent


# region Module

def conv(
    in_channels    : int,
    out_channels   : int,
    kernel_size    : int,
    stride         : int  = 1,
    bias           : bool = True,
    padding        : str  = "zero",
    downsample_mode: str  = "stride",
):
    downsampler = None
    if stride != 1 and downsample_mode != "stride":
        if downsample_mode == "avg":
            downsampler = nn.AvgPool2d(stride, stride)
        elif downsample_mode == "max":
            downsampler = nn.MaxPool2d(stride, stride)
        elif downsample_mode in ["lanczos2", "lanczos3"]:
            downsampler = nn.CustomDownsample(
                in_channels   = out_channels,
                factor        = stride,
                kernel_type   = downsample_mode,
                phase         = 0.5,
                preserve_size = True,
            )
        stride = 1

    padder = None
    to_pad = int((kernel_size - 1) / 2)
    if padding == "reflection":
        padder = nn.ReflectionPad2d(to_pad)
        to_pad = 0

    convolver = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=to_pad, bias=bias)
    layers    = [x for x in [padder, convolver, downsampler] if x is not None]
    return nn.Sequential(*layers)


def encoder_decoder_skip(
    in_channels     : int          = 2,
    out_channels    : int          = 3,
    channels_down   : list | tuple = [16, 32, 64, 128, 128],
    channels_up     : list | tuple = [16, 32, 64, 128, 128],
    channels_skip   : list | tuple = [4 , 4,  4,  4,   4],
    kernel_size_down: int          = 3,
    kernel_size_up  : int          = 3,
    kernel_size_skip: int          = 1,
    padding         : str          = "zero",
    bias            : bool         = True,
    upsample_mode   : str          = "nearest",
    downsample_mode : str          = "stride",
    up_1x1          : bool         = True,
    sigmoid         : bool         = True,
    act             : Any          = nn.LeakyReLU,
) -> nn.Module:
    """Encoder-decoder network with skip connections.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        channels_down: List of channels for downsampling layers.
        channels_up: List of channels for upsampling layers.
        channels_skip: List of channels for skip connections.
        kernel_size_down: Kernel size for downsampling layers.
        kernel_size_up: Kernel size for upsampling layers.
        kernel_size_skip: Kernel size for skip connection layers.
        padding: Padding mode. One of: ``'zero'`` or ``'reflect'``.
            Default: ``'zero'``.
        bias: Whether to use bias or not.
        upsample_mode: Upsampling mode. One of: ``'nearest'`` or ``'bilinear'``.
            Default: ``'nearest'``.
        downsample_mode: Downsampling mode. One of: ``'stride'``, ``'avg'``,
            ``'max'``, or ``'lanczos2'``. Default: ``'stride'``.
        up_1x1: Whether to use 1x1 convolution.
        sigmoid: Whether to use sigmoid function.
        act: Activation layer.
    """
    assert len(channels_down) == len(channels_up) == len(channels_skip)

    n_scales = len(channels_down)
    if not (isinstance(upsample_mode, list)    or isinstance(upsample_mode, tuple)):
        upsample_mode    = [upsample_mode]    * n_scales
    if not (isinstance(downsample_mode, list)  or isinstance(downsample_mode, tuple)):
        downsample_mode  = [downsample_mode]  * n_scales
    if not (isinstance(kernel_size_down, list) or isinstance(kernel_size_down, tuple)):
        kernel_size_down = [kernel_size_down] * n_scales
    if not (isinstance(kernel_size_up, list)   or isinstance(kernel_size_up, tuple)):
        kernel_size_up   = [kernel_size_up]   * n_scales

    last_scale  = n_scales - 1
    cur_depth   = None
    input_depth = in_channels
    model       = nn.Sequential()
    model_tmp   = model

    for i in range(len(channels_down)):
        deeper = nn.Sequential()
        skip   = nn.Sequential()

        if channels_skip[i] != 0:
            model_tmp.add(nn.CustomConcat(1, skip, deeper))
        else:
            model_tmp.add(deeper)

        model_tmp.add(nn.BatchNorm2d(channels_skip[i] + (channels_up[i + 1] if i < last_scale else channels_down[i])))

        if channels_skip[i] != 0:
            skip.add(conv(input_depth, channels_skip[i], kernel_size_skip, bias=bias, padding=padding))
            skip.add(nn.BatchNorm2d(channels_skip[i]))
            skip.add(act())

        deeper.add(conv(input_depth, channels_down[i], kernel_size_down[i], stride=2, bias=bias, padding=padding, downsample_mode=downsample_mode[i]))
        deeper.add(nn.BatchNorm2d(channels_down[i]))
        deeper.add(act())

        deeper.add(conv(channels_down[i], kernel_size_down[i], kernel_size_down[i], bias=bias, padding=padding))
        deeper.add(nn.BatchNorm2d(channels_down[i]))
        deeper.add(act())

        deeper_main = nn.Sequential()

        if i == len(channels_down) - 1:
            # The deepest
            k = channels_down[i]
        else:
            deeper.add(deeper_main)
            k = channels_up[i + 1]

        deeper.add(nn.Upsample(scale_factor=2, mode=upsample_mode[i], align_corners=True))

        model_tmp.add(conv(channels_skip[i] + k, channels_up[i], kernel_size_up[i], stride=1, bias=bias, padding=padding))
        model_tmp.add(nn.BatchNorm2d(channels_up[i]))
        model_tmp.add(act())

        if up_1x1:
            model_tmp.add(conv(channels_up[i], channels_up[i], kernel_size=1, bias=bias, padding=padding))
            model_tmp.add(nn.BatchNorm2d(channels_up[i]))
            model_tmp.add(act())

        input_depth = channels_down[i]
        model_tmp   = deeper_main

    model.add(conv(channels_up[0], out_channels, kernel_size=1, bias=bias, padding=padding))
    if sigmoid:
        model.add(nn.Sigmoid())

    return model

# endregion


# region Loss

class CustomLoss(nn.Loss):

    def __init__(
        self,
        bri_gamma      : float = 2.8,
        exp_patch_size : int   = 16,
        exp_mean_val   : float = 0.6,
        spa_num_regions: Literal[4, 8, 16, 24] = 4,  # 4
        spa_patch_size : int   = 4,     # 4
        weight_bri     : float = 1,
        weight_col     : float = 5,
        weight_crl     : float = 1,     # 20
        weight_edge    : float = 5,
        weight_exp     : float = 10,
        weight_kl      : float = 5,     # 5
        weight_spa     : float = 1,
        weight_tvA     : float = 1600,  # 200
        reduction      : str   = "mean",
        verbose        : bool  = False,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.weight_bri  = weight_bri
        self.weight_col  = weight_col
        self.weight_crl  = weight_crl
        self.weight_edge = weight_edge
        self.weight_exp  = weight_exp
        self.weight_kl   = weight_kl
        self.weight_spa  = weight_spa
        self.weight_tvA  = weight_tvA
        self.verbose     = verbose

        self.loss_bri  = nn.BrightnessConstancyLoss(reduction=reduction, gamma=bri_gamma)
        self.loss_col  = nn.ColorConstancyLoss(reduction=reduction)
        self.loss_crl  = nn.ChannelRatioConsistencyLoss(reduction=reduction)
        self.loss_kl   = nn.ChannelConsistencyLoss(reduction=reduction)
        self.loss_edge = nn.EdgeConstancyLoss(reduction=reduction)
        self.loss_exp  = nn.ExposureControlLoss(
            reduction  = reduction,
            patch_size = exp_patch_size,
            mean_val   = exp_mean_val,
        )
        self.loss_spa  = nn.SpatialConsistencyLoss(
            num_regions = spa_num_regions,
            patch_size  = spa_patch_size,
            reduction   = reduction,
        )
        self.loss_tvA  = nn.IlluminationSmoothnessLoss(reduction=reduction)

    def __str__(self) -> str:
        return f"zero-reference loss"

    def forward(
        self,
        input   : torch.Tensor | list[torch.Tensor],
        target  : list[torch.Tensor],
        previous: torch.Tensor = None,
        **_
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(target, list | tuple):
            if len(target) == 2:
                a       = target[-2]
                enhance = target[-1]
            elif len(target) == 3:
                a       = target[-3]
                g       = target[-2]
                enhance = target[-1]
        else:
            raise TypeError
        loss_bri  = self.loss_bri(input=g, target=input)              if self.weight_bri  > 0 else 0
        loss_col  = self.loss_col(input=enhance)                      if self.weight_col  > 0 else 0
        loss_edge = self.loss_edge(input=enhance, target=input)       if self.weight_edge > 0 else 0
        loss_exp  = self.loss_exp(input=enhance)                      if self.weight_exp  > 0 else 0
        loss_kl   = self.loss_kl(input=enhance, target=input)         if self.weight_kl   > 0 else 0
        loss_spa  = self.loss_spa(input=enhance, target=input)        if self.weight_spa  > 0 else 0
        loss_tvA  = self.loss_tvA(input=a)                            if self.weight_tvA  > 0 else 0
        if previous is not None and (enhance.shape == previous.shape):
            loss_crl = self.loss_crl(input=enhance, target=previous)  if self.weight_crl  > 0 else 0
        else:
            loss_crl = self.loss_crl(input=enhance, target=input)     if self.weight_crl  > 0 else 0

        loss = (
              self.weight_bri  * loss_bri
            + self.weight_col  * loss_col
            + self.weight_crl  * loss_crl
            + self.weight_edge * loss_edge
            + self.weight_exp  * loss_exp
            + self.weight_tvA  * loss_tvA
            + self.weight_kl   * loss_kl
            + self.weight_spa  * loss_spa
        )

        if self.verbose:
            console.log(f"{self.loss_bri.__str__():<30} : {loss_bri}")
            console.log(f"{self.loss_col.__str__():<30} : {loss_col}")
            console.log(f"{self.loss_edge.__str__():<30}: {loss_edge}")
            console.log(f"{self.loss_exp.__str__():<30} : {loss_exp}")
            console.log(f"{self.loss_kl.__str__():<30}  : {loss_kl}")
            console.log(f"{self.loss_spa.__str__():<30} : {loss_spa}")
            console.log(f"{self.loss_tvA.__str__():<30} : {loss_tvA}")
        return loss, enhance

# endregion


# region Model

@MODELS.register(name="zid")
class ZID(base.DehazingModel):
    """ZID (Zero-Shot Image Dehazing) model.
    
    See Also: :class:`mon.vision.enhance.les.base.Dehazing`
    """
    
    configs     = {}
    zoo         = {}
    map_weights = {}
    
    def __init__(
        self,
        config : Any        = None,
        loss   : Any        = CustomLoss(),
        variant: str | None = None,
        *args, **kwargs
    ):
        super().__init__(
            config = config,
            loss   = loss,
            *args, **kwargs
        )
        variant           = mon.to_int(variant)
        self.variant      = f"{variant:04d}" if isinstance(variant, int) else None
        self.out_channels = 3

        # Image Network
        self.image_net   = encoder_decoder_skip(
            in_channels   = 3,
            out_channels  = 3,
            channels_down = [8, 16, 32, 64, 128],
            channels_up   = [8, 16, 32, 64, 128],
            channels_skip = [0, 0 , 0 , 4 , 4  ],
            padding       = "reflection",
            bias          = True,
            upsample_mode = "bilinear",
            sigmoid       = True,
            act           = nn.LeakyReLU
        ).type(torch.cuda.FloatTensor)

        # Mask Network
        self.mask_net    = encoder_decoder_skip(
            in_channels   = 3,
            out_channels  = 1,
            channels_down = [8, 16, 32, 64, 128],
            channels_up   = [8, 16, 32, 64, 128],
            channels_skip = [0, 0 , 0 , 4 , 4  ],
            padding       = "reflection",
            bias          = True,
            upsample_mode = "bilinear",
            sigmoid       = True,
            act           = nn.LeakyReLU
        ).type(torch.cuda.FloatTensor)

        # Ambient Network
        self.ambient_net = None

    def init_weights(self, m: nn.Module):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            if hasattr(m, "conv"):
                m.conv.weight.data.normal_(0.0, 0.02)  # 0.02
            elif hasattr(m, "dw_conv"):
                m.dw_conv.weight.data.normal_(0.0, 0.02)  # 0.02
            elif hasattr(m, "pw_conv"):
                m.pw_conv.weight.data.normal_(0.0, 0.02)  # 0.02
            elif hasattr(m, "weight"):
                m.weight.data.normal_(0.0, 0.02)  # 0.02
    
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
        pred  = self.forward(input=input, *args, **kwargs)
        loss, self.previous = self.loss(input, pred, self.previous) if self.loss else (None, None)
        loss += self.regularization_loss(alpha=0.1)
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
        pass

# endregion

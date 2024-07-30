#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements ZID (Zero-Shot Image Dehazing) models."""

from __future__ import annotations

__all__ = [
    "ZID",
]

from typing import Any

import numpy as np
import torch
from cv2.ximgproc import guidedFilter

from mon import core, nn
from mon.core import _callable, _size_2_t
from mon.globals import MODELS, Scheme
from mon.vision.enhance.dehaze import base

console = core.console


# region Module

def add_module(self, module_):
    self.add_module(str(len(self) + 1), module_)


nn.Module.add = add_module


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
    in_channels      : int          = 2,
    out_channels     : int          = 3,
    channels_down    : list | tuple = [16, 32, 64, 128, 128],
    channels_up      : list | tuple = [16, 32, 64, 128, 128],
    channels_skip    : list | tuple = [4 , 4 , 4 , 4  , 4],
    kernel_size_down : int          = 3,
    kernel_size_up   : int          = 3,
    kernel_size_skip : int          = 1,
    padding          : str          = "zero",
    bias             : bool         = True,
    upsample_mode    : str          = "nearest",
    downsample_mode  : str          = "stride",
    need_1x1_up      : bool         = True,
    sigmoid          : bool         = True,
    act_layer        : Any          = nn.LeakyReLU,
) -> nn.Sequential:
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
        need_1x1_up: Whether to use 1x1 convolution.
        sigmoid: Whether to use sigmoid function.
        act_layer: Activation layer.
    """
    assert len(channels_down) == len(channels_up) == len(channels_skip)
    n_scales = len(channels_down)

    if not (isinstance(upsample_mode, list) or isinstance(upsample_mode, tuple)):
        upsample_mode    = [upsample_mode]    * n_scales
    if not (isinstance(downsample_mode, list) or isinstance(downsample_mode, tuple)):
        downsample_mode  = [downsample_mode]  * n_scales
    if not (isinstance(kernel_size_down, list) or isinstance(kernel_size_down, tuple)):
        kernel_size_down = [kernel_size_down] * n_scales
    if not (isinstance(kernel_size_up, list) or isinstance(kernel_size_up, tuple)):
        kernel_size_up   = [kernel_size_up]   * n_scales

    last_scale = n_scales - 1
    cur_depth  = None
    model      = nn.Sequential()
    model_tmp  = model

    input_depth = in_channels
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
            skip.add(act_layer())

        deeper.add(conv(input_depth, channels_down[i], kernel_size_down[i], 2, bias=bias, padding=padding, downsample_mode=downsample_mode[i]))
        deeper.add(nn.BatchNorm2d(channels_down[i]))
        deeper.add(act_layer())

        deeper.add(conv(channels_down[i], channels_down[i], kernel_size_down[i], bias=bias, padding=padding))
        deeper.add(nn.BatchNorm2d(channels_down[i]))
        deeper.add(act_layer())

        deeper_main = nn.Sequential()

        if i == len(channels_down) - 1:
            # The deepest
            k = channels_down[i]
        else:
            deeper.add(deeper_main)
            k = channels_up[i + 1]

        deeper.add(nn.Upsample(scale_factor=2, mode=upsample_mode[i], align_corners=True))

        model_tmp.add(conv(channels_skip[i] + k, channels_up[i], kernel_size_up[i], 1, bias=bias, padding=padding))
        model_tmp.add(nn.BatchNorm2d(channels_up[i]))
        # model_tmp.add(layer_norm(num_channels_up[i]))
        model_tmp.add(act_layer())

        if need_1x1_up:
            model_tmp.add(conv(channels_up[i], channels_up[i], 1, bias=bias, padding=padding))
            model_tmp.add(nn.BatchNorm2d(channels_up[i]))
            model_tmp.add(act_layer())

        input_depth = channels_down[i]
        model_tmp   = deeper_main

    model.add(conv(channels_up[0], out_channels, 1, bias=bias, padding=padding))
    if sigmoid:
        model.add(nn.Sigmoid())

    return model


class VariationalAutoEncoder(nn.Module):

    class Encoder(nn.Module):

        def __init__(self, size: list | tuple):
            super().__init__()
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, 16, 5, 1, 2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(16, 32, 5, 1, 2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)
            )
            self.conv3 = nn.Sequential(
                nn.Conv2d(32, 64, 5, 1, 2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)
            )
            self.conv4 = nn.Sequential(
                nn.Conv2d(64, 128, 5, 1, 2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)
            )
            self.fc1 = nn.Linear(int(128 * (size[0] // 16) * (size[1] // 16)), 100)
            self.fc2 = nn.Linear(int(128 * (size[0] // 16) * (size[1] // 16)), 100)

        def forward(self, input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            x     = input
            y     = self.conv1(x)
            y     = self.conv2(y)
            y     = self.conv3(y)
            y     = self.conv4(y)
            y     = y.view(y.size(0), -1)
            means = self.fc1(y)
            var   = self.fc2(y)
            return means, var

    class Decoder(nn.Module):
        def __init__(self, size: list | tuple):
            super().__init__()
            self.linear0 = nn.Linear(100, int(128 * (size[0] // 16) * (size[1] // 16)))
            self.size    = size
            self.conv1 = nn.Sequential(
                nn.Conv2d(128, 64, 5, 1, 2),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(64, 32, 5, 1, 2),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True)
            )
            self.conv3 = nn.Sequential(
                nn.Conv2d(32, 16, 5, 1, 2),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True)
            )
            self.conv4 = nn.Sequential(
                nn.Conv2d(16, 3, 5, 1, 2),
                nn.BatchNorm2d(3),
                nn.ReLU(inplace=True)
            )
            self.de = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear"),
                nn.Conv2d(128, 64, 5, 1, 2),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.Upsample(scale_factor=2, mode="bilinear"),
                nn.Conv2d(64, 32, 5, 1, 2),
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.Upsample(scale_factor=2, mode="bilinear"),
                nn.Conv2d(32, 16, 5, 1, 2),
                nn.BatchNorm2d(16),
                nn.ReLU(True),
                nn.Upsample(scale_factor=2, mode="bilinear"),
                nn.Conv2d(16, 3, 5, 1, 2),
                nn.Sigmoid()
            )

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            x = input
            y = self.linear0(x)
            y = y.view(1, -1, self.size[0] // 16, self.size[1] // 16)
            y = self.de(y)
            return y

    def __init__(self, size: list | tuple):
        super().__init__()
        self.encoder = self.Encoder(size=size)
        self.decoder = self.Decoder(size=size)
        self.means   = None
        self.var     = None

    def get_latent(self, means, var):
        log_var    = var
        epsilon    = torch.randn(means.size()).cuda()
        sigma      = torch.exp(0.5 * log_var)
        z          = means + sigma * epsilon
        self.means = means
        self.var   = var
        return z

    def sample(self):
        z = self.getLatent(self.means, self.var)
        return self.decoder(z)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        means, var = self.encoder(input)
        z          = self.get_latent(means, var)
        return self.decoder(z)

    def get_loss(self):
        # lossX = torch.nn.functional.mse_loss(res, inputs, reduction='sum')
        log_var = self.var
        loss_kl = 0.5 * torch.sum(log_var.exp() + self.means * self.means - 1 - log_var)
        # print(lossX, lossKL)
        # loss = lossX + lossKL
        loss    = loss_kl
        return loss

# endregion


# region Model

@MODELS.register(name="zid", arch="zid")
class ZID(base.DehazingModel):
    """ZID (Zero-Shot Image Dehazing) model.
    
    See Also: :class:`base.Dehazing`
    """
    
    arch  : str  = "zid"
    schemes: list[Scheme] = [Scheme.UNSUPERVISED, Scheme.ZEROSHOT]
    zoo   : dict = {}
    
    def __init__(
        self,
        in_channels : int       = 3,
        out_channels: int       = 3,
        image_size  : _size_2_t = (512, 512),
        clip        : bool      = True,
        save_image  : bool      = False,
        weights     : Any       = None,
        loss        : Any       = None,
        *args, **kwargs
    ):
        super().__init__(
            name         = "zid",
            in_channels  = in_channels,
            out_channels = out_channels,
            weights      = weights,
            loss         = loss,
            *args, **kwargs
        )
        self.image_size = core.parse_hw(size=image_size or [512, 512])
        self.clip       = clip
        self.save_image = save_image
        
        # Image Network
        self.image_net   = encoder_decoder_skip(
            in_channels   = self.in_channels,
            out_channels  = self.out_channels,
            channels_down = [8, 16, 32, 64, 128],
            channels_up   = [8, 16, 32, 64, 128],
            channels_skip = [0, 0 , 0 , 4 , 4],
            padding       = "reflection",
            bias          = True,
            upsample_mode = "bilinear",
            sigmoid       = True,
            act_layer     = nn.LeakyReLU,
        ).type(torch.cuda.FloatTensor)

        # Mask Network
        self.mask_net    = encoder_decoder_skip(
            in_channels   = self.in_channels,
            out_channels  = 1,
            channels_down = [8, 16, 32, 64, 128],
            channels_up   = [8, 16, 32, 64, 128],
            channels_skip = [0, 0 , 0 , 4 , 4],
            bias          = True,
            padding       = "reflection",
            upsample_mode = "bilinear",
            sigmoid       = True,
            act_layer     = nn.LeakyReLU,
        ).type(torch.cuda.FloatTensor)
        
        # Ambient Network
        self.ambient_net = VariationalAutoEncoder(size=self.image_size).type(torch.cuda.FloatTensor)
        
        # Loss Functions
        self.mse_loss = nn.MSELoss(reduction="mean").type(torch.cuda.FloatTensor)
        self.std_loss = nn.StdLoss(reduction="mean").type(torch.cuda.FloatTensor)
        
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)
    
    def init_weights(self, model: nn.Module):
        pass
    
    def forward_loss(
        self,
        input : torch.Tensor,
        target: torch.Tensor | None,
        *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        pred = self.forward(input=input, *args, **kwargs)
        image, ambient, mask, _ = pred
        
        loss         = self.mse_loss(mask * image + (1 - mask) * ambient, image)
        loss        += self.ambient_net.get_loss()
        loss        += 0.005 * self.std_loss(mask)
        loss        += 0.1   * self.std_loss(ambient)
        #
        dcp_prior    = torch.min(image.permute(0, 2, 3, 1), 3)[0]
        loss        += self.mse_loss(dcp_prior, torch.zeros_like(dcp_prior)) - 0.05
        #
        from mon.vision import prior
        atmosphere   = prior.get_atmosphere_prior(input.detach().cpu().numpy()[0])
        ambient_val  = nn.Parameter(data=torch.cuda.FloatTensor(atmosphere.reshape((1, 3, 1, 1))), requires_grad=False)
        loss        += self.mse_loss(ambient, ambient_val * torch.ones_like(ambient))
        
        return pred[-1], loss
    
    def forward(
        self,
        input    : torch.Tensor,
        augment  : _callable = None,
        profile  : bool      = False,
        out_index: int       = -1,
        *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x       = input
        image   = self.image_net(x)
        ambient = self.ambient_net(x)
        mask    = self.mask_net(x)
        
        ambient_clip = torch.clip(ambient, 0, 1)
        mask_clip    = torch.clip(mask,    0, 1)
        mask_clip    = self.t_matting(x, mask_clip).to(x.device)
        y            = torch.clip((x - ((1 - mask_clip) * ambient_clip)) / mask_clip, 0, 1)
        
        return image, ambient, mask, y
    
    def t_matting(self, input: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        input    = input.detach().cpu().numpy()[0]
        mask     = mask.detach().cpu().numpy()[0]
        refine_t = guidedFilter(
            guide  = input.transpose(1, 2, 0).astype(np.float32),
            src    = mask[0].astype(np.float32),
            radius = 50,
            eps    = 1e-4,
        )
        if self.clip:
            refine_t = np.array([np.clip(refine_t, 0.1, 1)])
        else:
            refine_t = np.array([np.clip(refine_t, 0, 1)])
        refine_t = torch.from_numpy(refine_t)[None, :]
        return refine_t
        
# endregion

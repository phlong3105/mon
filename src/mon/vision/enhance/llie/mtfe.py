#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements MTFE (Multiple Transformation Function Estimation for
Image Enhancement) models.

References:
    `<https://github.com/PJaemin/MTFE/tree/main>`__
"""

from __future__ import annotations

__all__ = [
    "MTFE",
]

from typing import Any, Literal

import numpy as np
import torch

from mon import core, nn
from mon.core import _callable, _size_2_t
from mon.globals import MODELS, Scheme
from mon.nn import functional as F
from mon.vision.enhance.llie import base

console = core.console


# region Loss

class TotalVariationLoss(nn.Loss):
    
    def __init__(
        self,
        loss_weight: float = 1e-4,
        reduction  : Literal["none", "mean", "sum"] = "mean",
    ):
        super().__init__(loss_weight=loss_weight, reduction=reduction)
    
    def forward(
        self,
        input : torch.Tensor,
        target: torch.Tensor | None = None
    ) -> torch.Tensor:
        w1, w2, w3 = input
        w       = torch.cat((w1, w2, w3), dim=1)
        b       = w.size()[0]
        h_x     = w.size()[2]
        w_x     = w.size()[3]
        count_h = (w.size()[2] - 1) * w.size()[3]
        count_w = w.size()[2] * (w.size()[3] - 1)
        h_tv    = torch.pow((w[:, :, 1:, :] - w[:, :, :h_x - 1, :]), 2).sum() / count_h
        w_tv    = torch.pow((w[:, :, :, 1:] - w[:, :, :, :w_x - 1]), 2).sum() / count_w
        loss    = self.loss_weight * (h_tv + w_tv) / b
        return loss
    

class Loss(nn.Loss):

    def __init__(
        self,
        reduction: Literal["none", "mean", "sum"] = "mean",
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.loss_c = nn.L2Loss(reduction=reduction)
        self.loss_e = nn.EntropyLoss(reduction=reduction)
        self.loss_t = TotalVariationLoss(reduction=reduction)
        self.cos    = nn.CosineSimilarity(dim=1)
    
    def forward(
        self,
        input  : torch.Tensor,
        target : torch.Tensor,
        weights: torch.Tensor,
        **_
    ) -> torch.Tensor:
        pass

# endregion


# region Module

class DoubleConv2d(nn.Module):
    """(Convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels: int, out_channels: int, mid_channels: int | None = None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.double_conv(input)


class UNetDownBlock(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv2d(in_channels, out_channels)
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(input)


class UNetUpBlock(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        # If bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv2d(in_channels, out_channels, in_channels // 2)
        else:
            self.up   = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv2d(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # Input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.bilinear     = bilinear
        factor            = 2 if bilinear else 1
        
        self.inc   = DoubleConv2d(self.in_channels, 16)
        self.down1 = UNetDownBlock(16, 32)
        self.down2 = UNetDownBlock(32, 64)
        self.down3 = UNetDownBlock(64, 128)
        self.down4 = UNetDownBlock(128, 256 // factor)
        self.up1   = UNetUpBlock(256, 128 // factor, bilinear)
        self.up2   = UNetUpBlock(128, 64 // factor, bilinear)
        self.up3   = UNetUpBlock(64, 32 // factor, bilinear)
        self.up4   = UNetUpBlock(32, 16, bilinear)
        self.outc  = nn.Conv2d(16, self.out_channels, kernel_size=1)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x  = input
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x  = self.up1(x5, x4)
        x  = self.up2(x, x3)
        x  = self.up3(x, x2)
        x  = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class IntensityTransform(nn.Module):
    
    def __init__(self, intensities: int, channels: int, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.scale    = intensities - 1

    def get_config(self):
        config = super(IntensityTransform, self).get_config()
        config.update({"channels": self.channels, "scale": self.scale})
        return config

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        images, transforms = inputs
        transforms = transforms.unsqueeze(3)  # Index tensor must have the same number of dimensions as input tensor

        # images = 0.5 * images + 0.5
        images     = torch.round(self.scale * images)
        images     = images.type(torch.LongTensor)
        images     = images.cuda()
        transforms = transforms.cuda()
        minimum_w  = images.size(3)
        iter_n     = 0
        temp       = 1
        while minimum_w > temp:
            temp   *= 2
            iter_n += 1

        for i in range(iter_n):
            transforms = torch.cat([transforms, transforms], dim=3)

        images     = torch.split(images, 1, dim=1)
        transforms = torch.split(transforms, 1, dim=1)

        x = torch.gather(input=transforms[0], dim=2, index=images[0])
        y = torch.gather(input=transforms[1], dim=2, index=images[1])
        z = torch.gather(input=transforms[2], dim=2, index=images[2])

        xx = torch.cat([x, y, z], dim=1)

        return xx


class ConvBlock(nn.Module):
    
    def __init__(
        self, 
        in_channels : int,
        out_channels: int,
        kernel_size : _size_2_t,
        stride      : _size_2_t,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.dropout_rate = dropout_rate
        padding           = kernel_size // 2
        self.cb_conv1     = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=False)
        self.cb_batchNorm = nn.BatchNorm2d(out_channels)
        if dropout_rate > 0:
            self.dropout = nn.Dropout2d(p=dropout_rate)

    @staticmethod
    def swish(x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        x = self.cb_conv1(x)
        x = self.cb_batchNorm(x)
        x = self.swish(x)
        if self.dropout_rate > 0:
            x = self.dropout(x)
        return x


class SFC(nn.Module):
    
    def __init__(self, in_channels: int, out_channels: int, expansion: int, num: int):
        super().__init__()
        exp_ch = int(in_channels * expansion)
        if num == 1:
            self.se_conv = nn.Conv2d(in_channels, exp_ch, 3, 1, 1, groups=in_channels)
        else:
            self.se_conv = nn.Conv2d(in_channels, exp_ch, 3, 2, 1, groups=in_channels)
        self.se_bn   = nn.BatchNorm2d(exp_ch)
        self.se_relu = nn.ReLU()
        self.hd_conv = nn.Conv2d(exp_ch, exp_ch, 3, 1, 1, groups=in_channels)
        self.hd_bn   = nn.BatchNorm2d(exp_ch)
        self.hd_relu = nn.ReLU()
        self.cp_conv = nn.Conv2d(exp_ch, out_channels, 1, 1, groups=in_channels)
        self.cp_bn   = nn.BatchNorm2d(out_channels)
        self.pw_conv = nn.Conv2d(out_channels, out_channels, 1, 1)
        self.pw_bn   = nn.BatchNorm2d(out_channels)
        self.pw_relu = nn.ReLU()

    @staticmethod
    def swish(x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        x = self.se_conv(x)
        x = self.se_bn(x)
        x = self.se_relu(x)
        x = self.hd_conv(x)
        x = self.hd_bn(x)
        x = self.hd_relu(x)
        x = self.cp_conv(x)
        x = self.cp_bn(x)
        x = self.pw_conv(x)
        x = self.pw_bn(x)
        x = self.pw_relu(x)
        return x


class HSFC(nn.Module):
    
    def __init__(self, in_channels: int, expansion: int):
        super().__init__()
        expand_channels = int(in_channels * expansion)
        self.se_conv    = nn.Conv1d(in_channels, expand_channels, 3, 1, 1, groups=in_channels)
        self.se_bn      = nn.BatchNorm1d(expand_channels)
        self.se_relu    = nn.ReLU()
        self.hd_conv    = nn.Conv1d(expand_channels, expand_channels, 3, 1, 1, groups=in_channels)
        self.hd_bn      = nn.BatchNorm1d(expand_channels)
        self.hd_relu    = nn.ReLU()
        self.cp_conv    = nn.Conv1d(expand_channels, in_channels, 1, 1, groups=in_channels)
        self.cp_bn      = nn.BatchNorm1d(in_channels)
        self.pw_conv    = nn.Conv1d(in_channels, in_channels, 1, 1)
        self.pw_bn      = nn.BatchNorm1d(in_channels)
        self.pw_relu    = nn.ReLU()

    @staticmethod
    def swish(x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        x = self.se_conv(x)
        x = self.se_bn(x)
        x = self.se_relu(x)
        x = self.hd_conv(x)
        x = self.hd_bn(x)
        x = self.hd_relu(x)
        x = self.cp_conv(x)
        x = self.cp_bn(x)
        x = self.pw_conv(x)
        x = self.pw_bn(x)
        x = self.pw_relu(x)
        return x


class HistogramNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        expansion   = 4
        self.stage1 = HSFC(3, expansion)
        self.stage2 = HSFC(3, expansion)
        self.stage3 = HSFC(3, expansion)
        self.stage4 = HSFC(3, expansion)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        h = input
        y = self.stage1(h)
        y = self.stage2(y)
        y = self.stage3(y)
        y = self.stage4(y)
        y = y.flatten(1)
        return y


class AttentionBlock(nn.Module):
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.g_conv  = nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=int(in_channels / 2))
        self.g_bn    = nn.BatchNorm2d(in_channels)
        self.g_relu  = nn.ReLU()

        self.pw_conv = nn.Conv2d(in_channels, out_channels, 1, 1)
        self.pw_bn   = nn.BatchNorm2d(out_channels)
        self.pw_relu = nn.ReLU()

    @staticmethod
    def swish(x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

    def forward(self, in1: torch.Tensor, in2: torch.Tensor) -> torch.Tensor:
        i1_1, i1_2, i1_3 = torch.chunk(in1, 3, dim=1)
        i2_1, i2_2, i2_3 = torch.chunk(in2, 3, dim=1)
       
        x = torch.cat([i1_1, i2_1, i1_2, i2_2, i1_3, i2_3], dim=1)

        x = self.g_conv(x)
        x = self.g_bn(x)
        x = self.g_relu(x)

        x = self.pw_conv(x)
        x = self.pw_bn(x)
        x = self.pw_relu(x)

        return x

# endregion


# region Model

@MODELS.register(name="mtfe", arch="mtfe")
class MTFE(base.LowLightImageEnhancementModel):
    """MTFE (Multiple Transformation Function Estimation for Image Enhancement)
    models.
    
    Args:
        in_channels: The first layer's input channel. Default: ``3`` for RGB image.
    
    References:
        `<https://github.com/PJaemin/MTFE/tree/main>`__
    
    See Also: :class:`base.LowLightImageEnhancementModel`
    """
    
    arch   : str  = "mtfe"
    schemes: list[Scheme] = [Scheme.SUPERVISED]
    zoo    : dict = {}

    def __init__(
        self,
        in_channels: int = 3,
        weights    : Any = None,
        *args, **kwargs
    ):
        super().__init__(
            name        = "mtfe",
            in_channels = in_channels,
            weights     = weights,
            *args, **kwargs
        )
        
        # Populate hyperparameter values from pretrained weights
        if isinstance(self.weights, dict):
            in_channels = self.weights.get("in_channels" , in_channels)
        self.in_channels   = in_channels
        self.expansion     = 4
        self.base_channels = 6
        
        # Construct model
        self.wm_gen     = UNet(12, 3)
        self.histnet    = HistogramNetwork()
        self.stage1     = nn.Conv2d(3, self.base_channels, 3, 1, 1)
        self.stage1_bn  = nn.BatchNorm2d(self.base_channels)
        self.stage1_af  = nn.ReLU()
        self.stage2     = SFC(     self.base_channels,   2 * self.base_channels, self.expansion, 1)
        self.stage3     = SFC( 2 * self.base_channels,   4 * self.base_channels, self.expansion, 2)
        self.stage4     = SFC( 4 * self.base_channels,   8 * self.base_channels, self.expansion, 3)
        self.stage5     = SFC( 8 * self.base_channels,  16 * self.base_channels, self.expansion, 4)
        self.stage6     = SFC(16 * self.base_channels,  32 * self.base_channels, self.expansion, 5)
        self.stage7     = SFC(32 * self.base_channels,  64 * self.base_channels, self.expansion, 6)
        self.stage8     = SFC(64 * self.base_channels, 128 * self.base_channels, self.expansion, 7)
        self.gap        = nn.AdaptiveAvgPool2d(1)
        self.fusion_cv1 = nn.Conv2d(2, 2, 1)
        self.fusion_bn1 = nn.BatchNorm2d(2)
        self.fusion_ru1 = nn.ReLU()
        self.fusion_cv2 = nn.Conv2d(2, 1, 1)
        self.fusion_bn2 = nn.BatchNorm2d(1)
        self.fusion_ru2 = nn.ReLU()
        self.fusion_FC  = nn.Linear(768, 768)
        self.fusion_bn  = nn.BatchNorm1d(768)
        self.fusion_sig = nn.Sigmoid()
        
        self.fc11       = nn.Linear(768, 768)
        self.fc12       = nn.Linear(768, 768)
        self.fc13       = nn.Linear(768, 768)
        self.fc21       = nn.Linear(768, 768)
        self.fc22       = nn.Linear(768, 768)
        self.fc23       = nn.Linear(768, 768)
        self.fc31       = nn.Linear(768, 768)
        self.fc32       = nn.Linear(768, 768)
        self.fc33       = nn.Linear(768, 768)

        self.intensity_trans = IntensityTransform(intensities=256, channels=3)
        
        # Loss
        self._loss = Loss()
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)

    def init_weights(self, m: nn.Module):
        classname = m.__class__.__name__
        if classname.find("Conv2d") != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
    
    def forward_loss(self, datapoint: dict, *args, **kwargs) -> dict | None:
        input  = datapoint.get("input",  None)
        target = datapoint.get("target", None)
        meta   = datapoint.get("meta",   None)
        pred   = self.forward(input=input, *args, **kwargs)
        adjust, enhance = pred
        loss   = self.loss(input, adjust, enhance)
        return {
            "pred": enhance,
            "loss": loss,
        }

    def forward(
        self,
        input    : torch.Tensor,
        augment  : _callable = None,
        profile  : bool      = False,
        out_index: int       = -1,
        *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x    = input
        hist = self.get_histogram(x)
        
        x_256 = F.interpolate(x, 256)
        y   = self.stage1(x_256)
        y   = self.stage1_bn(y)
        y   = self.stage1_af(y)
        
        y   = self.stage2(y)
        y   = self.stage3(y)
        y   = self.stage4(y)
        y   = self.stage5(y)
        y   = self.stage6(y)
        y   = self.stage7(y)
        y   = self.stage8(y)
        y   = self.gap(y)
        y   = y.squeeze(2)
        y   = y.squeeze(2)
        
        h   = self.histnet(hist)
        h   = h.unsqueeze(1)
        h   = h.unsqueeze(3)
        ya  = y.unsqueeze(1)
        ya  = ya.unsqueeze(3)
        ya  = torch.cat([ya, h], dim=1)
        ya  = self.fusion_cv1(ya)
        ya  = self.fusion_bn1(ya)
        ya  = self.fusion_ru1(ya)
        ya  = self.fusion_cv2(ya)
        ya  = self.fusion_bn2(ya)
        ya  = self.fusion_ru2(ya)
        ya  = ya.squeeze(3)
        ya  = ya.squeeze(1)
        ya  = self.fusion_FC(ya)
        ya  = self.fusion_bn(ya)
        ya  = self.fusion_sig(ya)
        y   = y * ya + y
        y   = torch.relu(y)
        
        y1  = self.fc11(y)
        y1  = self.fc12(y1)
        y1  = self.fc13(y1)
        y2  = self.fc21(y)
        y2  = self.fc22(y2)
        y2  = self.fc23(y2)
        y3  = self.fc31(y)
        y3  = self.fc32(y3)
        y3  = self.fc33(y3)
        
        y1  = y1.unsqueeze(1)
        y1  = torch.chunk(y1, 3, dim=2)
        tf1 = torch.cat(y1, dim=1)
        tf1 = torch.sigmoid(tf1)
        xy1 = self.intensity_trans((x, tf1))
        # xy1 = xy1 * 0.5 + 0.5
        
        y2  = y2.unsqueeze(1)
        y2  = torch.chunk(y2, 3, dim=2)
        tf2 = torch.cat(y2, dim=1)
        tf2 = torch.sigmoid(tf2)
        xy2 = self.intensity_trans((x, tf2))
        # xy2 = xy2 * 0.5 + 0.5
        
        y3  = y3.unsqueeze(1)
        y3  = torch.chunk(y3, 3, dim=2)
        tf3 = torch.cat(y3, dim=1)
        tf3 = torch.sigmoid(tf3)
        xy3 = self.intensity_trans((x, tf3))
        # xy3 = xy3 * 0.5 + 0.5
        
        w = self.wm_gen(torch.cat((x, xy1, xy2, xy3), dim=1))
        w = torch.sigmoid(w)
        w1, w2, w3 = torch.chunk(w, 3, dim=1)
        w1 = w1 / (w1 + w2 + w3)
        w2 = w2 / (w1 + w2 + w3)
        w3 = w3 / (w1 + w2 + w3)
        
        xy = w1 * xy1 + w2 * xy2 + w3 * xy3
        return xy, (tf1, tf2, tf3), (w1, w2, w3), (xy1, xy2, xy3)
    
    @staticmethod
    def swish(x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)
    
    @staticmethod
    def get_histogram(input: torch.Tensor) -> torch.Tensor:
        input_np  = core.to_image_nparray(input, keepdim=False, denormalize=True)
        histogram = []
        for input in input_np:
            hist_s = np.zeros((3, 256))
            for (j, color) in enumerate(("red", "green", "blue")):
                s = input[..., j]
                hist_s[j, ...], _ = np.histogram(s.flatten(), 256, [0, 256])
                hist_s[j, ...]    = hist_s[j, ...] / np.sum(hist_s[j, ...])
            hist_s = torch.from_numpy(hist_s).float()
            histogram.append(hist_s)
        
        histogram = torch.stack(histogram, dim=0)
        return histogram

# endregion

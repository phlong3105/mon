#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""LLFlow: Low-Light Image Enhancement with Normalizing Flow.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor

from one.core import Indexes
from one.core import ListOrTupleAnyT
from one.core import MODELS
from one.core import Pretrained
from one.nn import ConcatPadding
from one.nn import ConvAct
from one.nn import ConvTransposeAct
from one.nn import RRDB
from one.vision.image_enhancement.image_enhancer import ImageEnhancer

__all__ = [

]


# MARK: - Modules

class RRDBNet(nn.Module):
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        in_channels    : int,
        out_channels   : int,
        nf             : int,
        nb             : int,
        gc             : int                  = 32,
        scale          : int                  = 4,
        fea_up0        : bool                 = True,
        rrdb_out_blocks: ListOrTupleAnyT[int] = (),
    ):
        super().__init__()
        self.scale           = scale
        self.fea_up0_en      = fea_up0
        self.rrdb_out_blocks = rrdb_out_blocks
        
        self.conv_first = nn.Conv2d(in_channels, nf, (3, 3), (2, 2), 1, bias=True)
        self.rrdb_trunk = nn.Sequential(*[RRDB(nf=nf, gc=gc) for _ in range(nb)])
        self.trunk_conv = nn.Conv2d(nf, nf, (3, 3), (2, 2), 1, bias=True)
        
        # NOTE: upsampling
        self.upconv1    = nn.Conv2d(nf, nf, (3, 3), (1, 1), 1, bias=True)
        self.upconv2    = nn.Conv2d(nf, nf, (3, 3), (1, 1), 1, bias=True)
        if self.scale >= 8:
            self.upconv3 = nn.Conv2d(nf, nf, (3, 3), (1, 1), 1, bias=True)
        if self.scale >= 16:
            self.upconv4 = nn.Conv2d(nf, nf, (3, 3), (1, 1), 1, bias=True)
        if self.scale >= 32:
            self.upconv5 = nn.Conv2d(nf, nf, (3, 3), (1, 1), 1, bias=True)

        self.hrconv     = nn.Conv2d(nf, nf, (3, 3), (1, 1), 1, bias=True)
        self.conv_last  = nn.Conv2d(nf, out_channels, (3, 3), (1, 1), 1, bias=True)
        self.lrelu      = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    # MARK: Forward Pass
    
    def forward(self, x: Tensor, get_steps: bool = False):
        fea = self.conv_first(x)
        
        block_idxs    = self.rrdb_out_blocks
        block_results = {}
        for idx, m in enumerate(self.rrdb_trunk.children()):
            fea = m(fea)
            for b in block_idxs:
                if b == idx:
                    block_results["block_{}".format(idx)] = fea
       
        trunk       = self.trunk_conv(fea)
        fea         = F.max_pool2d(fea, 2)
        last_lr_fea = fea + trunk
        fea_up2     = self.upconv1(F.interpolate(last_lr_fea, scale_factor=2, mode="nearest"))
        fea         = self.lrelu(fea_up2)
        fea_up4     = self.upconv2(F.interpolate(fea, scale_factor=2, mode="nearest"))
        fea         = self.lrelu(fea_up4)
        
        fea_up8     = None
        fea_up16    = None
        fea_up32    = None
        if self.scale >= 8:
            fea_up8 = self.upconv3(fea)
            fea     = self.lrelu(fea_up8)
        if self.scale >= 16:
            fea_up16 = self.upconv4(fea)
            fea      = self.lrelu(fea_up16)
        if self.scale >= 32:
            fea_up32 = self.upconv5(fea)
            fea      = self.lrelu(fea_up32)

        out     = self.conv_last(self.lrelu(self.hrconv(fea)))
        results = {
            "last_lr_fea": last_lr_fea,
            "fea_up1"    : last_lr_fea,
            "fea_up2"    : fea_up2,
            "fea_up4"    : fea_up4,  # raw
            "fea_up8"    : fea_up8,
            "fea_up16"   : fea_up16,
            "fea_up32"   : fea_up32,
            "out"        : out
        }  # raw

        if self.fea_up0_en:
            results["fea_up0"] = F.interpolate(
                last_lr_fea, scale_factor=1/2, mode="bilinear",
                align_corners=False, recompute_scale_factor=True
            )
        fea_upn1_en = True
        if fea_upn1_en:
            results["fea_up-1"] = F.interpolate(
                last_lr_fea, scale_factor=1/4, mode="bilinear",
                align_corners=False, recompute_scale_factor=True
            )

        if get_steps:
            for k, v in block_results.items():
                results[k] = v
            return results
        else:
            return out


class ColorEncoder(nn.Module):
    
    # MARK: Magic Functions
    
    def __init__(self, nf: int):
        super().__init__()
        self.conv_input = ConvAct(3, nf, act_layer=nn.LeakyReLU)
        
        # NOTE: Top path build Reflectance map
        self.maxpool_r1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_r1    = ConvAct(
            in_channels=nf, out_channels=nf * 2, kernel_size=3, stride=1,
            padding=1, act_layer=nn.LeakyReLU
        )
        self.maxpool_r2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_r2    = ConvAct(
            in_channels=nf * 2, out_channels=nf * 4, kernel_size=3, stride=1,
            padding=1, act_layer=nn.LeakyReLU
        )
        self.deconv_r1  = ConvTransposeAct(
            in_channels=nf * 4, out_channels=nf * 2, kernel_size=2, stride=2,
            padding=0, act_layer= nn.LeakyReLU
        )
        self.concat_r1  = ConcatPadding()
        self.conv_r3    = ConvAct(
            in_channels=nf * 4, out_channels=nf * 2, kernel_size=3, stride=1,
            padding=1, act_layer=nn.LeakyReLU
        )
        self.deconv_r2  = ConvTransposeAct(
            in_channels=nf * 2, out_channels=nf, kernel_size=2, stride=2,
            padding=0, act_layer= nn.LeakyReLU
        )
        self.concat_r2  = ConcatPadding()
        self.conv_r4    = ConvAct(
            in_channels=nf * 2, out_channels=nf, kernel_size=3, stride=1,
            padding=1, act_layer=nn.LeakyReLU
        )
        self.conv_r5    = nn.Conv2d(nf, 3, kernel_size=(3, 3), padding=1)
        self.r_out      = nn.Sigmoid()
    
    # MARK: Forward Pass
    def forward(self, x: Tensor) -> Tensor:
        conv_input = self.conv_input(x)
        # Build Reflectance map
        maxpool_r1 = self.maxpool_r1(conv_input)
        conv_r1    = self.conv_r1(maxpool_r1)
        maxpool_r2 = self.maxpool_r2(conv_r1)
        conv_r2    = self.conv_r2(maxpool_r2)
        deconv_r1  = self.deconv_r1(conv_r2)
        concat_r1  = self.concat_r1(conv_r1, deconv_r1)
        conv_r3    = self.conv_r3(concat_r1)
        deconv_r2  = self.deconv_r2(conv_r3)
        concat_r2  = self.concat_r2(conv_input, deconv_r2)
        conv_r4    = self.conv_r4(concat_r2)
        conv_r5    = self.conv_r5(conv_r4)
        R_out      = self.r_out(conv_r5)
        color_x    = nn.functional.avg_pool2d(
            R_out,
            self.opt["avg_kernel_size"], 1,
            self.opt["avg_kernel_size"] // 2
        )
        return color_x


# noinspection PyMethodMayBeStatic
class ConditionEncoder(nn.Module):
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        in_channels     : int,
        out_channels    : int,
        nf              : int,
        nb              : int,
        gc              : int                  = 32,
        scale           : int                  = 4,
        rrdb_out_blocks : ListOrTupleAnyT[int] = (),
        concat_histeq   : bool                 = True,
        concat_color_map: bool                 = False,
        gray_map        : bool                 = False,
    ):
        self.gray_map_bool    = False
        self.concat_color_map = False
        if concat_histeq:
            in_channels += 3
        if concat_color_map:
            in_channels          += 3
            self.concat_color_map = True
        if gray_map:
            in_channels       += 1
            self.gray_map_bool = True
        in_channels += 6
        
        super().__init__()
        self.scale           = scale
        self.rrdb_out_blocks = rrdb_out_blocks
        
        self.conv_first  = nn.Conv2d(in_channels, nf, (3, 3), (1, 1), 1, bias=True)
        self.conv_second = nn.Conv2d(nf,          nf, (3, 3), (1, 1), 1, bias=True)
        self.rrdb_trunk  = nn.Sequential(*[RRDB(nf=nf, gc=gc) for _ in range(nb)])
        self.trunk_conv  = nn.Conv2d(nf, nf, (3, 3), (1, 1), 1, bias=True)
       
        # NOTE: downsampling
        self.downconv1   = nn.Conv2d(nf, nf, (3, 3), (1, 1), 1, bias=True)
        self.downconv2   = nn.Conv2d(nf, nf, (3, 3), (1, 1), 1, bias=True)
        self.downconv3   = nn.Conv2d(nf, nf, (3, 3), (1, 1), 1, bias=True)
        # self.downconv4 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.hrconv      = nn.Conv2d(nf, nf, (3, 3), (1, 1), 1, bias=True)
        self.conv_last   = nn.Conv2d(nf, out_channels, (3, 3), (1, 1), 1, bias=True)
        self.lrelu       = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.awb_para    = nn.Linear(nf, 3)
        self.fine_tune_color_map = nn.Sequential(
            nn.Conv2d(nf, 3, (1, 1), (1, 1)), nn.Sigmoid()
        )
    
    # MARK: Forward Pass
    
    def forward(self, x: Tensor, get_steps: bool = False):
        if self.gray_map_bool:
            x = torch.cat([x, 1 - x.mean(dim=1, keepdim=True)], dim=1)
        if self.concat_color_map:
            x = torch.cat([x, x / (x.sum(dim=1, keepdim=True) + 1e-4)], dim=1)

        raw_low_input = x[:, 0:3].exp()
        awb_weight    = 1
        low_after_awb = raw_low_input * awb_weight
        color_map     = low_after_awb / (low_after_awb.sum(dim=1, keepdims=True) + 1e-4)
        dx, dy        = self.gradient(color_map)
        noise_map     = torch.max(torch.stack([dx.abs(), dy.abs()], dim=0), dim=0)[0]

        fea           = self.conv_first(torch.cat([x, color_map, noise_map], dim=1))
        fea           = self.lrelu(fea)
        fea           = self.conv_second(fea)
        fea_head      = F.max_pool2d(fea, 2)

        block_idxs    = self.rrdb_out_blocks
        block_results = {}
        fea           = fea_head
        for idx, m in enumerate(self.rrdb_trunk.children()):
            fea = m(fea)
            for b in block_idxs:
                if b == idx:
                    block_results["block_{}".format(idx)] = fea
                    
        trunk     = self.trunk_conv(fea)
        fea_down2 = fea_head + trunk
        fea_down4 = self.downconv1(
            F.interpolate(
                fea_down2, scale_factor=1 / 2, mode="bilinear",
                align_corners=False, recompute_scale_factor=True
            )
        )
        fea       = self.lrelu(fea_down4)
        fea_down8 = self.downconv2(
            F.interpolate(
                fea, scale_factor=1 / 2, mode="bilinear",
                align_corners=False, recompute_scale_factor=True
            )
        )
      
        results = {
            "fea_up0"    : fea_down8,
            "fea_up1"    : fea_down4,
            "fea_up2"    : fea_down2,
            "fea_up4"    : fea_head,
            "last_lr_fea": fea_down4,
            "color_map"  : self.fine_tune_color_map(
                    F.interpolate(fea_down2, scale_factor=2)
                )
        }
        
        if get_steps:
            for k, v in block_results.items():
                results[k] = v
            return results
        else:
            return None

    def gradient(self, x: Tensor) -> tuple[Tensor, Tensor]:
        def sub_gradient(x):
            left_shift_x              = torch.zeros_like(x)
            right_shift_x             = torch.zeros_like(x)
            left_shift_x[ :, :, 0:-1] = x[:, :, 1:]
            right_shift_x[:, :, 1:  ] = x[:, :, 0:-1]
            grad                      = 0.5 * (left_shift_x - right_shift_x)
            return grad

        return sub_gradient(x), \
               sub_gradient(torch.transpose(x, 2, 3)).transpose(2, 3)


# MARK: - LLFlow

@MODELS.register(name="llflow")
class LLFlow(ImageEnhancer):
    
    def __init__(
        self,
        # Hyperparameters
        in_channels : int           = 3,
        out_channels: int           = 3,
        nf          : int           = 32,
        nb          : int           = 4,  # 12 for low light encoder, 23 for LLFlow
        gc          : int           = 32,
        scale       : int           = 4,
        K                           = None,
        # BaseModel's args
        basename    : Optional[str] = "ffa",
        name        : Optional[str] = "ffa",
        out_indexes : Indexes       = -1,
        pretrained  : Pretrained    = False,
        *args, **kwargs
    ):
        super().__init__(
            basename=basename, name=name, out_indexes=out_indexes,
            pretrained=pretrained, *args, **kwargs
        )
        # NOTE: Get Hyperparameters
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.nf           = nf
        self.nb           = nb
        self.gc           = gc
        self.scale        = scale
        self.K            = K

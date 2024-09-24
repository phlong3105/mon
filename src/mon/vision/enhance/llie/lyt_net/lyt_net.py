#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""LYT-Net.

This module implements the paper: "LYT-Net: Lightweight YUV Transformer-based
Network for Low-Light Image Enhancement"

References:
    https://github.com/albrateanu/LYT-Net
"""

from __future__ import annotations

__all__ = [
    "LYTNet_RE",
]

from typing import Any, Literal

import torch
import torch.nn.functional as F
import torch.nn.init as init
from torchvision.models import vgg19, VGG19_Weights

from mon import core, nn
from mon.globals import MODELS, Scheme, Task
from mon.vision.enhance import base

console      = core.console
current_file = core.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# region Loss

class Loss(nn.Loss):
    
    def __init__(
        self,
        alpha1   : float = 1.00,
        alpha2   : float = 0.06,
        alpha3   : float = 0.05,
        alpha4   : float = 0.5,
        alpha5   : float = 0.0083,
        alpha6   : float = 0.25,
        reduction: Literal["none", "mean", "sum"] = "mean",
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs, reduction=reduction)
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.alpha3 = alpha3
        self.alpha4 = alpha4
        self.alpha5 = alpha5
        self.alpha6 = alpha6
        
        self.smooth_l1_loss  = nn.SmoothL1Loss(reduction=reduction)
        self.perceptual_loss = nn.PerceptualLoss(net=vgg19(weights=VGG19_Weights))
        self.histogram_loss  = nn.HistogramLoss(bins=256, reduction=reduction)
        self.psnr_loss       = nn.PSNRLoss(reduction=reduction)
        self.ssim_loss       = nn.SSIMLoss(reduction)
        
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        smooth_l1_loss  = self.smooth_l1_loss(input, target)
        perceptual_loss = self.perceptual_loss(input, target)
        histogram_loss  = self.histogram_loss(input, target)
        ssim_loss       = self.ssim_loss(input, target)
        psnr_loss       = self.psnr_loss(input, target)
        color_loss      = self._color_loss(input, target)
        loss = (
              self.alpha1 * smooth_l1_loss
            + self.alpha2 * perceptual_loss
            + self.alpha3 * histogram_loss
            + self.alpha4 * ssim_loss
            + self.alpha5 * psnr_loss
            + self.alpha6 * color_loss
        )
        loss = nn.reduce_loss(loss=loss, reduction=self.reduction)
        return loss
    
    @staticmethod
    def _color_loss(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mean_input  = torch.mean(input,  dim=[1, 2])
        mean_target = torch.mean(target, dim=[1, 2])
        abs_diff    = torch.abs(mean_input - mean_target)
        result      = torch.mean(abs_diff)
        return result
    
# endregion


# region Module

class LayerNormalization(nn.Module):
    
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Rearrange the tensor for LayerNorm (B, C, H, W) to (B, H, W, C)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        # Rearrange back to (B, C, H, W)
        return x.permute(0, 3, 1, 2)
    
    
class SEBlock(nn.Module):
    
    def __init__(self, input_channels: int, reduction_ratio: int = 16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1  = nn.Linear(input_channels, input_channels // reduction_ratio)
        self.fc2  = nn.Linear(input_channels // reduction_ratio, input_channels)
        self.init_weights()
    
    def init_weights(self):
        init.kaiming_uniform_(self.fc1.weight, a=0, mode="fan_in", nonlinearity="relu")
        init.kaiming_uniform_(self.fc2.weight, a=0, mode="fan_in", nonlinearity="relu")
        init.constant_(self.fc1.bias, 0)
        init.constant_(self.fc2.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, _, _ = x.size()
        y = self.pool(x).reshape(batch_size, num_channels)
        y = F.relu(self.fc1(y))
        y = torch.tanh(self.fc2(y))
        y = y.reshape(batch_size, num_channels, 1, 1)
        return x * y
    
    
class MSEFBlock(nn.Module):
    
    def __init__(self, filters: int):
        super().__init__()
        self.layer_norm     = LayerNormalization(filters)
        self.depthwise_conv = nn.Conv2d(filters, filters, kernel_size=3, padding=1, groups=filters)
        self.se_attn        = SEBlock(filters)
        self.init_weights()
    
    def init_weights(self):
        init.kaiming_uniform_(self.depthwise_conv.weight, a=0, mode="fan_in", nonlinearity="relu")
        init.constant_(self.depthwise_conv.bias, 0)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm  = self.layer_norm(x)
        x1      = self.depthwise_conv(x_norm)
        x2      = self.se_attn(x_norm)
        x_fused = x1 * x2
        x_out   = x_fused + x
        return x_out


class MultiHeadSelfAttention(nn.Module):
    
    def __init__(self, embed_size: int, num_heads: int):
        super().__init__()
        self.embed_size    = embed_size
        self.num_heads     = num_heads
        assert embed_size % num_heads == 0
        self.head_dim      = embed_size // num_heads
        self.query_dense   = nn.Linear(embed_size, embed_size)
        self.key_dense     = nn.Linear(embed_size, embed_size)
        self.value_dense   = nn.Linear(embed_size, embed_size)
        self.combine_heads = nn.Linear(embed_size, embed_size)
        self.init_weights()
    
    def init_weights(self):
        init.xavier_uniform_(self.query_dense.weight)
        init.xavier_uniform_(self.key_dense.weight)
        init.xavier_uniform_(self.value_dense.weight)
        init.xavier_uniform_(self.combine_heads.weight)
        init.constant_(self.query_dense.bias,   0)
        init.constant_(self.key_dense.bias,     0)
        init.constant_(self.value_dense.bias,   0)
        init.constant_(self.combine_heads.bias, 0)
        
    def split_heads(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        x = x.reshape(batch_size, -1, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        x     = x.reshape(b, h * w, -1)
        query = self.split_heads(self.query_dense(x), b)
        key   = self.split_heads(self.key_dense(x), b)
        value = self.split_heads(self.value_dense(x), b)
        attention_weights = F.softmax(torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5), dim=-1)
        attention = torch.matmul(attention_weights, value)
        attention = attention.permute(0, 2, 1, 3).contiguous().reshape(b, -1, self.embed_size)
        output    = self.combine_heads(attention)

        return output.reshape(b, h, w, self.embed_size).permute(0, 3, 1, 2)
        

class Denoiser(nn.Module):
    
    def __init__(self, num_filters: int, kernel_size: int = 3, activation: str = "relu"):
        super(Denoiser, self).__init__()
        self.conv1        = nn.Conv2d(1, num_filters, kernel_size=kernel_size, padding=1)
        self.conv2        = nn.Conv2d(num_filters, num_filters, kernel_size=kernel_size, stride=2, padding=1)
        self.conv3        = nn.Conv2d(num_filters, num_filters, kernel_size=kernel_size, stride=2, padding=1)
        self.conv4        = nn.Conv2d(num_filters, num_filters, kernel_size=kernel_size, stride=2, padding=1)
        self.bottleneck   = MultiHeadSelfAttention(embed_size=num_filters, num_heads=4)
        self.up2          = nn.Upsample(scale_factor=2, mode="nearest")
        self.up3          = nn.Upsample(scale_factor=2, mode="nearest")
        self.up4          = nn.Upsample(scale_factor=2, mode="nearest")
        self.output_layer = nn.Conv2d(1, 1, kernel_size=kernel_size, padding=1)
        self.res_layer    = nn.Conv2d(num_filters, 1, kernel_size=kernel_size, padding=1)
        self.activation   = getattr(F, activation)
        self.init_weights()
   
    def init_weights(self):
        for layer in [self.conv1, self.conv2, self.conv3, self.conv4, self.output_layer, self.res_layer]:
            init.kaiming_uniform_(layer.weight, a=0, mode="fan_in", nonlinearity="relu")
            if layer.bias is not None:
                init.constant_(layer.bias, 0)
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.activation(self.conv1(x))
        x2 = self.activation(self.conv2(x1))
        x3 = self.activation(self.conv3(x2))
        x4 = self.activation(self.conv4(x3))
        x  = self.bottleneck(x4)
        x  = self.up4(x)
        x  = self.up3(x + x3)
        x  = self.up2(x + x2)
        x  = x + x1
        x  = self.res_layer(x)
        return torch.tanh(self.output_layer(x + x))
    
# endregion


# region Model

@MODELS.register(name="lyt_net_re", arch="lyt_net")
class LYTNet_RE(base.ImageEnhancementModel):
    """LYT-Net: Lightweight YUV Transformer-based Network for Low-Light Image
    Enhancement.
    
    References:
        https://github.com/albrateanu/LYT-Net
    """
    
    model_dir: core.Path    = current_dir
    arch     : str          = "lyt_net"
    tasks    : list[Task]   = [Task.LLIE]
    schemes  : list[Scheme] = [Scheme.SUPERVISED]
    zoo      : dict         = {}

    def __init__(
        self,
        in_channels : int = 1,
        out_channels: int = 3,
        filters     : int = 32,
        weights     : Any = None,
        *args, **kwargs
    ):
        super().__init__(
            name        = "lyt_net_re",
            in_channels = in_channels,
            weights     = weights,
            *args, **kwargs
        )
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.filters      = filters
        
        # Construct model
        self.process_y   = nn.Sequential(
            nn.Conv2d(self.in_channels, self.filters, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.process_cb  = nn.Sequential(
            nn.Conv2d(self.in_channels, self.filters, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.process_cr  = nn.Sequential(
            nn.Conv2d(self.in_channels, self.filters, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.denoiser_cb = Denoiser(self.filters // 2)
        self.denoiser_cr = Denoiser(self.filters // 2)
        self.lum_pool    = nn.MaxPool2d(8)
        self.lum_mhsa    = MultiHeadSelfAttention(embed_size=self.filters, num_heads=4)
        self.lum_up      = nn.Upsample(scale_factor=8, mode="nearest")
        self.lum_conv    = nn.Conv2d(self.filters,     self.filters, kernel_size=1, padding=0)
        self.ref_conv    = nn.Conv2d(self.filters * 2, self.filters, kernel_size=1, padding=0)
        self.msef        = MSEFBlock(self.filters)
        self.recombine   = nn.Conv2d(self.filters * 2, self.filters, kernel_size=3, padding=1)
        
        self.final_adjustments = nn.Conv2d(self.filters, self.out_channels, kernel_size=3, padding=1)
        
        # Loss
        self.loss = Loss(reduction="mean")
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)

    def init_weights(self, m: nn.Module):
        pass
    
    def forward(self, datapoint: dict, *args, **kwargs) -> dict:
        self.assert_datapoint(datapoint)
        x         = datapoint.get("image")
        ycbcr     = self._rgb_to_ycbcr(x)
        y, cb, cr = torch.split(ycbcr, 1, dim=1)
        cb        = self.denoiser_cb(cb) + cb
        cr        = self.denoiser_cr(cr) + cr
        
        y_processed  = self.process_y(y)
        cb_processed = self.process_cb(cb)
        cr_processed = self.process_cr(cr)
        
        ref   = torch.cat([cb_processed, cr_processed], dim=1)
        lum   = y_processed
        lum_1 = self.lum_pool(lum)
        lum_1 = self.lum_mhsa(lum_1)
        lum_1 = self.lum_up(lum_1)
        lum   = lum + lum_1
        
        ref      = self.ref_conv(ref)
        shortcut = ref
        ref      = ref + 0.2 * self.lum_conv(lum)
        ref      = self.msef(ref)
        ref      = ref + shortcut
        
        recombined = self.recombine(torch.cat([ref, lum], dim=1))
        enhanced   = self.final_adjustments(recombined)
        return {
            "enhanced": enhanced
        }
    
    @staticmethod
    def rgb_to_ycbcr(image: torch.Tensor) -> torch.Tensor:
        r, g, b = image[:, 0, :, :], image[:, 1, :, :], image[:, 2, :, :]
        y       =  0.299   * r + 0.587   * g + 0.114   * b
        u       = -0.14713 * r - 0.28886 * g + 0.436   * b + 0.5
        v       =  0.615   * r - 0.51499 * g - 0.10001 * b + 0.5
        yuv     = torch.stack((y, u, v), dim=1)
        return yuv
    
# endregion

#!/usr/bin/env python
# -*- coding: utf-8 -*-

__all__ = [
    "AttnBlock",
    "EncoderFFTime2d",
    "Normalize",
    "PaddedConv2d",
    "Snake",
    "UpscaleFFTime2d",
]

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


# --- Module ---
class PaddedConv2d(nn.Module):
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : int,
        stride      : int  = 1,
        dilation    : int  = 1,
        weight      : bool = True
    ):
        super().__init__()
        if weight:
            self.depth = weight_norm(
                nn.Conv2d(
                    in_channels  = in_channels,
                    out_channels = out_channels,
                    kernel_size  = kernel_size,
                    padding      = ((kernel_size - 1) // 2 * dilation, (kernel_size - 1) // 2 * dilation),
                    stride       = stride,
                    dilation     = (dilation, dilation)
                )
            )
        else:
            self.depth = (
                nn.Conv2d(
                    in_channels  = in_channels,
                    out_channels = out_channels,
                    kernel_size  = kernel_size,
                    padding      = ((kernel_size - 1) // 2 * dilation, (kernel_size - 1) // 2 * dilation),
                    stride       = stride,
                    dilation     = (dilation, dilation)
                )
            )
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.depth(x)


class EncoderFFTime2d(nn.Module):
    
    def __init__(
        self,
        channels   : int,
        kernel_size: int  = 3,
        stride     : int  = 1,
        dilation   : int  = 1,
        mul        : int  = 1,
        l          : bool = False,
        linear     : int  = 32,
        weight     : bool = False
    ):
        super().__init__()
        self.kernel    = kernel_size
        self.stride    = stride
        self.conv1     = PaddedConv2d(channels,       channels + mul, kernel_size=self.kernel, stride=stride, dilation=dilation, weight=weight)
        self.conv2     = PaddedConv2d(channels + mul, channels + mul, kernel_size=self.kernel, stride=1,      dilation=dilation, weight=weight)
        self.conv3     = PaddedConv2d(channels + mul, channels + mul, kernel_size=self.kernel, stride=1,      dilation=dilation, weight=weight)
        self.one       = PaddedConv2d(channels + mul, channels + mul, kernel_size= 1)
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(linear, channels + mul),
        )
        
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Downsample layer
        x     = torch.sin(self.conv1(x))
        emb   = (self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1]))
        x     = x + emb
        # Start block
        clone = x.clone()
        x     = torch.sin(self.conv2(x))
        x     = torch.sin(self.conv3(x))
        x     = x + clone
        return x


class UpscaleFFTime2d(nn.Module):
    
    def __init__(
        self,
        channels    : int,
        kernel_size : int  = 3,
        stride      : int  = 1,
        dilation    : int  = 1,
        scale_factor: int  = 2,
        mul         : int  = 1,
        linear      : int  = 32,
        weight      : bool = False
    ):
        super().__init__()
        self.stride = stride
        self.kernel = kernel_size
        self.conv1  = nn.Upsample(scale_factor=scale_factor)
        # self.conv1 = nn.ConvTranspose1d(channels, channels-mul, stride= 4,kernel_size=self.kernel, padding=0)
        self.conv2  = PaddedConv2d(channels,       channels - mul, kernel_size=self.kernel, dilation=dilation, weight=weight)
        self.conv3  = PaddedConv2d(channels - mul, channels - mul, kernel_size=self.kernel, dilation=dilation, weight=weight)
        self.conv4  = PaddedConv2d(channels - mul, channels - mul, kernel_size=self.kernel, dilation=dilation, weight=weight)
        self.one    = PaddedConv2d(channels - mul, channels - mul, kernel_size=1)
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(linear, channels - mul),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x     = self.conv1(x)
        x     = torch.sin(self.conv2(x))
        emb   = (self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1]))
        x     = x + emb
        # Start block
        clone = x.clone()
        x     = torch.sin(self.conv3(x))
        x     = torch.sin(self.conv4(x))
        x     = x + clone
        return x


def Normalize(in_channels: int, num_groups: int = 8):
    return nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class AttnBlock(nn.Module):
    
    def __init__(self, in_channels: int):
        super().__init__()
        self.norm     = Normalize(in_channels)
        self.q        = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k        = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v        = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_ = x
        h_ = self.norm(h_)
        q  = self.q(h_)
        k  = self.k(h_)
        v  = self.v(h_)

        # Compute attention
        b, c, h, w = q.shape
        q  = q.reshape(b, c, h * w)
        q  = q.permute(0, 2, 1)      # b, hw, c
        k  = k.reshape(b, c, h * w)  # b, c,  hw
        w_ = torch.bmm(q, k)         # b, hw, hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # Attend to values
        v  = v.reshape(b, c, h*w)
        w_ = w_.permute(0, 2, 1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)      # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, h, w)
        h_ = self.proj_out(h_)
        
        return x + h_


class Snake(nn.Module):
    
    def __init__(self, in_features: int):
        super().__init__()
        self.scale = nn.Parameter(torch.Tensor(1, in_features))
        nn.init.uniform_(self.scale, a=0.1, b=3)  # Initialize scale parameter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1,-1)
        x = x + (1.0 / self.scale) * pow(torch.sin(x * self.scale), 2)
        # x = torch.sin(x)
        # x = x * self.scale
        # x = torch.nn.functional.leaky_relu(x,.56)
        x = x.transpose(1, -1)
        return x


# --- Network ---
# 2D Unet that takes in a timestep T in encoder and decoder blocks
# uses sin activations, denoising target is done in the fourier domain instead of raw signal
class AutoFFTime2d(nn.Module):
    
    def __init__(
        self,
        blocks          : int,
        in_channels     : int,
        kernel_size     : int  = 3,
        layout          : int  = 2,  # Stereo, can be used for images just set spectrogram to false.
        channel_factor  : int  = 48,
        scale_factor    : int  = 2,
        encoder_dilation: int  = 4,
        decoder_dilation: int  = 1,
        neck            : bool = False,
        weight          : bool = False,
    ):
        super().__init__()
        self.neck             = neck
        self.layout           = layout
        self.channel_factor   = channel_factor
        self.scale_factor     = scale_factor
        self.encoder_dilation = encoder_dilation
        self.decoder_dilation = decoder_dilation
        
        self.linear  = nn.Linear(1, in_channels)
        self.conv    = PaddedConv2d(self.layout, in_channels, kernel_size=3, stride=1)
        self.act     = nn.SiLU()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        
        start = in_channels
        for block in range(blocks):
            stride = self.scale_factor
            self.encoder.append(
                EncoderFFTime2d(
                    channels    = start,
                    kernel_size = kernel_size,
                    stride      = stride,
                    dilation    = 1,
                    mul         = self.channel_factor,
                    weight      = weight,
                    linear      = in_channels,
                )
            )
            # Set channel size for next block
            start = start + self.channel_factor
        
        # Condition to add bottleneck
        self.linearAct = Snake(start)
        if self.neck:
            self.bottleneck = AttnBlock(start)

        for block in range(blocks):
            stride = 1
            self.decoder.append(
                UpscaleFFTime2d(
                    channels     = start,
                    kernel_size  = kernel_size,
                    stride       = stride,
                    dilation     = 1,
                    scale_factor = self.scale_factor,
                    mul          = self.channel_factor,
                    weight       = weight,
                    linear       = in_channels,
                )
            )
            # Set channel size for next block
            start = start - self.channel_factor
        
        self.process = nn.Conv2d(start, self.layout, kernel_size=1, stride=1, padding=0)

    def forward(self, mix: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t  = (self.linear(t.unsqueeze(-1).type(torch.float).to(mix.device)))
        x  = mix.clone()
        original = x[:, :self.layout, :].clone()
        x  = self.act(self.conv(x))
        og = x.clone()
        features = [x]
        for module in self.encoder:
            x = module(x, t)
            features.append(x)
        clone = x.clone()
        if self.neck:
            x = self.bottleneck(x.clone())
            x = clone + x
        for i, module in enumerate(self.decoder):
            index = i + 1
            x     = x[:, :, :features[-abs(index)].size(-2), :features[-abs(index)].size(-1)] + features[-abs(index)]
            x      = module(x, t)
        x = x[:, :, :original.size(-2), :original.size(-1)]
        x = (og[:, :, :original.size(-2), :original.size(-1)] + x)
        # split X
        # layer specific resnet
        cut   = self.process(x)
        synth = (cut[:, :self.layout, :original.size(-2), :original.size(-1)])
        out   = synth
        return out

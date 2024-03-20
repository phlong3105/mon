#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements DCEFormer (Zero-Reference Deep Curve Estimation
Transformer) models.
"""

from __future__ import annotations

__all__ = [
    "DCEFormerV1",
]

from typing import Any, Literal

import cv2
import torch
from einops import rearrange

from mon import core, nn
from mon.core import _callable
from mon.globals import MODELS, Scheme
from mon.nn import functional as F
from mon.vision.enhance.llie import base

console = core.console


# region Module

class PreNorm(nn.Module):
    
    def __init__(self, normalized_shape: int | list[int], fn: _callable):
        super().__init__()
        self.fn   = fn
        self.norm = nn.LayerNorm(normalized_shape)
    
    def forward(self, input: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        x = input
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class FeedForward(nn.Module):
    
    def __init__(self, in_channels: int, multiplier: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * multiplier, 1, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(in_channels * multiplier, in_channels * multiplier, 3, 1, 1, bias=False, groups=in_channels * multiplier),
            nn.GELU(),
            nn.Conv2d(in_channels * multiplier, in_channels, 1, 1, bias=False),
        )
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = self.net(x.permute(0, 3, 1, 2))
        y = y.permute(0, 2, 3, 1)
        return y
    
    
class IGMSA(nn.Module):
    """Illumination-Guided Multi-head Self-Attention."""
    
    def __init__(
        self,
        in_channels  : int,
        head_channels: int = 64,
        heads   	 : int = 8,
    ):
        super().__init__()
        self.num_heads = heads
        self.dim_head  = head_channels
        self.to_q 	   = nn.Linear(in_channels, head_channels * heads, bias=False)
        self.to_k 	   = nn.Linear(in_channels, head_channels * heads, bias=False)
        self.to_v 	   = nn.Linear(in_channels, head_channels * heads, bias=False)
        self.rescale   = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj      = nn.Linear(head_channels * heads, in_channels, bias=True)
        self.pos_emb   = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False, groups=in_channels),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False, groups=in_channels),
        )
        self.dim = in_channels
    
    def forward(self, input: torch.Tensor, illu_fea_trans: torch.Tensor) -> torch.Tensor:
        x 		   = input
        b, h, w, c = x.shape
        x 		   = x.reshape(b, h * w, c)
        q_inp 	   = self.to_q(x)
        k_inp 	   = self.to_k(x)
        v_inp 	   = self.to_v(x)
        illu_attn  = illu_fea_trans  # illu_fea: b,c,h,w -> b,h,w,c
        q, k, v, illu_attn = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.num_heads), (q_inp, k_inp, v_inp, illu_attn.flatten(1, 2)))
        v     = v * illu_attn
        # q: b,heads,hw,c
        q     = q.transpose(-2, -1)
        k     = k.transpose(-2, -1)
        v     = v.transpose(-2, -1)
        q     = F.normalize(q, dim=-1, p=2)
        k     = F.normalize(k, dim=-1, p=2)
        attn  = (k @ q.transpose(-2, -1))  # A = K^T*Q
        attn  = attn * self.rescale
        attn  = attn.softmax(dim=-1)
        x     = attn @ v  # b,heads,d,hw
        x     = x.permute(0, 3, 1, 2)  # Transpose
        x     = x.reshape(b, h * w, self.num_heads * self.dim_head)
        out_c = self.proj(x).view(b, h, w, c)
        out_p = self.pos_emb(v_inp.reshape(b, h, w, c).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        y = out_c + out_p
        return y


class IGAB(nn.Module):
    
    def __init__(
        self,
        in_channels  : int,
        head_channels: int = 64,
        heads        : int = 8,
        num_blocks   : int = 2,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(
                nn.ModuleList([
                    IGMSA(in_channels=in_channels, head_channels=head_channels, heads=heads),
                    PreNorm(in_channels, FeedForward(in_channels=in_channels))
                ])
            )
    
    def forward(self, input: torch.Tensor, illu_fea: torch.Tensor) -> torch.Tensor:
        x = input
        x = x.permute(0, 2, 3, 1)
        for (attn, ff) in self.blocks:
            x = attn(x, illu_fea_trans=illu_fea.permute(0, 2, 3, 1)) + x
            x = ff(x) + x
        y = x.permute(0, 3, 1, 2)
        return y
    
# endregion


# region Loss

class Loss(nn.Loss):
    
    def __init__(
        self,
        exp_patch_size : int   = 16,
        exp_mean_val   : float = 0.6,
        spa_num_regions: Literal[4, 8, 16, 24] = 8,  # 4
        spa_patch_size : int   = 4,     # 4
        weight_col     : float = 5,
        weight_edge    : float = 1,
        weight_exp     : float = 10,
        weight_kl      : float = 0.1,   # 5
        weight_spa     : float = 1,
        weight_tvA     : float = 1600,  # 200
        reduction      : str   = "mean",
        verbose        : bool  = False,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.weight_col  = weight_col
        self.weight_edge = weight_edge
        self.weight_exp  = weight_exp
        self.weight_kl   = weight_kl
        self.weight_spa  = weight_spa
        self.weight_tvA  = weight_tvA
        self.verbose     = verbose
        
        self.loss_col  = nn.ColorConstancyLoss(reduction=reduction)
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
    ) -> torch.Tensor:
        if isinstance(target, list | tuple):
            if len(target) == 2:
                a       = target[-2]
                enhance = target[-1]
            else:
                raise ValueError
        else:
            raise TypeError
       
        loss_col  = self.loss_col(input=enhance)                if self.weight_col  > 0 else 0
        loss_edge = self.loss_edge(input=enhance, target=input) if self.weight_edge > 0 else 0
        loss_exp  = self.loss_exp(input=enhance)                if self.weight_exp  > 0 else 0
        loss_kl   = self.loss_kl(input=enhance, target=input)   if self.weight_kl   > 0 else 0
        loss_spa  = self.loss_spa(input=enhance, target=input)  if self.weight_spa  > 0 else 0
        loss_tvA  = self.loss_tvA(input=a)                      if self.weight_tvA  > 0 else 0
        
        loss = (
            self.weight_col  * loss_col
            + self.weight_edge * loss_edge
            + self.weight_exp  * loss_exp
            + self.weight_tvA  * loss_tvA
            + self.weight_kl   * loss_kl
            + self.weight_spa  * loss_spa
        )
        
        if self.verbose:
            console.log(f"{self.loss_col.__str__():<30} : {loss_col}")
            console.log(f"{self.loss_edge.__str__():<30}: {loss_edge}")
            console.log(f"{self.loss_exp.__str__():<30} : {loss_exp}")
            console.log(f"{self.loss_kl.__str__():<30}  : {loss_kl}")
            console.log(f"{self.loss_spa.__str__():<30} : {loss_spa}")
            console.log(f"{self.loss_tvA.__str__():<30} : {loss_tvA}")
        return loss
        
# endregion


# region DCEFormer

@MODELS.register(name="dceformerv1")
class DCEFormerV1(base.LowLightImageEnhancementModel):
    """DCEFormer (Zero-Reference Deep Curve Estimation Transformer) models.
    
    Args:
        channels: The first layer's input channel. Default: ``3`` for RGB image.
        num_channels: Output channels for subsequent layers. Default: ``64``.
        num_iters: The number of convolutional layers in the model.
            Default: ``8``.
        scale_factor: Downsampling/upsampling ratio. Defaults: ``1``.
        gamma: Gamma value for dark channel prior. Default: ``2.8``.
        
    See Also: :class:`base.LowLightImageEnhancementModel`
    """
    
    _scheme: list[Scheme] = [Scheme.UNSUPERVISED, Scheme.ZEROSHOT]
    _zoo   : dict = {}

    def __init__(
        self,
        channels    : int       = 3,
        num_channels: int       = 32,
        num_blocks  : list[int] = [2, 4, 4],
        num_iters   : int       = 8,
        scale_factor: int       = 1,
        weights     : Any       = None,
        *args, **kwargs
    ):
        super().__init__(
            name     = "dceformerv1",
            channels = channels,
            weights  = weights,
            *args, **kwargs
        )
        
        # Populate hyperparameter values from pretrained weights
        if isinstance(self.weights, dict):
            channels      = self.weights.get("channels"    , channels)
            num_channels  = self.weights.get("num_channels", num_channels)
            num_blocks    = self.weights.get("num_blocks"  , num_blocks)
            num_iters     = self.weights.get("num_iters"   , num_iters)
            scale_factor  = self.weights.get("scale_factor", scale_factor)
        
        self._channels    = channels
        self.num_channels = num_channels
        self.num_blocks   = num_blocks
        self.num_iters    = num_iters
        self.scale_factor = scale_factor
        
        # Construct model
        self.illu_conv1 = nn.Conv2d(self.channels + 1,     self.num_channels, 1, 1, 1, bias=True)
        self.illu_conv2 = nn.Conv2d(self.num_channels,     self.num_channels, 3, 1, 0, bias=True, groups=self.channels + 1)
        self.conv1      = nn.Conv2d(self.channels + 1,     self.num_channels, 3, 1, 1, bias=True)
        self.conv2      = nn.Conv2d(self.num_channels,     self.num_channels, 3, 1, 1, bias=True)
        self.conv3      = nn.Conv2d(self.num_channels,     self.num_channels, 3, 1, 1, bias=True)
        self.conv4      = nn.Conv2d(self.num_channels,     self.num_channels, 3, 1, 1, bias=True)
        self.conv5      = nn.Conv2d(self.num_channels * 2, self.num_channels, 3, 1, 1, bias=True)
        self.conv6      = nn.Conv2d(self.num_channels * 2, self.num_channels, 3, 1, 1, bias=True)
        self.conv7      = nn.Conv2d(self.num_channels * 2, self.channels,     3, 1, 1, bias=True)
        self.igab4      = IGAB(in_channels=self.num_channels, num_blocks=self.num_blocks[0], head_channels=self.num_channels, heads=1)
        # self.igab5      = IGAB(in_channels=self.num_channels, num_blocks=self.num_blocks[1], head_channels=self.num_channels, heads=1)
        # self.igab6      = IGAB(in_channels=self.num_channels, num_blocks=self.num_blocks[2], head_channels=self.num_channels, heads=1)
        self.act        = nn.PReLU()
        self.upsample   = nn.UpsamplingBilinear2d(self.scale_factor)
        
        # Loss
        self._loss      = Loss()
        self._mae_loss  = nn.MAELoss(reduction="mean")
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self._init_weights)
    
    def _init_weights(self, m: nn.Module):
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
        pred = self.forward(input=input, *args, **kwargs)
        if target is not None:
            loss = self._mae_loss(pred[-1], target)
            # loss = self._loss(input, pred)
        else:
            loss = self._loss(input, pred)
        return pred[-1], loss
    
    def forward(
        self,
        input    : torch.Tensor,
        augment  : _callable = None,
        profile  : bool      = False,
        out_index: int       = -1,
        *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = input
        
        # Prior
        mean_c   = x.mean(dim=1).unsqueeze(1)
        x_concat = torch.cat([x, mean_c], dim=1)
        
        # Down-sampling
        x_down = x_concat
        if self.scale_factor != 1:
            x_down = F.interpolate(x, scale_factor=1 / self.scale_factor, mode="bilinear")
        
        # Illumination features
        illu_fea = self.illu_conv2(self.illu_conv1(x_down))
        # Encoder
        f1   = self.act(self.conv1(x_down))
        f2   = self.act(self.conv2(f1))
        f3   = self.act(self.conv3(f2))
        f4   = self.act(self.conv4(f3))
        # Attentive Decoder
        f4_a = self.igab4(f4, illu_fea)
        f5   = self.act(self.conv5(torch.cat([f3, f4_a], dim=1)))
        # f5   = self.act(self.conv5(torch.cat([f3, f4], dim=1)))
        # f5_a = self.igab5(f5, illu_fea)
        # f6   = self.act(self.conv6(torch.cat([f2, f5_a], dim=1)))
        f6   = self.act(self.conv6(torch.cat([f2, f5], dim=1)))
        # f6_a = self.igab6(f6, illu_fea)
        # a    =   F.tanh(self.conv7(torch.cat([f1, f6_a], dim=1)))
        a    =   F.tanh(self.conv7(torch.cat([f1, f6], dim=1)))
        
        # Up-sampling
        if self.scale_factor != 1:
            a = self.upsample(a)
        
        # Enhancement
        y = x
        for _ in range(self.num_iters):
            y = y + a * (torch.pow(y, 2) - y)
        
        """
        if not self.predicting:
            y = x
            for _ in range(self.num_iters):
                y = y + a * (torch.pow(y, 2) - y)
        else:
            y = x
            g = proc.get_guided_brightness_enhancement_map_prior(x, self.gamma, 9)
            for _ in range(self.num_iters):
                b = y * (1 - g)
                d = y * g
                y = b + d + a * (torch.pow(d, 2) - d)
        """
        
        return a, y
    
# endregion


# region Main

def run_dceformerv1():
    path   = core.Path("/home/longpham/10-workspace/11-code/mon/project/enhance/data/10.jpg")
    image  = cv2.imread(str(path))
    device = torch.device("cuda:0")
    input  = core.to_image_tensor(image, False, True, device)
    net    = DCEFormerV1().to(device)
    pred   = net(input)
    pred   = pred[-1]
    pred   = core.to_image_nparray(pred, False, True)
    cv2.imshow("Image", image)
    cv2.imshow("Relight", pred)
    cv2.waitKey(0)


if __name__ == "__main__":
    run_dceformerv1()

# endregion

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Modules.

This module provides various layers, blocks, and modules for the HVI-CIDNet model.
"""

from __future__ import annotations

__all__ = [
    "CIDNet",
    "RGB_HVI",
]

import torch
from einops import rearrange
from torch import nn, Tensor
from torch.nn import functional as F


# ==============================================================================
# region MODULES
# ==============================================================================

# --- Transformer Utils ---

class LayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: ``channels_last`` (default)
    or ``channels_first``.

    The ordering of the dimensions in the inputs ``channels_last`` corresponds
    to inputs with shape (B, H, W, C) while ``channels_first`` corresponds to
    inputs with shape (B, C, H, W).
    """

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        normalized_shape: tuple[int, ...],
        eps: float = 1e-6,
        data_format: str = "channels_first"
    ):
        super().__init__()

        # Validate inputs
        if data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError

        # Assign attributes
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape, )

    # --- Callable & Context Manager ---
    def forward(self, x: Tensor) -> Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
        else:
            raise ValueError(f"unsupported data format {self.data_format}.")


class NormDownsample(nn.Module):

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale: float = 0.5,
        use_norm: bool = False,
    ):
        super().__init__()

        # Assign attributes
        self.use_norm = use_norm
        if self.use_norm:
            self.norm = LayerNorm(out_channels)

        self.prelu = nn.PReLU()
        self.down = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.UpsamplingBilinear2d(scale_factor=scale),
        )

    # --- Callable & Context Manager ---
    def forward(self, x: Tensor) -> Tensor:
        x = self.down(x)
        x = self.prelu(x)
        if self.use_norm:
            x = self.norm(x)
            return x
        else:
            return x


class NormUpsample(nn.Module):

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale: float = 2,
        use_norm: bool = False
    ):
        super().__init__()

        # Assign attributes
        self.use_norm = use_norm
        if self.use_norm:
            self.norm = LayerNorm(out_channels)
        self.prelu = nn.PReLU()
        self.up_scale = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.UpsamplingBilinear2d(scale_factor=scale),
        )
        self.up = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

    # --- Callable & Context Manager ---
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        x = self.up_scale(x)
        x = torch.cat([x, y], dim=1)
        x = self.up(x)
        x = self.prelu(x)
        if self.use_norm:
            return self.norm(x)
        else:
            return x


# --- Attention Mechanism ---

class CAB(nn.Module):
    """Cross Attention Block."""

    # --- Lifecycle & Initialization ---
    def __init__(self, dim: int, num_heads: int, bias: bool):
        super().__init__()

        # Assign attributes
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.kv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    # --- Callable & Context Manager ---
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        b, c, h, w = x.shape

        q = self.q_dwconv(self.q(x))
        kv = self.kv_dwconv(self.kv(y))
        k, v = kv.chunk(2, dim=1)

        q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = nn.functional.softmax(attn,dim=-1)

        out = (attn @ v)
        out = rearrange(out, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out


class IEL(nn.Module):
    """Intensity Enhancement Layer"""

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        dim: int,
        ffn_expansion_factor: float = 2.66,
        bias: bool = False,
    ):
        super().__init__()

        # Assign attributes
        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
        self.dwconv1 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias)
        self.dwconv2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
        self.Tanh = nn.Tanh()

    # --- Callable & Context Manager ---
    def forward(self, x: Tensor) -> Tensor:
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x1 = self.Tanh(self.dwconv1(x1)) + x1
        x2 = self.Tanh(self.dwconv2(x2)) + x2
        x = x1 * x2
        x = self.project_out(x)
        return x


class HV_LCA(nn.Module):
    """Lightweight Cross-Attention"""

    # --- Lifecycle & Initialization ---
    def __init__(self, dim: int, num_heads: int, bias: bool = False):
        super().__init__()
        self.gdfn = IEL(dim)  # IEL and CDL have same structure
        self.norm = LayerNorm(dim)
        self.ffn = CAB(dim, num_heads, bias)

    # --- Callable & Context Manager ---
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        x = x + self.ffn(self.norm(x), self.norm(y))
        x = self.gdfn(self.norm(x))
        return x


class I_LCA(nn.Module):
    """Lightweight Cross-Attention"""

    # --- Lifecycle & Initialization ---
    def __init__(self, dim: int, num_heads: int, bias: bool = False):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.gdfn = IEL(dim)
        self.ffn = CAB(dim, num_heads, bias=bias)

    # --- Callable & Context Manager ---
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        x = x + self.ffn(self.norm(x), self.norm(y))
        x = x + self.gdfn(self.norm(x))
        return x

# endregion


# ==============================================================================
# region COLOR MODELS
# ==============================================================================

class RGB_HVI(nn.Module):

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        alpha: float = 1.0,
        alpha_s: float = 1.3,
        gated: bool = False,
        gated2: bool = False,
    ):
        super().__init__()

        # Assign attributes
        self.density_k = torch.nn.Parameter(torch.full([1], 0.2)) #  k is reciprocal to the paper mentioned
        self.gated = gated
        self.gated2 = gated2
        self.alpha = alpha
        self.alpha_s = alpha_s
        self.this_k = 0
        self.pi = 3.141592653589793

    # --- Callable & Context Manager ---
    def HVIT(self, image: Tensor) -> Tensor:
        eps = 1e-8
        device = image.device
        dtypes = image.dtype

        hue = torch.Tensor(image.shape[0], image.shape[2], image.shape[3]).to(device).to(dtypes)
        value = image.max(1)[0].to(dtypes)
        img_min = image.min(1)[0].to(dtypes)
        hue[image[:,2] == value] = 4.0 + ((image[:,0] - image[:,1]) / (value - img_min + eps)) [image[:,2] == value]
        hue[image[:,1] == value] = 2.0 + ((image[:,2] - image[:,0]) / (value - img_min + eps)) [image[:,1] == value]
        hue[image[:,0] == value] = (0.0 + ((image[:,1] - image[:,2]) / (value - img_min + eps)) [image[:,0] == value]) % 6

        hue[image.min(1)[0] == value] = 0.0
        hue = hue / 6.0

        saturation = (value - img_min) / (value + eps)
        saturation[value == 0] = 0

        hue = hue.unsqueeze(1)
        saturation = saturation.unsqueeze(1)
        value = value.unsqueeze(1)

        k = self.density_k
        self.this_k = k.item()

        color_sensitive = ((value * 0.5 * self.pi).sin() + eps).pow(k)
        ch = (2.0 * self.pi * hue).cos()
        cv = (2.0 * self.pi * hue).sin()
        H = color_sensitive * saturation * ch
        V = color_sensitive * saturation * cv
        I = value
        xyz = torch.cat([H, V, I], dim=1)
        return xyz

    def PHVIT(self, image: Tensor) -> Tensor:
        eps = 1e-8
        H,V,I = image[:,0, :, :],image[:,1, :, :],image[:,2, :, :]

        # Clip
        H = torch.clamp(H,-1,1)
        V = torch.clamp(V,-1,1)
        I = torch.clamp(I,0,1)

        v = I
        k = self.this_k
        color_sensitive = ((v * 0.5 * self.pi).sin() + eps).pow(k)
        H = H / (color_sensitive + eps)
        V = V / (color_sensitive + eps)
        H = torch.clamp(H, -1, 1)
        V = torch.clamp(V, -1, 1)
        h = torch.atan2(V + eps, H + eps) / (2 * self.pi)
        h = h % 1
        s = torch.sqrt(H ** 2 + V ** 2 + eps)

        if self.gated:
            s = s * self.alpha_s

        s = torch.clamp(s, 0, 1)
        v = torch.clamp(v, 0, 1)

        r = torch.zeros_like(h)
        g = torch.zeros_like(h)
        b = torch.zeros_like(h)

        hi = torch.floor(h * 6.0)
        f = h * 6.0 - hi
        p = v * (1. - s)
        q = v * (1. - (f * s))
        t = v * (1. - ((1. - f) * s))

        hi0 = hi == 0
        hi1 = hi == 1
        hi2 = hi == 2
        hi3 = hi == 3
        hi4 = hi == 4
        hi5 = hi == 5

        r[hi0] = v[hi0]
        g[hi0] = t[hi0]
        b[hi0] = p[hi0]

        r[hi1] = q[hi1]
        g[hi1] = v[hi1]
        b[hi1] = p[hi1]

        r[hi2] = p[hi2]
        g[hi2] = v[hi2]
        b[hi2] = t[hi2]

        r[hi3] = p[hi3]
        g[hi3] = q[hi3]
        b[hi3] = v[hi3]

        r[hi4] = t[hi4]
        g[hi4] = p[hi4]
        b[hi4] = v[hi4]

        r[hi5] = v[hi5]
        g[hi5] = p[hi5]
        b[hi5] = q[hi5]

        r = r.unsqueeze(1)
        g = g.unsqueeze(1)
        b = b.unsqueeze(1)
        rgb = torch.cat([r, g, b], dim=1)
        if self.gated2:
            rgb = rgb * self.alpha
        return rgb

# endregion


# ==============================================================================
# region NETWORKS
# ==============================================================================

class CIDNet(nn.Module):

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        channels: tuple = (36, 36, 72, 144),
        heads: tuple = (1, 2, 4, 8),
        norm: bool = False,
        alpha: float = 1.0,
        alpha_s: float = 1.3,
        gated: bool = False,
        gated2: bool = False,
    ):
        super().__init__()

        # Assign attributes
        [ch1, ch2, ch3, ch4] = channels
        [head1, head2, head3, head4] = heads

        # HV_ways
        self.HVE_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(3, ch1, 3, stride=1, padding=0,bias=False)
        )
        self.HVE_block1 = NormDownsample(ch1, ch2, use_norm = norm)
        self.HVE_block2 = NormDownsample(ch2, ch3, use_norm = norm)
        self.HVE_block3 = NormDownsample(ch3, ch4, use_norm = norm)

        self.HVD_block3 = NormUpsample(ch4, ch3, use_norm = norm)
        self.HVD_block2 = NormUpsample(ch3, ch2, use_norm = norm)
        self.HVD_block1 = NormUpsample(ch2, ch1, use_norm = norm)
        self.HVD_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(ch1, 2, 3, stride=1, padding=0,bias=False)
        )

        # I_ways
        self.IE_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(1, ch1, 3, stride=1, padding=0,bias=False),
        )
        self.IE_block1 = NormDownsample(ch1, ch2, use_norm = norm)
        self.IE_block2 = NormDownsample(ch2, ch3, use_norm = norm)
        self.IE_block3 = NormDownsample(ch3, ch4, use_norm = norm)

        self.ID_block3 = NormUpsample(ch4, ch3, use_norm=norm)
        self.ID_block2 = NormUpsample(ch3, ch2, use_norm=norm)
        self.ID_block1 = NormUpsample(ch2, ch1, use_norm=norm)
        self.ID_block0 =  nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(ch1, 1, 3, stride=1, padding=0,bias=False),
        )

        self.HV_LCA1 = HV_LCA(ch2, head2)
        self.HV_LCA2 = HV_LCA(ch3, head3)
        self.HV_LCA3 = HV_LCA(ch4, head4)
        self.HV_LCA4 = HV_LCA(ch4, head4)
        self.HV_LCA5 = HV_LCA(ch3, head3)
        self.HV_LCA6 = HV_LCA(ch2, head2)

        self.I_LCA1 = I_LCA(ch2, head2)
        self.I_LCA2 = I_LCA(ch3, head3)
        self.I_LCA3 = I_LCA(ch4, head4)
        self.I_LCA4 = I_LCA(ch4, head4)
        self.I_LCA5 = I_LCA(ch3, head3)
        self.I_LCA6 = I_LCA(ch2, head2)

        self.trans = RGB_HVI(
            alpha=alpha,
            alpha_s=alpha_s,
            gated=gated,
            gated2=gated2,
        )

    # --- Callable & Context Manager ---
    def forward(self, x: Tensor) -> Tensor:
        dtypes = x.dtype
        hvi = self.trans.HVIT(x)
        i = hvi[:, 2, :, :].unsqueeze(1).to(dtypes)

        # low
        i_enc0 = self.IE_block0(i)
        i_enc1 = self.IE_block1(i_enc0)
        hv_0 = self.HVE_block0(hvi)
        hv_1 = self.HVE_block1(hv_0)
        i_jump0 = i_enc0
        hv_jump0 = hv_0

        i_enc2 = self.I_LCA1(i_enc1, hv_1)
        hv_2 = self.HV_LCA1(hv_1, i_enc1)
        v_jump1 = i_enc2
        hv_jump1 = hv_2
        i_enc2 = self.IE_block2(i_enc2)
        hv_2 = self.HVE_block2(hv_2)

        i_enc3 = self.I_LCA2(i_enc2, hv_2)
        hv_3 = self.HV_LCA2(hv_2, i_enc2)
        v_jump2 = i_enc3
        hv_jump2 = hv_3
        i_enc3 = self.IE_block3(i_enc2)
        hv_3 = self.HVE_block3(hv_2)

        i_enc4 = self.I_LCA3(i_enc3, hv_3)
        hv_4 = self.HV_LCA3(hv_3, i_enc3)

        i_dec4 = self.I_LCA4(i_enc4,hv_4)
        hv_4 = self.HV_LCA4(hv_4, i_enc4)

        hv_3 = self.HVD_block3(hv_4, hv_jump2)
        i_dec3 = self.ID_block3(i_dec4, v_jump2)
        i_dec2 = self.I_LCA5(i_dec3, hv_3)
        hv_2 = self.HV_LCA5(hv_3, i_dec3)

        hv_2 = self.HVD_block2(hv_2, hv_jump1)
        i_dec2 = self.ID_block2(i_dec3, v_jump1)

        i_dec1 = self.I_LCA6(i_dec2, hv_2)
        hv_1 = self.HV_LCA6(hv_2, i_dec2)

        i_dec1 = self.ID_block1(i_dec1, i_jump0)
        i_dec0 = self.ID_block0(i_dec1)
        hv_1 = self.HVD_block1(hv_1, hv_jump0)
        hv_0 = self.HVD_block0(hv_1)

        output_hvi = torch.cat([hv_0, i_dec0], dim=1) + hvi
        output_rgb = self.trans.PHVIT(output_hvi)

        return output_rgb

    def HVIT(self, x: Tensor) -> Tensor:
        hvi = self.trans.HVIT(x)
        return hvi

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion

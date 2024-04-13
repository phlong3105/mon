#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements Uformer models."""

from __future__ import annotations

__all__ = [
    "Uformer",
    "UformerB",
    "UformerS",
    "UformerSFastleff",
    "UformerSNoshift",
    "UformerT",
]

import math
from typing import Any

import torch
import torch.utils.checkpoint as checkpoint
from einops import rearrange

from mon import core, nn
from mon.core import _callable, _size_2_t
from mon.globals import MODELS, Scheme, Task
from mon.nn import functional as F
from mon.vision.enhance.multitask import base

console = core.console


# region Module

def conv(
    in_channels : int,
    out_channels: int,
    kernel_size : _size_2_t,
    bias        : bool = False,
    stride      : int  = 1,
):
    return nn.Conv2d(
        in_channels  = in_channels,
        out_channels = out_channels,
        kernel_size  = kernel_size,
        padding      = (kernel_size  // 2),
        bias         = bias,
        stride       = stride,
    )


def window_partition(x: torch.Tensor, win_size: _size_2_t, dilation_rate: int = 1) -> torch.Tensor:
    b, h, w, c = x.shape
    if dilation_rate != 1:
        x = x.permute(0, 3, 1, 2)  # b, c, h, w
        assert type(dilation_rate) is int, ':param:`dilation_rate` should be a :class:s`int`'
        x = F.unfold(
            x,
            kernel_size = win_size,
            dilation    = dilation_rate,
            padding     = 4 * (dilation_rate - 1),
            stride      = win_size
        )  # b, c*Wh*Ww, h/Wh*w/Ww
        windows = x.permute(0, 2, 1).contiguous().view(-1, c, win_size, win_size)  # b' ,c ,Wh ,Ww
        windows = windows.permute(0, 2, 3, 1).contiguous()  # b' ,Wh ,Ww ,c
    else:
        x       = x.view(b, h // win_size, win_size, w // win_size, win_size, c)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, win_size,  win_size, c)  # b' ,Wh ,Ww ,c
    return windows


def window_reverse(windows: torch.Tensor, win_size: _size_2_t, h: int, w: int, dilation_rate: int = 1) -> torch.Tensor:
    # b' ,Wh ,Ww ,C
    b = int(windows.shape[0] / (h * w / win_size / win_size))
    x = windows.view(b, h // win_size, w // win_size, win_size, win_size, -1)
    if dilation_rate != 1:
        x = windows.permute(0, 5, 3, 4, 1, 2).contiguous()  # b, C*Wh*Ww, H/Wh*W/Ww
        x = F.fold(
            x,
            output_size = (h, w),
            kernel_size = win_size,
            dilation    = dilation_rate,
            padding     = 4 * (dilation_rate - 1),
            stride      = win_size
        )
    else:
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x


class FastLeFF(nn.Module):
    
    def __init__(
        self,
        in_channels : int       = 32,
        out_channels: int       = 128,
        act_layer   : _callable = nn.GELU,
        dropout     : float     = 0.4,
    ):
        super().__init__()
        self.linear1 = nn.Sequential(nn.Linear(in_channels, out_channels), act_layer())
        self.dwconv  = nn.Sequential(
            nn.DWConv2d(out_channels, kernel_size=3, stride=1, padding=1),
            act_layer()
        )
        self.linear2    = nn.Sequential(nn.Linear(out_channels, in_channels))
        self.dim        = in_channels
        self.hidden_dim = out_channels

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x  = input
        # bs x hw x c
        bs, hw, c = x.size()
        hh = int(math.sqrt(hw))
        x  = self.linear1(x)
        # Spatial restore
        x  = rearrange(x, ' b (h w) (c) -> b c h w ', h=hh, w=hh)
        # bs,hidden_dim,32x32
        x  = self.dwconv(x)
        # Flatten
        x  = rearrange(x, ' b c h w -> b (h w) c', h=hh, w=hh)
        x  = self.linear2(x)
        return x

    def flops(self, h: int, w: int) -> int:
        flops  = 0
        # fc1
        flops += h * w * self.dim * self.hidden_dim
        # dwconv
        flops += h * w * self.hidden_dim * 3 * 3
        # fc2
        flops += h * w * self.hidden_dim * self.dim
        # print("LeFF:{%.2f}" % (flops / 1e9))
        return flops


class Attention(nn.Module):
    
    def __init__(
        self,
        in_channels     : int,
        num_heads       : int,
        token_projection: str   = "linear",
        qkv_bias        : bool  = True,
        qk_scale        : Any   = None,
        attn_drop       : float = 0.0,
        proj_drop       : float = 0.0,
    ):
        super().__init__()
        self.in_channels      = in_channels
        self.num_heads        = num_heads
        head_channels         = in_channels // num_heads
        self.scale            = qk_scale or head_channels ** -0.5
        self.qkv              = nn.LinearProjection(in_channels, num_heads, in_channels // num_heads, bias=qkv_bias)
        self.token_projection = token_projection
        self.attn_drop        = nn.Dropout(attn_drop)
        self.proj             = nn.Linear(in_channels, in_channels)
        self.proj_drop        = nn.Dropout(proj_drop)
        self.softmax          = nn.Softmax(dim=-1)

    def forward(
        self,
        input  : torch.Tensor,
        attn_kv: torch.Tensor | None = None,
        mask   : torch.Tensor | None = None,
    ) -> torch.Tensor:
        x        = input
        b_, n, c = x.shape
        q, k, v  = self.qkv(x, attn_kv)
        q        = q * self.scale
        attn     = (q @ k.transpose(-2, -1))
        
        if mask is not None:
            nW   = mask.shape[0]
            # mask = repeat(mask, 'nW m n -> nW m (n d)',d = ratio)
            attn = attn.view(b_ // nW, nW, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.in_channels}, num_heads={self.num_heads}"

    def flops(self, q_num, kv_num) -> int:
        # Calculate flops for 1 window with token length of N
        # print(N, self.dim)
        flops = 0
        # N = self.win_size[0]*self.win_size[1]
        # nW = H*W/N
        # qkv = self.qkv(x)
        # flops += N * self.dim * 3 * self.dim
        flops += self.qkv.flops(q_num, kv_num)
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * q_num * (self.in_channels // self.num_heads) * kv_num
        #  x = (attn @ v)
        flops += self.num_heads * q_num * (self.in_channels // self.num_heads) * kv_num
        # x = self.proj(x)
        flops += q_num * self.in_channels * self.in_channels
        # print("MCA:{%.2f}" % (flops / 1e9))
        return flops


class MLP(nn.Module):
    
    def __init__(
        self,
        in_channels    : int,
        hidden_channels: int | None = None,
        out_channels   : int | None = None,
        act_layer      : _callable  = nn.GELU,
        dropout        : float      = 0.0
    ):
        super().__init__()
        out_channels    = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.fc1  = nn.Linear(in_channels, hidden_channels)
        self.act  = act_layer()
        self.fc2  = nn.Linear(hidden_channels, out_channels)
        self.drop = nn.Dropout(dropout)
        self.in_channels     = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels    = out_channels

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

    def flops(self, h: int, w: int) -> int:
        flops  = 0
        # fc1
        flops += h * w * self.in_channels * self.hidden_channels
        # fc2
        flops += h * w * self.hidden_channels * self.out_channels
        # print("MLP:{%.2f}" % (flops / 1e9))
        return flops


class LeFF(nn.Module):
    
    def __init__(
        self,
        channels       : int       = 32,
        hidden_channels: int       = 128,
        act_layer      : _callable = nn.GELU,
        dropout        : float     = 0.0,
        use_eca        : bool      = False,
        *args, **kwargs
    ):
        super().__init__()
        self.linear1 = nn.Sequential(nn.Linear(channels, hidden_channels), act_layer())
        self.dwconv  = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, groups=hidden_channels, kernel_size=3, stride=1, padding=1),
            act_layer()
        )
        self.linear2    = nn.Sequential(nn.Linear(hidden_channels, channels))
        self.dim        = channels
        self.hidden_dim = hidden_channels
        self.eca        = nn.ECA1d(channels=channels) if use_eca else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # bs x hw x c
        bs, hw, c = x.size()
        hh = int(math.sqrt(hw))
        x  = self.linear1(x)
        # Spatial restore
        x = rearrange(x, ' b (h w) (c) -> b c h w ', h=hh, w=hh)
        # bs,hidden_dim,32x32
        x = self.dwconv(x)
        # Flatten
        x = rearrange(x, ' b c h w -> b (h w) c', h=hh, w=hh)
        x = self.linear2(x)
        x = self.eca(x)
        return x

    def flops(self, h: int, w: int) -> int:
        flops = 0
        # fc1
        flops += h * w * self.dim * self.hidden_dim
        # dwconv
        flops += h * w * self.hidden_dim * 3 * 3
        # fc2
        flops += h * w * self.hidden_dim * self.dim
        # print("LeFF:{%.2f}" % (flops / 1e9))
        # eca
        if hasattr(self.eca, 'flops'):
            flops += self.eca.flops()
        return flops


class InputProj(nn.Module):
    
    def __init__(
        self,
        in_channels : int       = 3,
        out_channels: int       = 64,
        kernel_size : _size_2_t = 3,
        stride      : _size_2_t = 1,
        norm_layer  : _callable = None,
        act_layer   : _callable = nn.LeakyReLU
    ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=kernel_size // 2),
            act_layer(inplace=True)
        )
        if norm_layer is not None:
            self.norm = norm_layer(out_channels)
        else:
            self.norm = None
        self.in_channels  = in_channels
        self.out_channels = out_channels

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        b, c, h, w = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2).contiguous()  # b h*w c
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self, h: int, w: int) -> int:
        flops = 0
        # conv
        flops += h * w * self.in_channels * self.out_channels * 3 * 3
        if self.norm is not None:
            flops += h * w * self.out_channels
        # print("Input_proj:{%.2f}" % (flops / 1e9))
        return flops


class OutputProj(nn.Module):
    
    def __init__(
        self,
        in_channels : int       = 64,
        out_channels: int       = 3,
        kernel_size : _size_2_t = 3,
        stride      : _size_2_t = 1,
        norm_layer  : _callable = None,
        act_layer   : _callable = None
    ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=kernel_size // 2),
        )
        if act_layer is not None:
            self.proj.append(act_layer(inplace=True))
        if norm_layer is not None:
            self.norm = norm_layer(out_channels)
        else:
            self.norm = None
        self.in_channel  = in_channels
        self.out_channel = out_channels

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        b, l, c = x.shape
        h = int(math.sqrt(l))
        w = int(math.sqrt(l))
        x = x.transpose(1, 2).view(b, c, h, w)
        x = self.proj(x)
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self, h: int, w: int):
        flops = 0
        # conv
        flops += h * w * self.in_channel * self.out_channel * 3 * 3
        if self.norm is not None:
            flops += h * w * self.out_channel
        # print("Output_proj:{%.2f}" % (flops / 1e9))
        return flops


class LeWinTransformerBlock(nn.Module):
    
    def __init__(
        self,
        in_channels     : int,
        input_resolution: _size_2_t,
        num_heads       : int,
        window_size     : int       = 8,
        shift_size      : int       = 0,
        mlp_ratio       : float     = 4.0,
        qkv_bias        : bool      = True,
        qk_scale        : Any       = None,
        dropout         : float     = 0.0,
        attn_drop       : float     = 0.0,
        drop_path       : float     = 0.0,
        act_layer       : _callable = nn.GELU,
        norm_layer      : _callable = nn.LayerNorm,
        token_projection: str       = "linear",
        token_mlp       : str       = "leff",
        modulator       : bool      = False,
        cross_modulator : bool      = False,
    ):
        super().__init__()
        self.in_channels      = in_channels
        self.input_resolution = input_resolution
        self.num_heads        = num_heads
        self.window_size      = window_size
        self.shift_size       = shift_size
        self.mlp_ratio        = mlp_ratio
        self.token_mlp        = token_mlp
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, ":paramL`shift_size` must in :math:`[0 - :param`window_size`]`"

        if modulator:
            self.modulator = nn.Embedding(window_size * window_size, channels)  # modulator
        else:
            self.modulator = None

        if cross_modulator:
            self.cross_modulator = nn.Embedding(window_size * window_size, channels)  # cross_modulator
            self.cross_attn = Attention(
                channels         = channels,
                num_heads        = num_heads,
                qkv_bias         = qkv_bias,
                qk_scale         = qk_scale,
                attn_drop        = attn_drop,
                proj_drop        = dropout,
                token_projection = token_projection,
            )
            self.norm_cross = norm_layer(channels)
        else:
            self.cross_modulator = None

        self.norm1 = norm_layer(channels)
        self.attn  = nn.WindowAttention(
            channels         = channels,
            window_size      = core.to_2tuple(self.window_size),
            num_heads        = num_heads,
            qkv_bias         = qkv_bias,
            qk_scale         = qk_scale,
            attn_drop        = attn_drop,
            proj_drop        = dropout,
            token_projection = token_projection,
        )

        self.drop_path      = nn.DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2          = norm_layer(channels)
        mlp_hidden_channels = int(channels * mlp_ratio)
        if token_mlp in ["ffn", "mlp"]:
            self.mlp = MLP(
                in_channels     = channels,
                hidden_channels = mlp_hidden_channels,
                act_layer       = act_layer,
                dropout         = dropout,
            )
        elif token_mlp == "leff":
            self.mlp = LeFF(
                channels        = channels,
                hidden_channels = mlp_hidden_channels,
                act_layer       = act_layer,
                dropout         = dropout,
            )
        elif token_mlp == "fastleff":
            self.mlp = FastLeFF(
                in_channels  = channels,
                out_channels = mlp_hidden_channels,
                act_layer    = act_layer,
                dropout      = dropout,
            )
        else:
            raise Exception("FFN error!")

    def with_pos_embed(self, tensor: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        return tensor if pos is None else tensor + pos

    def extra_repr(self) -> str:
        return (
            f"dim={self.in_channels}, "
            f"input_resolution={self.input_resolution}, "
            f"num_heads={self.num_heads}, "
            f"win_size={self.window_size}, "
            f"shift_size={self.shift_size}, "
            f"mlp_ratio={self.mlp_ratio}, "
            f"modulator={self.modulator}"
        )

    def forward(self, input: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = input
        b, l, c = x.shape
        h = int(math.sqrt(l))
        w = int(math.sqrt(l))
        
        if mask is not None:
            input_mask         = F.interpolate(mask, size=(h, w)).permute(0, 2, 3, 1)
            input_mask_windows = window_partition(input_mask, self.window_size)  # nW, win_size, win_size, 1
            attn_mask = input_mask_windows.view(-1, self.window_size * self.window_size)  # nW, win_size * win_size
            attn_mask = attn_mask.unsqueeze(2) * attn_mask.unsqueeze(1)  # nW, win_size * win_size, win_size * win_size
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        
        # Shift mask
        if self.shift_size > 0:
            # Calculate attention mask for SW-MSA
            shift_mask = torch.zeros((1, h, w, 1)).type_as(x)
            h_slices   = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None)
            )
            w_slices   = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None)
            )
            cnt = 0
            for hs in h_slices:
                for ws in w_slices:
                    shift_mask[:, hs, ws, :] = cnt
                    cnt += 1
            
            shift_mask_windows = window_partition(shift_mask, self.window_size)  # nW, win_size, win_size, 1
            shift_mask_windows = shift_mask_windows.view(-1, self.window_size * self.window_size)  # nW, win_size*win_size
            shift_attn_mask    = shift_mask_windows.unsqueeze(1) - shift_mask_windows.unsqueeze(2)  # nW, win_size*win_size, win_size*win_size
            shift_attn_mask    = shift_attn_mask.masked_fill(shift_attn_mask != 0, float(-100.0)).masked_fill(shift_attn_mask == 0, float(0.0))
            attn_mask          = attn_mask + shift_attn_mask if attn_mask is not None else shift_attn_mask

        if self.cross_modulator is not None:
            shortcut = x
            x_cross  = self.norm_cross(x)
            x_cross  = self.cross_attn(x, self.cross_modulator.weight)
            x        = shortcut + x_cross
        
        shortcut = x
        x = self.norm1(x)
        x = x.view(b, h, w, c)

        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*b, win_size, win_size, c  N*c->c
        x_windows = x_windows.view(-1, self.window_size * self.window_size, c)  # nW*b, win_size*win_size, c

        # With_modulator
        if self.modulator is not None:
            wmsa_in = self.with_pos_embed(x_windows, self.modulator.weight)
        else:
            wmsa_in = x_windows
        
        # w-MSA/SW-MSA
        attn_windows = self.attn(wmsa_in, mask=attn_mask)  # nW*b, win_size*win_size, c

        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)
        shifted_x    = window_reverse(attn_windows, self.window_size, h, w)  # b h' w' c

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(b, h * w, c)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        del attn_mask
        return x

    def flops(self) -> int:
        flops = 0
        h, w  = self.input_resolution
        if self.cross_modulator is not None:
            flops += self.in_channels * h * w
            flops += self.cross_attn.flops(h * w, self.window_size * self.window_size)
        # norm1
        flops += self.in_channels * h * w
        # w-MSA/SW-MSA
        flops += self.attn.flops(h, w)
        # norm2
        flops += self.in_channels * h * w
        # mlp
        flops += self.mlp.flops(h, w)
        # print("LeWin:{%.2f}"%(flops/1e9))
        return flops


class BasicUformerLayer(nn.Module):
    
    def __init__(
        self,
        in_channels     : int,
        out_channels    : int,
        input_resolution: _size_2_t,
        depth           : int,
        num_heads       : int,
        window_size     : int,
        mlp_ratio       : float     = 4.0,
        qkv_bias        : bool      = True,
        qk_scale        : Any       = None,
        dropout         : float     = 0.0,
        attn_drop       : float     = 0.0,
        drop_path       : Any       = 0.0,
        norm_layer      : _callable = nn.LayerNorm,
        use_checkpoint  : bool      = False,
        token_projection: str       = "linear",
        token_mlp       : str       = "ffn",
        shift_flag      : bool      = True,
        modulator       : bool      = False,
        cross_modulator : bool      = False,
    ):
        super().__init__()
        self.in_channels      = in_channels
        self.input_resolution = input_resolution
        self.depth            = depth
        self.use_checkpoint   = use_checkpoint
        # Build blocks
        if shift_flag:
            self.blocks = nn.ModuleList([
                LeWinTransformerBlock(
                    channels         = in_channels,
                    input_resolution = input_resolution,
                    num_heads        = num_heads,
                    window_size      = window_size,
                    shift_size       = 0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio        = mlp_ratio,
                    qkv_bias         = qkv_bias,
                    qk_scale         = qk_scale,
                    attn_drop        = attn_drop,
                    drop_path        = drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer       = norm_layer,
                    token_projection = token_projection,
                    token_mlp        = token_mlp,
                    modulator        = modulator,
                    cross_modulator  = cross_modulator,
                )
                for i in range(depth)])
        else:
            self.blocks = nn.ModuleList([
                LeWinTransformerBlock(
                    channels         = in_channels,
                    input_resolution = input_resolution,
                    num_heads        = num_heads,
                    window_size      = window_size,
                    shift_size       = 0,
                    mlp_ratio        = mlp_ratio,
                    qkv_bias         = qkv_bias,
                    qk_scale         = qk_scale,
                    dropout          = dropout,
                    attn_drop        = attn_drop,
                    drop_path        = drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer       = norm_layer,
                    token_projection = token_projection,
                    token_mlp        = token_mlp,
                    modulator        = modulator,
                    cross_modulator  = cross_modulator
                )
                for i in range(depth)])

    def extra_repr(self) -> str:
        return (
            f"dim={self.in_channels}, "
            f"input_resolution={self.input_resolution}, "
            f"depth={self.depth}"
        )

    def forward(self, input: torch.Tensor, mask: torch.Tensor = None):
        x = input
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x, mask)
        return x

    def flops(self) -> int:
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        return flops

# endregion


# region Model

@MODELS.register(name="uformer")
class Uformer(base.MultiTaskImageEnhancementModel):
    """A General U-Shaped Transformer (Uformer) Network.
    
    References:
        `<https://github.com/ZhendongWang6/Uformer>`__

    See Also: :class:`base.MultiTaskImageEnhancementModel`
    """
    
    _tasks : list[Task]   = [Task.DEBLUR, Task.DENOISE, Task.DERAIN, Task.DESNOW, Task.LES, Task.LLIE]
    _scheme: list[Scheme] = [Scheme.SUPERVISED]
    _zoo   : dict = {}
    
    def __init__(
        self,
        image_size      : _size_2_t = 256,
        in_channels     : int       = 3,
        dd_in           : int       = 3,
        embed_channels  : int       = 32,
        depths          : list[int] = [2, 2, 2, 2, 2 , 2 , 2, 2, 2],
        num_heads       : list[int] = [1, 2, 4, 8, 16, 16, 8, 4, 2],
        window_size     : int       = 8,
        mlp_ratio       : float     = 4.0,
        qkv_bias        : bool      = True,
        qk_scale        : Any       = None,
        dropout         : float     = 0.0,
        attn_drop_rate  : float     = 0.0,
        drop_path_rate  : float     = 0.1,
        norm_layer      : _callable = nn.LayerNorm,
        patch_norm      : bool      = True,
        use_checkpoint  : bool      = False,
        token_projection: str       = "linear",
        token_mlp       : str       = "leff",
        downsample      : _callable = nn.DownsampleConv2d,
        upsample        : _callable = nn.UpsampleConv2d,
        shift_flag      : bool      = True,
        modulator       : bool      = False,
        cross_modulator : bool      = False,
        weights         : Any       = None,
        *args, **kwargs
    ):
        super().__init__(
            in_channels = in_channels,
            weights     = weights,
            *args, **kwargs
        )
        
        # Populate hyperparameter values from pretrained weights
        if isinstance(self.weights, dict):
            image_size  = self.weights.get("image_size" , image_size)
            dd_in       = self.weights.get("dd_in"      , dd_in)
            in_channels = self.weights.get("in_channels", in_channels)
        
        self.image_size       = core.parse_hw(image_size)
        self.in_channels      = in_channels
        self.dd_in            = dd_in
        self.embed_channels   = embed_channels
        self.depths           = depths
        self.num_heads        = num_heads
        self.window_size      = window_size
        self.mlp_ratio        = mlp_ratio
        self.qkv_bias         = qkv_bias
        self.qk_scale         = qk_scale
        self.dropout          = dropout
        self.attn_drop_rate   = attn_drop_rate
        self.patch_norm       = patch_norm
        self.token_projection = token_projection
        self.token_mlp        = token_mlp
        self.shift_flag       = shift_flag
        self.modulator        = modulator
        self.cross_modulator  = cross_modulator
        self.pos_drop         = nn.Dropout(p=self.dropout)
        self.num_enc_layers   = len(self.depths) // 2
        self.num_dec_layers   = len(self.depths) // 2
        
        # Stochastic depth
        enc_dpr  = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths[:self.num_enc_layers]))]
        conv_dpr = [drop_path_rate] * self.depths[4]
        dec_dpr  = enc_dpr[::-1]

        # Construct model
        # Input/Output
        self.input_proj = InputProj(
            in_channels  = self.dd_in,
            out_channels = self.embed_channels,
            kernel_size  = 3,
            stride       = 1,
            act_layer    = nn.LeakyReLU,
        )
        self.output_proj = OutputProj(
            in_channels  = 2 * self.embed_channels,
            out_channels = self.in_channels,
            kernel_size  = 3,
            stride       = 1,
        )
        
        # Encoder
        self.encoderlayer_0 = BasicUformerLayer(
            in_channels      = self.embed_channels,
            out_channels     = self.embed_channels,
            input_resolution = (self.image_size[0], self.image_size[1]),
            depth            = self.depths[0],
            num_heads        = self.num_heads[0],
            window_size      = self.window_size,
            mlp_ratio        = self.mlp_ratio,
            qkv_bias         = self.qkv_bias,
            qk_scale         = self.qk_scale,
            dropout          = self.dropout,
            attn_drop        = self.attn_drop_rate,
            drop_path        = enc_dpr[sum(self.depths[:0]):sum(self.depths[:1])],
            norm_layer       = norm_layer,
            use_checkpoint   = use_checkpoint,
            token_projection = self.token_projection,
            token_mlp        = self.token_mlp,
            shift_flag       = self.shift_flag,
        )
        self.downsample_0    = downsample(self.embed_channels, self.embed_channels * 2)
        self.encoderlayer_1 = BasicUformerLayer(
            in_channels      = self.embed_channels * 2,
            out_channels     = self.embed_channels * 2,
            input_resolution = (self.image_size[0] // 2, self.image_size[1] // 2),
            depth            = self.depths[1],
            num_heads        = self.num_heads[1],
            window_size      = self.window_size,
            mlp_ratio        = self.mlp_ratio,
            qkv_bias         = self.qkv_bias,
            qk_scale         = self.qk_scale,
            dropout          = self.dropout,
            attn_drop        = self.attn_drop_rate,
            drop_path        = enc_dpr[sum(self.depths[:1]): sum(self.depths[:2])],
            norm_layer       = norm_layer,
            use_checkpoint   = use_checkpoint,
            token_projection = self.token_projection,
            token_mlp        = self.token_mlp,
            shift_flag       = self.shift_flag,
        )
        self.downsample_1    = downsample(self.embed_channels * 2, self.embed_channels * 4)
        self.encoderlayer_2 = BasicUformerLayer(
            in_channels      = self.embed_channels * 4,
            out_channels     = self.embed_channels * 4,
            input_resolution = (self.image_size[0] // (2 ** 2), self.image_size[1] // (2 ** 2)),
            depth            = self.depths[2],
            num_heads        = self.num_heads[2],
            window_size      = self.window_size,
            mlp_ratio        = self.mlp_ratio,
            qkv_bias         = self.qkv_bias,
            qk_scale         = self.qk_scale,
            dropout          = self.dropout,
            attn_drop        = self.attn_drop_rate,
            drop_path        = enc_dpr[sum(self.depths[:2]): sum(self.depths[:3])],
            norm_layer       = norm_layer,
            use_checkpoint   = use_checkpoint,
            token_projection = self.token_projection,
            token_mlp        = self.token_mlp,
            shift_flag       = self.shift_flag,
        )
        self.downsample_2    = downsample(self.embed_channels * 4, self.embed_channels * 8)
        self.encoderlayer_3 = BasicUformerLayer(
            in_channels      = self.embed_channels * 8,
            out_channels     = self.embed_channels * 8,
            input_resolution = (self.image_size[0] // (2 ** 3), self.image_size[1] // (2 ** 3)),
            depth            = self.depths[3],
            num_heads        = self.num_heads[3],
            window_size      = self.window_size,
            mlp_ratio        = self.mlp_ratio,
            qkv_bias         = self.qkv_bias,
            qk_scale         = self.qk_scale,
            dropout          = self.dropout,
            attn_drop        = self.attn_drop_rate,
            drop_path        = enc_dpr[sum(self.depths[:3]):sum(self.depths[:4])],
            norm_layer       = norm_layer,
            use_checkpoint   = use_checkpoint,
            token_projection = self.token_projection,
            token_mlp        = self.token_mlp,
            shift_flag       = self.shift_flag,
        )
        self.downsample_3 = downsample(self.embed_channels * 8, self.embed_channels * 16)

        # Bottleneck
        self.conv = BasicUformerLayer(
            in_channels      = self.embed_channels * 16,
            out_channels     = self.embed_channels * 16,
            input_resolution = (self.image_size[0] // (2 ** 4), self.image_size[1] // (2 ** 4)),
            depth            = self.depths[4],
            num_heads        = self.num_heads[4],
            window_size      = self.window_size,
            mlp_ratio        = self.mlp_ratio,
            qkv_bias         = self.qkv_bias,
            qk_scale         = self.qk_scale,
            dropout          = self.dropout,
            attn_drop        = self.attn_drop_rate,
            drop_path        = conv_dpr,
            norm_layer       = norm_layer,
            use_checkpoint   = use_checkpoint,
            token_projection = self.token_projection,
            token_mlp        = self.token_mlp,
            shift_flag       = self.shift_flag,
        )

        # Decoder
        self.upsample_0     = upsample(self.embed_channels * 16, self.embed_channels * 8)
        self.decoderlayer_0 = BasicUformerLayer(
            in_channels      = self.embed_channels * 16,
            out_channels     = self.embed_channels * 16,
            input_resolution = (self.image_size[0] // (2 ** 3), self.image_size[1] // (2 ** 3)),
            depth            = self.depths[5],
            num_heads        = self.num_heads[5],
            window_size      = self.window_size,
            mlp_ratio        = self.mlp_ratio,
            qkv_bias         = self.qkv_bias,
            qk_scale         = self.qk_scale,
            dropout          = self.dropout,
            attn_drop        = self.attn_drop_rate,
            drop_path        = dec_dpr[:self.depths[5]],
            norm_layer       = norm_layer,
            use_checkpoint   = use_checkpoint,
            token_projection = self.token_projection,
            token_mlp        = self.token_mlp,
            shift_flag       = self.shift_flag,
            modulator        = self.modulator,
            cross_modulator  = self.cross_modulator,
        )
        self.upsample_1     = upsample(self.embed_channels * 16, self.embed_channels * 4)
        self.decoderlayer_1 = BasicUformerLayer(
            in_channels      = self.embed_channels * 8,
            out_channels     = self.embed_channels * 8,
            input_resolution = (self.image_size[0] // (2 ** 2), self.image_size[1] // (2 ** 2)),
            depth            = self.depths[6],
            num_heads        = self.num_heads[6],
            window_size      = self.window_size,
            mlp_ratio        = self.mlp_ratio,
            qkv_bias         = self.qkv_bias,
            qk_scale         = self.qk_scale,
            dropout          = self.dropout,
            attn_drop        = self.attn_drop_rate,
            drop_path        = dec_dpr[sum(self.depths[5:6]):sum(self.depths[5:7])],
            norm_layer       = norm_layer,
            use_checkpoint   = use_checkpoint,
            token_projection = self.token_projection,
            token_mlp        = self.token_mlp,
            shift_flag       = self.shift_flag,
            modulator        = self.modulator,
            cross_modulator  = self.cross_modulator
        )
        self.upsample_2     = upsample(self.embed_channels * 8, self.embed_channels * 2)
        self.decoderlayer_2 = BasicUformerLayer(
            in_channels      = self.embed_channels * 4,
            out_channels     = self.embed_channels * 4,
            input_resolution = (self.image_size[0] // 2, self.image_size[1] // 2),
            depth            = self.depths[7],
            num_heads        = self.num_heads[7],
            window_size      = self.window_size,
            mlp_ratio        = self.mlp_ratio,
            qkv_bias         = self.qkv_bias,
            qk_scale         = self.qk_scale,
            dropout          = self.dropout,
            attn_drop        = self.attn_drop_rate,
            drop_path        = dec_dpr[sum(depths[5: 7]): sum(depths[5: 8])],
            norm_layer       = norm_layer,
            use_checkpoint   = use_checkpoint,
            token_projection = self.token_projection,
            token_mlp        = self.token_mlp,
            shift_flag       = self.shift_flag,
            modulator        = self.modulator,
            cross_modulator  = self.cross_modulator,
        )
        self.upsample_3     = upsample(self.embed_channels * 4, self.embed_channels)
        self.decoderlayer_3 = BasicUformerLayer(
            in_channels      = self.embed_channels * 2,
            out_channels     = self.embed_channels * 2,
            input_resolution = (self.image_size[0], self.image_size[0]),
            depth            = self.depths[8],
            num_heads        = self.num_heads[8],
            window_size      = self.window_size,
            mlp_ratio        = self.mlp_ratio,
            qkv_bias         = self.qkv_bias,
            qk_scale         = self.qk_scale,
            dropout          = self.dropout,
            attn_drop        = self.attn_drop_rate,
            drop_path        = dec_dpr[sum(self.depths[5: 8]):sum(self.depths[5: 9])],
            norm_layer       = norm_layer,
            use_checkpoint   = use_checkpoint,
            token_projection = self.token_projection,
            token_mlp        = self.token_mlp,
            shift_flag       = self.shift_flag,
            modulator        = self.modulator,
            cross_modulator  = self.cross_modulator,
        )
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self._init_weights)
    
    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)
    
    @torch.jit.ignore
    def no_weight_decay(self):
        return {"absolute_pos_embed"}
    
    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"relative_position_bias_table"}
    
    def extra_repr(self) -> str:
        return (
            f"embed_dim={self.embed_dim}, "
            f"token_projection={self.token_projection}, "
            f"token_mlp={self.token_mlp}, "
            f"win_size={self.window_size}"
        )
    
    def flops(self) -> int:
        flops  = 0
        # Input Projection
        flops += self.input_proj.flops(self.reso, self.reso)
        # Encoder
        flops += self.encoderlayer_0.flops() + self.downsample_0.flops(self.reso,           self.reso)
        flops += self.encoderlayer_1.flops() + self.downsample_1.flops(self.reso // 2,      self.reso // 2)
        flops += self.encoderlayer_2.flops() + self.downsample_2.flops(self.reso // 2 ** 2, self.reso // 2 ** 2)
        flops += self.encoderlayer_3.flops() + self.downsample_3.flops(self.reso // 2 ** 3, self.reso // 2 ** 3)
        # Bottleneck
        flops += self.conv.flops()
        # Decoder
        flops += self.upsample_0.flops(self.reso // 2 ** 4, self.reso // 2 ** 4) + self.decoderlayer_0.flops()
        flops += self.upsample_1.flops(self.reso // 2 ** 3, self.reso // 2 ** 3) + self.decoderlayer_1.flops()
        flops += self.upsample_2.flops(self.reso // 2 ** 2, self.reso // 2 ** 2) + self.decoderlayer_2.flops()
        flops += self.upsample_3.flops(self.reso // 2,           self.reso // 2) + self.decoderlayer_3.flops()
        # Output Projection
        flops += self.output_proj.flops(self.reso, self.reso)
        return flops
    
    def expand2square(self, input: torch.Tensor, factor: float = 16.0) -> tuple[torch.Tensor, torch.Tensor]:
        """Crop the input tensor. Used in inference pass."""
        _, _, h, w = input.size()
        x     = int(math.ceil(max(h, w) / float(factor)) * factor)
        image = torch.zeros(1, 3, x, x).type_as(input)  # 3, h, w
        mask  = torch.zeros(1, 1, x, x).type_as(input)
        image[:, :, ((x - h)//2):((x - h)//2 + h), ((x - w)//2):((x - w)//2 + w)] = input
        mask[:, :, ((x - h)//2):((x - h)//2 + h), ((x - w)//2):((x - w)//2 + w)].fill_(1)
        return image, mask
    
    def forward(
        self,
        input    : torch.Tensor,
        augment  : _callable = None,
        profile  : bool      = False,
        out_index: int       = -1,
        *args, **kwargs
    ) -> torch.Tensor:
        mask = None
        if self.predicting:
            _, _, h, w  = input.shape
            input, mask = self.expand2square(input, factor=self.image_size[0])
        
        x = input
        # Input Projection
        y = self.input_proj(x)
        y = self.pos_drop(y)
        # Encoder
        conv0 = self.encoderlayer_0(y, mask=mask)
        pool0 = self.downsample_0(conv0)
        conv1 = self.encoderlayer_1(pool0, mask=mask)
        pool1 = self.downsample_1(conv1)
        conv2 = self.encoderlayer_2(pool1, mask=mask)
        pool2 = self.downsample_2(conv2)
        conv3 = self.encoderlayer_3(pool2, mask=mask)
        pool3 = self.downsample_3(conv3)
        # Bottleneck
        conv4 = self.conv(pool3, mask=mask)
        # Decoder
        up0     = self.upsample_0(conv4)
        deconv0 = torch.cat([up0, conv3], -1)
        deconv0 = self.decoderlayer_0(deconv0, mask=mask)
        up1     = self.upsample_1(deconv0)
        deconv1 = torch.cat([up1, conv2], -1)
        deconv1 = self.decoderlayer_1(deconv1, mask=mask)
        up2     = self.upsample_2(deconv1)
        deconv2 = torch.cat([up2, conv1], -1)
        deconv2 = self.decoderlayer_2(deconv2, mask=mask)
        up3     = self.upsample_3(deconv2)
        deconv3 = torch.cat([up3, conv0], -1)
        deconv3 = self.decoderlayer_3(deconv3, mask=mask)
        # Output Projection
        y = self.output_proj(deconv3)
        y = x + y if self.dd_in == 3 else y
        
        if self.predicting:
            y = torch.masked_select(y, mask.bool()).reshape(1, 3, h, w)
            # y = torch.clamp(y, 0, 1)
            # y = torch.clamp(y, 0, 1).cpu().detach().numpy().squeeze().transpose((1, 2, 0))
            # y = skimage.util.img_as_ubyte(y)
        
        return y
    

@MODELS.register(name="uformer_t")
class UformerT(Uformer):
    """Uformer Tiny model.
    
    References:
        `<https://github.com/ZhendongWang6/Uformer>`__
    """
    
    _zoo: dict = {}
    
    def __init__(self, args, **kwargs):
        super().__init__(
            name             = "uformer_t",
            embed_channels   = 16,
            window_size      = 8,
            token_projection = "linear",
            token_mlp        = "leff",
            shift_flag       = True,
            modulator        = True,
            *args, **kwargs
        )


@MODELS.register(name="uformer_s")
class UformerS(Uformer):
    """Uformer Small model.
    
    References:
        `<https://github.com/ZhendongWang6/Uformer>`__
    """
    
    _zoo: dict = {}
    
    def __init__(
        self,
        image_size: _size_2_t = 256,
        *args, **kwargs
    ):
        super().__init__(
            name             = "uformer_s",
            image_size       = image_size,
            embed_channels   = 32,
            window_size      = 8,
            token_projection = "linear",
            token_mlp        = "leff",
            shift_flag       = True,
            modulator        = True,
            *args, **kwargs
        )


@MODELS.register(name="uformer_s_noshift")
class UformerSNoshift(Uformer):
    
    _zoo: dict = {}
    
    def __init__(
        self,
        image_size: _size_2_t = 256,
        *args, **kwargs
    ):
        super().__init__(
            name             = "uformer_s_noshift",
            image_size       = image_size,
            embed_channels   = 32,
            window_size      = 8,
            token_projection = "linear",
            token_mlp        = "leff",
            shift_flag       = False,
            modulator        = True,
            *args, **kwargs
        )


@MODELS.register(name="uformer_s_fastleff")
class UformerSFastleff(Uformer):
    
    _zoo: dict = {}
    
    def __init__(
        self,
        image_size: _size_2_t = 256,
        *args, **kwargs
    ):
        super().__init__(
            name             = "uformer_s_fastleff",
            image_size       = image_size,
            embed_channels   = 32,
            depths           = [1, 2, 8, 8, 2, 8, 8, 2, 1],
            window_size      = 8,
            token_projection = "linear",
            token_mlp        = "fastleff",
            shift_flag       = True,
            modulator        = True,
            *args, **kwargs
        )


@MODELS.register(name="uformer_b")
class UformerB(Uformer):
    """Uformer Big model.
    
    References:
        `<https://github.com/ZhendongWang6/Uformer>`__
    """
    
    _zoo: dict = {
        "gopro" : {
            "url"        : None,
            "path"       : "uformer_b/uformer_b_gopro",
            "num_classes": None,
            "image_size" : 128,
            "dd_in"      : 3,
            "map": {
                "dowsample_0": "downsample_0",
                "dowsample_1": "downsample_1",
                "dowsample_2": "downsample_2",
                "dowsample_3": "downsample_3",
            },
        },
        "gtrain": {
            "url"        : None,
            "path"       : "uformer_b/uformer_b_gtrain",
            "num_classes": None,
            "image_size" : 128,
            "dd_in"      : 3,
            "map": {
                "dowsample_0": "downsample_0",
                "dowsample_1": "downsample_1",
                "dowsample_2": "downsample_2",
                "dowsample_3": "downsample_3",
            },
        },
        "sidd"  : {
            "url"        : None,
            "path"       : "uformer_b/uformer_b_sidd",
            "num_classes": None,
            "image_size" : 128,
            "dd_in"      : 3,
            "map": {
                "dowsample_0": "downsample_0",
                "dowsample_1": "downsample_1",
                "dowsample_2": "downsample_2",
                "dowsample_3": "downsample_3",
            },
        },
    }
    
    def __init__(
        self,
        dd_in     : int       = 3,
        image_size: _size_2_t = 256,
        *args, **kwargs
    ):
        super().__init__(
            name             = "uformer_b",
            image_size       = image_size,
            dd_in            = dd_in,
            embed_channels   = 32,
            depths           = [1, 2, 8, 8, 2, 8, 8, 2, 1],
            window_size      = 8,
            token_projection = "linear",
            token_mlp        = "leff",
            shift_flag       = True,
            modulator        = True,
            *args, **kwargs
        )
        
# endregion

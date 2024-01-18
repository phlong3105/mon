#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements Uformer models."""

from __future__ import annotations

__all__ = [

]

from typing import Any, Callable

import torch
import torch.utils.checkpoint as checkpoint
from einops import rearrange

from mon.globals import MODELS
from mon.nn.typing import _callable, _size_2_t
from mon.vision import core, nn
from mon.vision.enhance.universal import base
from mon.vision.nn import functional as F

console      = core.console
math         = core.math
_current_dir = core.Path(__file__).absolute().parent


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


class FastLeFF(nn.Module):
    
    def __init__(
        self,
        in_channels : int       = 32,
        out_channels: int       = 128,
        act_layer   : _callable = nn.GELU,
        drop        : float     = 0.4,
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
        dim             : int,
        num_heads       : int,
        token_projection: str   = "linear",
        qkv_bias        : bool  = True,
        qk_scale        : Any   = None,
        attn_drop       : float = 0.0,
        proj_drop       : float = 0.0,
    ):
        super().__init__()
        self.dim              = dim
        self.num_heads        = num_heads
        head_dim              = dim // num_heads
        self.scale            = qk_scale or head_dim ** -0.5
        self.qkv              = nn.LinearProjection(dim, num_heads, dim // num_heads, bias=qkv_bias)
        self.token_projection = token_projection
        self.attn_drop        = nn.Dropout(attn_drop)
        self.proj             = nn.Linear(dim, dim)
        self.proj_drop        = nn.Dropout(proj_drop)
        self.softmax          = nn.Softmax(dim=-1)

    def forward(
        self,
        input  : torch.Tensor,
        attn_kv: torch.Tensor | None = None,
        mask   : torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = input
        B_, N, C = x.shape
        q, k, v  = self.qkv(x, attn_kv)
        q        = q * self.scale
        attn     = (q @ k.transpose(-2, -1))
        
        if mask is not None:
            nW   = mask.shape[0]
            # mask = repeat(mask, 'nW m n -> nW m (n d)',d = ratio)
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, num_heads={self.num_heads}"

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
        flops += self.num_heads * q_num * (self.dim // self.num_heads) * kv_num
        #  x = (attn @ v)
        flops += self.num_heads * q_num * (self.dim // self.num_heads) * kv_num
        # x = self.proj(x)
        flops += q_num * self.dim * self.dim
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
        self.in_features     = in_channels
        self.hidden_features = hidden_channels
        self.out_features    = out_channels

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

    def flops(self, h: int, w: int) -> int:
        flops = 0
        # fc1
        flops += h * w * self.in_features * self.hidden_features
        # fc2
        flops += h * w * self.hidden_features * self.out_features
        # print("MLP:{%.2f}" % (flops / 1e9))
        return flops


class LeFF(nn.Module):
    
    def __init__(
        self,
        dim        = 32,
        hidden_dim = 128,
        act_layer  = nn.GELU,
        drop: float       = 0.0,
        use_eca: bool    = False
    ):
        super().__init__()
        self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim),
                                     act_layer())
        self.dwconv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, groups=hidden_dim, kernel_size=3,
                      stride=1, padding=1),
            act_layer())
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim, dim))
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.eca = nn.ECA1d(channels=dim) if use_eca else nn.Identity()

    def forward(self, x):
        # bs x hw x c
        bs, hw, c = x.size()
        hh = int(math.sqrt(hw))

        x = self.linear1(x)

        # spatial restore
        x = rearrange(x, ' b (h w) (c) -> b c h w ', h=hh, w=hh)
        # bs,hidden_dim,32x32

        x = self.dwconv(x)

        # flaten
        x = rearrange(x, ' b c h w -> b (h w) c', h=hh, w=hh)

        x = self.linear2(x)
        x = self.eca(x)

        return x

    def flops(self, H, W):
        flops = 0
        # fc1
        flops += H * W * self.dim * self.hidden_dim
        # dwconv
        flops += H * W * self.hidden_dim * 3 * 3
        # fc2
        flops += H * W * self.hidden_dim * self.dim
        print("LeFF:{%.2f}" % (flops / 1e9))
        # eca
        if hasattr(self.eca, 'flops'):
            flops += self.eca.flops()
        return flops


#########################################
########### window operation#############
def window_partition(x, win_size, dilation_rate=1):
    B, H, W, C = x.shape
    if dilation_rate != 1:
        x = x.permute(0, 3, 1, 2)  # B, C, H, W
        assert type(dilation_rate) is int, 'dilation_rate should be a int'
        x = F.unfold(x, kernel_size=win_size, dilation=dilation_rate,
                     padding=4 * (dilation_rate - 1),
                     stride=win_size)  # B, C*Wh*Ww, H/Wh*W/Ww
        windows = x.permute(0, 2, 1).contiguous().view(-1, C, win_size,
                                                       win_size)  # B' ,C ,Wh ,Ww
        windows = windows.permute(0, 2, 3, 1).contiguous()  # B' ,Wh ,Ww ,C
    else:
        x = x.view(B, H // win_size, win_size, W // win_size, win_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, win_size,
                                                                win_size,
                                                                C)  # B' ,Wh ,Ww ,C
    return windows


def window_reverse(windows, win_size, H, W, dilation_rate=1):
    # B' ,Wh ,Ww ,C
    B = int(windows.shape[0] / (H * W / win_size / win_size))
    x = windows.view(B, H // win_size, W // win_size, win_size, win_size, -1)
    if dilation_rate != 1:
        x = windows.permute(0, 5, 3, 4, 1,
                            2).contiguous()  # B, C*Wh*Ww, H/Wh*W/Ww
        x = F.fold(x, (H, W), kernel_size=win_size, dilation=dilation_rate,
                   padding=4 * (dilation_rate - 1), stride=win_size)
    else:
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


#########################################
# Downsample Block
class Downsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Downsample, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=4, stride=2,
                      padding=1),
        )
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        B, L, C = x.shape
        # import pdb;pdb.set_trace()
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        out = self.conv(x).flatten(2).transpose(1, 2).contiguous()  # B H*W C
        return out

    def flops(self, H, W):
        flops = 0
        # conv
        flops += H / 2 * W / 2 * self.in_channel * self.out_channel * 4 * 4
        print("Downsample:{%.2f}" % (flops / 1e9))
        return flops


# Upsample Block
class Upsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Upsample, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2,
                               stride=2),
        )
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        out = self.deconv(x).flatten(2).transpose(1, 2).contiguous()  # B H*W C
        return out

    def flops(self, H, W):
        flops = 0
        # conv
        flops += H * 2 * W * 2 * self.in_channel * self.out_channel * 2 * 2
        print("Upsample:{%.2f}" % (flops / 1e9))
        return flops


# Input Projection
class InputProj(nn.Module):
    def __init__(self, in_channel=3, out_channel=64, kernel_size=3, stride=1,
                 norm_layer=None, act_layer=nn.LeakyReLU):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride,
                      padding=kernel_size // 2),
            act_layer(inplace=True)
        )
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2).contiguous()  # B H*W C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self, H, W):
        flops = 0
        # conv
        flops += H * W * self.in_channel * self.out_channel * 3 * 3

        if self.norm is not None:
            flops += H * W * self.out_channel
        print("Input_proj:{%.2f}" % (flops / 1e9))
        return flops


# Output Projection
class OutputProj(nn.Module):
    def __init__(self, in_channel=64, out_channel=3, kernel_size=3, stride=1,
                 norm_layer=None, act_layer=None):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride,
                      padding=kernel_size // 2),
        )
        if act_layer is not None:
            self.proj.add_module(act_layer(inplace=True))
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.proj(x)
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self, H, W):
        flops = 0
        # conv
        flops += H * W * self.in_channel * self.out_channel * 3 * 3

        if self.norm is not None:
            flops += H * W * self.out_channel
        print("Output_proj:{%.2f}" % (flops / 1e9))
        return flops


class LeWinTransformerBlock(nn.Module):
    
    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        win_size         = 8,
        shift_size       = 0,
        mlp_ratio        = 4.0,
        qkv_bias         = True,
        qk_scale         = None,
        drop             = 0.0,
        attn_drop        = 0.0,
        drop_path        = 0.0,
        act_layer        = nn.GELU,
        norm_layer       = nn.LayerNorm,
        token_projection = 'linear',
        token_mlp        = 'leff',
        modulator        = False,
        cross_modulator  = False,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.win_size = win_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.token_mlp = token_mlp
        if min(self.input_resolution) <= self.win_size:
            self.shift_size = 0
            self.win_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.win_size, "shift_size must in 0-win_size"

        if modulator:
            self.modulator = nn.Embedding(win_size * win_size, dim)  # modulator
        else:
            self.modulator = None

        if cross_modulator:
            self.cross_modulator = nn.Embedding(win_size * win_size,
                                                dim)  # cross_modulator
            self.cross_attn = Attention(dim, num_heads, qkv_bias=qkv_bias,
                                        qk_scale=qk_scale, attn_drop=attn_drop,
                                        proj_drop=drop,
                                        token_projection=token_projection, )
            self.norm_cross = norm_layer(dim)
        else:
            self.cross_modulator = None

        self.norm1 = norm_layer(dim)
        self.attn  = nn.WindowAttention(
            dim,
            window_size=core.to_2tuple(self.win_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            token_projection=token_projection,
        )

        self.drop_path = nn.DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2     = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if token_mlp in ['ffn', 'mlp']:
            self.mlp = MLP(in_channels=dim, hidden_channels=mlp_hidden_dim,
                           act_layer=act_layer, dropout=drop)
        elif token_mlp == 'leff':
            self.mlp = LeFF(dim, mlp_hidden_dim, act_layer=act_layer, drop=drop)

        elif token_mlp == 'fastleff':
            self.mlp = FastLeFF(dim, mlp_hidden_dim, act_layer=act_layer, drop=drop)
        else:
            raise Exception("FFN error!")

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"win_size={self.win_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio},modulator={self.modulator}"

    def forward(self, x, mask=None):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))

        ## input mask
        if mask != None:
            input_mask = F.interpolate(mask, size=(H, W)).permute(0, 2, 3, 1)
            input_mask_windows = window_partition(input_mask,
                                                  self.win_size)  # nW, win_size, win_size, 1
            attn_mask = input_mask_windows.view(-1,
                                                self.win_size * self.win_size)  # nW, win_size*win_size
            attn_mask = attn_mask.unsqueeze(2) * attn_mask.unsqueeze(
                1)  # nW, win_size*win_size, win_size*win_size
            attn_mask = attn_mask.masked_fill(attn_mask != 0,
                                              float(-100.0)).masked_fill(
                attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        ## shift mask
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            shift_mask = torch.zeros((1, H, W, 1)).type_as(x)
            h_slices = (slice(0, -self.win_size),
                        slice(-self.win_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.win_size),
                        slice(-self.win_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    shift_mask[:, h, w, :] = cnt
                    cnt += 1
            shift_mask_windows = window_partition(shift_mask,
                                                  self.win_size)  # nW, win_size, win_size, 1
            shift_mask_windows = shift_mask_windows.view(-1,
                                                         self.win_size * self.win_size)  # nW, win_size*win_size
            shift_attn_mask = shift_mask_windows.unsqueeze(
                1) - shift_mask_windows.unsqueeze(
                2)  # nW, win_size*win_size, win_size*win_size
            shift_attn_mask = shift_attn_mask.masked_fill(shift_attn_mask != 0,
                                                          float(
                                                              -100.0)).masked_fill(
                shift_attn_mask == 0, float(0.0))
            attn_mask = attn_mask + shift_attn_mask if attn_mask is not None else shift_attn_mask

        if self.cross_modulator is not None:
            shortcut = x
            x_cross = self.norm_cross(x)
            x_cross = self.cross_attn(x, self.cross_modulator.weight)
            x = shortcut + x_cross

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x,
                                   shifts=(-self.shift_size, -self.shift_size),
                                   dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x,
                                     self.win_size)  # nW*B, win_size, win_size, C  N*C->C
        x_windows = x_windows.view(-1, self.win_size * self.win_size,
                                   C)  # nW*B, win_size*win_size, C

        # with_modulator
        if self.modulator is not None:
            wmsa_in = self.with_pos_embed(x_windows, self.modulator.weight)
        else:
            wmsa_in = x_windows

        # W-MSA/SW-MSA
        attn_windows = self.attn(wmsa_in,
                                 mask=attn_mask)  # nW*B, win_size*win_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.win_size, self.win_size, C)
        shifted_x = window_reverse(attn_windows, self.win_size, H,
                                   W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size),
                           dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        del attn_mask
        return x

    def flops(self):
        flops = 0
        H, W = self.input_resolution

        if self.cross_modulator is not None:
            flops += self.dim * H * W
            flops += self.cross_attn.flops(H * W, self.win_size * self.win_size)

        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        flops += self.attn.flops(H, W)
        # norm2
        flops += self.dim * H * W
        # mlp
        flops += self.mlp.flops(H, W)
        # print("LeWin:{%.2f}"%(flops/1e9))
        return flops


class BasicUformerLayer(nn.Module):
    
    def __init__(
        self,
        dim,
        output_dim,
        input_resolution,
        depth,
        num_heads,
        win_size,
        mlp_ratio        = 4.0,
        qkv_bias         = True,
        qk_scale         = None,
        drop             = 0.,
        attn_drop        = 0.,
        drop_path        = 0.,
        norm_layer       = nn.LayerNorm,
        use_checkpoint   = False,
        token_projection = 'linear',
        token_mlp        = 'ffn',
        shift_flag       = True,
        modulator        = False,
        cross_modulator  = False,
    ):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        # build blocks
        if shift_flag:
            self.blocks = nn.ModuleList([
                LeWinTransformerBlock(dim=dim,
                                      input_resolution=input_resolution,
                                      num_heads=num_heads, win_size=win_size,
                                      shift_size=0 if (
                                                  i % 2 == 0) else win_size // 2,
                                      mlp_ratio=mlp_ratio,
                                      qkv_bias=qkv_bias, qk_scale=qk_scale,
                                      drop=drop, attn_drop=attn_drop,
                                      drop_path=drop_path[i] if isinstance(
                                          drop_path, list) else drop_path,
                                      norm_layer=norm_layer,
                                      token_projection=token_projection,
                                      token_mlp=token_mlp,
                                      modulator=modulator,
                                      cross_modulator=cross_modulator)
                for i in range(depth)])
        else:
            self.blocks = nn.ModuleList([
                LeWinTransformerBlock(dim=dim,
                                      input_resolution=input_resolution,
                                      num_heads=num_heads, win_size=win_size,
                                      shift_size=0,
                                      mlp_ratio=mlp_ratio,
                                      qkv_bias=qkv_bias, qk_scale=qk_scale,
                                      drop=drop, attn_drop=attn_drop,
                                      drop_path=drop_path[i] if isinstance(
                                          drop_path, list) else drop_path,
                                      norm_layer=norm_layer,
                                      token_projection=token_projection,
                                      token_mlp=token_mlp,
                                      modulator=modulator,
                                      cross_modulator=cross_modulator)
                for i in range(depth)])

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def forward(self, x, mask=None):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x, mask)
        return x

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        return flops


class Uformer(nn.Module):
    
    def __init__(self, img_size=256, in_chans=3, dd_in=3,
                 embed_dim=32, depths=[2, 2, 2, 2, 2, 2, 2, 2, 2],
                 num_heads=[1, 2, 4, 8, 16, 16, 8, 4, 2],
                 win_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, token_projection='linear',
                 token_mlp='leff',
                 dowsample=Downsample, upsample=Upsample, shift_flag=True,
                 modulator=False,
                 cross_modulator=False, **kwargs):
        super().__init__()

        self.num_enc_layers = len(depths) // 2
        self.num_dec_layers = len(depths) // 2
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio
        self.token_projection = token_projection
        self.mlp = token_mlp
        self.win_size = win_size
        self.reso = img_size
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.dd_in = dd_in

        # stochastic depth
        enc_dpr = [x.item() for x in torch.linspace(0, drop_path_rate,
                                                    sum(depths[
                                                        :self.num_enc_layers]))]
        conv_dpr = [drop_path_rate] * depths[4]
        dec_dpr = enc_dpr[::-1]

        # build layers

        # Input/Output
        self.input_proj = InputProj(in_channel=dd_in, out_channel=embed_dim,
                                    kernel_size=3, stride=1,
                                    act_layer=nn.LeakyReLU)
        self.output_proj = OutputProj(in_channel=2 * embed_dim,
                                      out_channel=in_chans, kernel_size=3,
                                      stride=1)

        # Encoder
        self.encoderlayer_0 = BasicUformerLayer(dim=embed_dim,
                                                output_dim=embed_dim,
                                                input_resolution=(img_size,
                                                                  img_size),
                                                depth=depths[0],
                                                num_heads=num_heads[0],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias,
                                                qk_scale=qk_scale,
                                                drop=drop_rate,
                                                attn_drop=attn_drop_rate,
                                                drop_path=enc_dpr[
                                                          sum(depths[:0]):sum(
                                                              depths[:1])],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint,
                                                token_projection=token_projection,
                                                token_mlp=token_mlp,
                                                shift_flag=shift_flag)
        self.dowsample_0 = dowsample(embed_dim, embed_dim * 2)
        self.encoderlayer_1 = BasicUformerLayer(dim=embed_dim * 2,
                                                output_dim=embed_dim * 2,
                                                input_resolution=(img_size // 2,
                                                                  img_size // 2),
                                                depth=depths[1],
                                                num_heads=num_heads[1],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias,
                                                qk_scale=qk_scale,
                                                drop=drop_rate,
                                                attn_drop=attn_drop_rate,
                                                drop_path=enc_dpr[
                                                          sum(depths[:1]):sum(
                                                              depths[:2])],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint,
                                                token_projection=token_projection,
                                                token_mlp=token_mlp,
                                                shift_flag=shift_flag)
        self.dowsample_1 = dowsample(embed_dim * 2, embed_dim * 4)
        self.encoderlayer_2 = BasicUformerLayer(dim=embed_dim * 4,
                                                output_dim=embed_dim * 4,
                                                input_resolution=(
                                                img_size // (2 ** 2),
                                                img_size // (2 ** 2)),
                                                depth=depths[2],
                                                num_heads=num_heads[2],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias,
                                                qk_scale=qk_scale,
                                                drop=drop_rate,
                                                attn_drop=attn_drop_rate,
                                                drop_path=enc_dpr[
                                                          sum(depths[:2]):sum(
                                                              depths[:3])],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint,
                                                token_projection=token_projection,
                                                token_mlp=token_mlp,
                                                shift_flag=shift_flag)
        self.dowsample_2 = dowsample(embed_dim * 4, embed_dim * 8)
        self.encoderlayer_3 = BasicUformerLayer(dim=embed_dim * 8,
                                                output_dim=embed_dim * 8,
                                                input_resolution=(
                                                img_size // (2 ** 3),
                                                img_size // (2 ** 3)),
                                                depth=depths[3],
                                                num_heads=num_heads[3],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias,
                                                qk_scale=qk_scale,
                                                drop=drop_rate,
                                                attn_drop=attn_drop_rate,
                                                drop_path=enc_dpr[
                                                          sum(depths[:3]):sum(
                                                              depths[:4])],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint,
                                                token_projection=token_projection,
                                                token_mlp=token_mlp,
                                                shift_flag=shift_flag)
        self.dowsample_3 = dowsample(embed_dim * 8, embed_dim * 16)

        # Bottleneck
        self.conv = BasicUformerLayer(dim=embed_dim * 16,
                                      output_dim=embed_dim * 16,
                                      input_resolution=(img_size // (2 ** 4),
                                                        img_size // (2 ** 4)),
                                      depth=depths[4],
                                      num_heads=num_heads[4],
                                      win_size=win_size,
                                      mlp_ratio=self.mlp_ratio,
                                      qkv_bias=qkv_bias, qk_scale=qk_scale,
                                      drop=drop_rate, attn_drop=attn_drop_rate,
                                      drop_path=conv_dpr,
                                      norm_layer=norm_layer,
                                      use_checkpoint=use_checkpoint,
                                      token_projection=token_projection,
                                      token_mlp=token_mlp,
                                      shift_flag=shift_flag)

        # Decoder
        self.upsample_0 = upsample(embed_dim * 16, embed_dim * 8)
        self.decoderlayer_0 = BasicUformerLayer(dim=embed_dim * 16,
                                                output_dim=embed_dim * 16,
                                                input_resolution=(
                                                img_size // (2 ** 3),
                                                img_size // (2 ** 3)),
                                                depth=depths[5],
                                                num_heads=num_heads[5],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias,
                                                qk_scale=qk_scale,
                                                drop=drop_rate,
                                                attn_drop=attn_drop_rate,
                                                drop_path=dec_dpr[:depths[5]],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint,
                                                token_projection=token_projection,
                                                token_mlp=token_mlp,
                                                shift_flag=shift_flag,
                                                modulator=modulator,
                                                cross_modulator=cross_modulator)
        self.upsample_1 = upsample(embed_dim * 16, embed_dim * 4)
        self.decoderlayer_1 = BasicUformerLayer(dim=embed_dim * 8,
                                                output_dim=embed_dim * 8,
                                                input_resolution=(
                                                img_size // (2 ** 2),
                                                img_size // (2 ** 2)),
                                                depth=depths[6],
                                                num_heads=num_heads[6],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias,
                                                qk_scale=qk_scale,
                                                drop=drop_rate,
                                                attn_drop=attn_drop_rate,
                                                drop_path=dec_dpr[
                                                          sum(depths[5:6]):sum(
                                                              depths[5:7])],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint,
                                                token_projection=token_projection,
                                                token_mlp=token_mlp,
                                                shift_flag=shift_flag,
                                                modulator=modulator,
                                                cross_modulator=cross_modulator)
        self.upsample_2 = upsample(embed_dim * 8, embed_dim * 2)
        self.decoderlayer_2 = BasicUformerLayer(dim=embed_dim * 4,
                                                output_dim=embed_dim * 4,
                                                input_resolution=(img_size // 2,
                                                                  img_size // 2),
                                                depth=depths[7],
                                                num_heads=num_heads[7],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias,
                                                qk_scale=qk_scale,
                                                drop=drop_rate,
                                                attn_drop=attn_drop_rate,
                                                drop_path=dec_dpr[
                                                          sum(depths[5:7]):sum(
                                                              depths[5:8])],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint,
                                                token_projection=token_projection,
                                                token_mlp=token_mlp,
                                                shift_flag=shift_flag,
                                                modulator=modulator,
                                                cross_modulator=cross_modulator)
        self.upsample_3 = upsample(embed_dim * 4, embed_dim)
        self.decoderlayer_3 = BasicUformerLayer(dim=embed_dim * 2,
                                                output_dim=embed_dim * 2,
                                                input_resolution=(img_size,
                                                                  img_size),
                                                depth=depths[8],
                                                num_heads=num_heads[8],
                                                win_size=win_size,
                                                mlp_ratio=self.mlp_ratio,
                                                qkv_bias=qkv_bias,
                                                qk_scale=qk_scale,
                                                drop=drop_rate,
                                                attn_drop=attn_drop_rate,
                                                drop_path=dec_dpr[
                                                          sum(depths[5:8]):sum(
                                                              depths[5:9])],
                                                norm_layer=norm_layer,
                                                use_checkpoint=use_checkpoint,
                                                token_projection=token_projection,
                                                token_mlp=token_mlp,
                                                shift_flag=shift_flag,
                                                modulator=modulator,
                                                cross_modulator=cross_modulator)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def extra_repr(self) -> str:
        return f"embed_dim={self.embed_dim}, token_projection={self.token_projection}, token_mlp={self.mlp},win_size={self.win_size}"

    def forward(self, x, mask=None):
        # Input Projection
        y = self.input_proj(x)
        y = self.pos_drop(y)
        # Encoder
        conv0 = self.encoderlayer_0(y, mask=mask)
        pool0 = self.dowsample_0(conv0)
        conv1 = self.encoderlayer_1(pool0, mask=mask)
        pool1 = self.dowsample_1(conv1)
        conv2 = self.encoderlayer_2(pool1, mask=mask)
        pool2 = self.dowsample_2(conv2)
        conv3 = self.encoderlayer_3(pool2, mask=mask)
        pool3 = self.dowsample_3(conv3)

        # Bottleneck
        conv4 = self.conv(pool3, mask=mask)

        # Decoder
        up0 = self.upsample_0(conv4)
        deconv0 = torch.cat([up0, conv3], -1)
        deconv0 = self.decoderlayer_0(deconv0, mask=mask)

        up1 = self.upsample_1(deconv0)
        deconv1 = torch.cat([up1, conv2], -1)
        deconv1 = self.decoderlayer_1(deconv1, mask=mask)

        up2 = self.upsample_2(deconv1)
        deconv2 = torch.cat([up2, conv1], -1)
        deconv2 = self.decoderlayer_2(deconv2, mask=mask)

        up3 = self.upsample_3(deconv2)
        deconv3 = torch.cat([up3, conv0], -1)
        deconv3 = self.decoderlayer_3(deconv3, mask=mask)

        # Output Projection
        y = self.output_proj(deconv3)
        return x + y if self.dd_in == 3 else y

    def flops(self):
        flops = 0
        # Input Projection
        flops += self.input_proj.flops(self.reso, self.reso)
        # Encoder
        flops += self.encoderlayer_0.flops() + self.dowsample_0.flops(self.reso,
                                                                      self.reso)
        flops += self.encoderlayer_1.flops() + self.dowsample_1.flops(
            self.reso // 2, self.reso // 2)
        flops += self.encoderlayer_2.flops() + self.dowsample_2.flops(
            self.reso // 2 ** 2, self.reso // 2 ** 2)
        flops += self.encoderlayer_3.flops() + self.dowsample_3.flops(
            self.reso // 2 ** 3, self.reso // 2 ** 3)

        # Bottleneck
        flops += self.conv.flops()

        # Decoder
        flops += self.upsample_0.flops(self.reso // 2 ** 4,
                                       self.reso // 2 ** 4) + self.decoderlayer_0.flops()
        flops += self.upsample_1.flops(self.reso // 2 ** 3,
                                       self.reso // 2 ** 3) + self.decoderlayer_1.flops()
        flops += self.upsample_2.flops(self.reso // 2 ** 2,
                                       self.reso // 2 ** 2) + self.decoderlayer_2.flops()
        flops += self.upsample_3.flops(self.reso // 2,
                                       self.reso // 2) + self.decoderlayer_3.flops()

        # Output Projection
        flops += self.output_proj.flops(self.reso, self.reso)
        return flops
    
# endregion


# region Model

@MODELS.register(name="uformer")
class Uformer(base.UniversalImageEnhancementModel):
    """A General U-Shaped Transformer (Uformer) Network.
    
    Reference:
        `<https://github.com/ZhendongWang6/Uformer>`__

    See Also: :class:`mon.vision.enhance.universal.base.UniversalImageEnhancementModel`
    """
    
    zoo = {}

    def __init__(
        self,
        image_size      : int | list[int] = 256,
        channels        : int             = 3,
        dd_in           : int             = 3,
        embed_dim       : int             = 32,
        depths          : list[int]       = [2, 2, 2, 2, 2 , 2 , 2, 2, 2],
        num_heads       : list[int]       = [1, 2, 4, 8, 16, 16, 8, 4, 2],
        win_size        : int             = 8,
        mlp_ratio       : float           = 4.,
        qkv_bias        : bool            = True,
        qk_scale        : Any             = None,
        drop_rate       : float           = 0.0,
        attn_drop_rate  : float           = 0.0,
        drop_path_rate  : float           = 0.1,
        norm_layer      : Callable[..., nn.Module] = nn.LayerNorm,
        patch_norm      : bool            = True,
        use_checkpoint  : bool            = False,
        token_projection: str             = "linear",
        token_mlp       : str             = "leff",
        dowsample       : Callable[..., nn.Module] = Downsample,
        upsample        : Callable[..., nn.Module] = Upsample,
        shift_flag      : bool            = True,
        modulator       : bool            = False,
        cross_modulator : bool            = False,
        weights         : Any             = None,
        name            : str             = "uformer",
        *args, **kwargs
    ):
        super().__init__(
            weights = weights,
            name    = name,
            *args, **kwargs
        )
        
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)
    
    def init_weights(self, m: nn.Module):
        gain      = torch.nn.init.calculate_gain('leaky_relu', 0.20)
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            if hasattr(m, "conv"):
                nn.init.orthogonal_(m.weight, gain=gain)
                if not m.bias is None:
                    nn.init.constant_(m.bias, 0)

    def forward_loss(
        self,
        input : torch.Tensor,
        target: torch.Tensor | None,
        *args, **kwargs
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
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
        if self.loss:
            loss = 0
            for p in pred:
                loss += self.loss(p, target)
        else:
            loss = None
        return pred[-1], loss

    def forward_once(
        self,
        input    : torch.Tensor,
        profile  : bool = False,
        out_index: int  = -1,
        *args, **kwargs
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
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

        # Stage 1
        x1   = self.conv_01(x)
        encs = []
        decs = []
        for i, down in enumerate(self.down_path_1):
            if (i + 1) < self.depth:
                x1, x1_up = down(x1)
                encs.append(x1_up)
            else:
                x1 = down(x1)
        for i, up in enumerate(self.up_path_1):
            x1 = up(x1, self.skip_conv_1[i](encs[-i - 1]))
            decs.append(x1)

        # SAM
        sam_feats, y1 = self.sam12(input=[x1, x])

        # Stage 2
        x2     = self.conv_02(x)
        x2     = self.cat12(torch.cat([x2, sam_feats], dim=1))
        blocks = []
        for i, down in enumerate(self.down_path_2):
            if (i + 1) < self.depth:
                x2, x2_up = down(x2, encs[i], decs[-i - 1])
                blocks.append(x2_up)
            else:
                x2 = down(x2)
        for i, up in enumerate(self.up_path_2):
            x2 = up(x2, self.skip_conv_2[i](blocks[-i - 1]))

        y2 = self.last(x2)
        y2 = y2 + x
        return [y1, y2]

# endregion

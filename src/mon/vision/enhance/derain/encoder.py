import math
from functools import partial

import torch

from mon import nn
from mon.nn.modules import (activation, conv, dropout, linear, normalization)
from .drop import DropPath


class Multi_Head_Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.,
        proj_drop=0.,
        sr_ratio=1
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = linear.Linear(dim, dim, bias=qkv_bias)
        self.kv = linear.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = dropout.Dropout(attn_drop)
        self.proj = linear.Linear(dim, dim)
        self.proj_drop = dropout.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = conv.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = normalization.LayerNorm(dim)

    def forward(
        self,
        x,
        H,
        W
    ):

        B, N, C = x.shape
        #Number of Batches  (B)
        #images' feature (N) = H*W
        #Channels     (C)

        #reshape features for n_head
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            #x_: change from (B, N, C) -> (B, C, N) -> (B, C, H, W)
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B,-1,2, self.num_heads, C//self.num_heads).permute(2,0,3,1,4)
        else:
            kv = self.kv(x).reshape(B,-1,2, self.num_heads, C//self.num_heads).permute(2,0,3,1,4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        # kv = (k.transpose(-2, -1) @ v)
        # attn = (q @ kv) * self.scale
        # x = attn.transpose(1, 2).reshape(B, N, C)
        # x = self.proj(x)
        # x = self.proj_drop(x)
        return x
    
class MFFN(nn.Module):
    def __init__(self, dim, FFN_expand=2,norm_layer='WithBias'):
        super(MFFN, self).__init__()
        self.fc = linear.Linear(
          in_features=dim,
          out_features=dim
        )
        self.conv1 = nn.Conv2d(dim,dim*FFN_expand,1)
        self.conv33 = nn.Conv2d(dim*FFN_expand,dim*FFN_expand,3,1,1,groups=dim*FFN_expand)
        self.conv55 = nn.Conv2d(dim*FFN_expand,dim*FFN_expand,5,1,2,groups=dim*FFN_expand)
        self.sg = activation.SimpleGate()
        self.conv4 = nn.Conv2d(dim,dim,1)

        self.apply(self._init_weights)
    def _init_weights(self,m):
        if isinstance(m,nn.Linear):
            torch.nn.init.trunc_normal_(m.weight,std=0.02)
            # m.weight.data.trunc_(std=0.02)
            if isinstance(m,nn.Linear) and m.biasImageDataset:
                # nn.init.constant_(m.bias,0)
                m.bias.data.__contains__(0)
        elif isinstance(m,nn.LayerNorm):
            m.bias.data.__contains__(0)
            m.weight.data.__contains__(1.0)
            # nn.init.constant_(m.bias,0)
            # nn.init.constant_(m.weight,1.0)
        elif isinstance(m,nn.Conv2d):
            fan_out = m.kernel_size[0]*m.kernel_size[1]*m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0,math.sqrt(2.0/fan_out))
            if m.biasImageDataset:
                m.bias.data.zero_()
    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        x1 = self.conv1(x)
        x33 = self.conv33(x1)
        x55 = self.conv55(x1)
        x = x1+x33+x55
        x = self.sg(x)
        x = self.conv4(x)
        x = x.permute(0, 2, 3, 1).reshape(B, H*W, C)
        return x

class Transformer_Block(nn.Module):

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.,
        qkv_bias=False,
        drop=0.,
        attn_drop=0.,
        drop_path=0.,
        act_layer=nn.GELU,
        norm_layer=partial(normalization.LayerNorm),
        sr_ratio=1
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Multi_Head_Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        # self.ffn = FFN(
        #  in_dim=dim,
        #  out_dim=dim,
        #  hidden_conv_dims=mlp_hidden_dim,
        #  hidden_dims=mlp_hidden_dim,
        #  activation=act_layer,
        #  dropout=drop,
        # )
        self.ffn = MFFN(dim=dim)

    def forward(self, x, H, W):

        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.ffn(self.norm2(x), H, W))
        return x

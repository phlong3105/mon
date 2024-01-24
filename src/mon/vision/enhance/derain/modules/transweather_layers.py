from __future__ import annotations

__all__ = [
    "Transformer_Block",
    "Block_dec",
    "CBAM",
]

from functools import partial

import torch

from mon.nn import functional
from mon import core, nn

math = core.math


class Bottleneck(nn.Module):
    """
    Bottleneck module for channel attention

    Args:
        channels (int): number of input channels
        reduction_ratio (int): reduction ratio for the middle layer
        depth (int): number of layers in the middle layer

    Returns:
        out (torch.Tensor): output tensor
    """

    def __init__(self, channels, reduction_ratio, depth):
        super(Bottleneck, self).__init__()
        self.channels = channels
        self.reduction_ratio = reduction_ratio
        self.depth = depth
        self.middle_layer_size = int(self.channels / float(self.reduction_ratio))

        self.l1 = nn.Linear(self.channels, self.middle_layer_size)
        self.act = nn.ReLU()
        self.l2 = nn.Linear(self.middle_layer_size, self.channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        res = self.l1(x)
        res = self.act(res)
        res = self.l2(res) + x
        res = self.sigmoid(res)
        return res


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction_ratio):
        super(ChannelAttention, self).__init__()
        self.channels = channels
        self.reduction_ratio = reduction_ratio
        self.middle_layer_size = int(self.channels / float(self.reduction_ratio))

        self.bottleneck = Bottleneck(self.channels, self.reduction_ratio, 3)

    def forward(self, x):
        kernel = (x.size()[2], x.size()[3])
        avg_pool = functional.avg_pool2d(x, kernel)
        avg_pool = avg_pool.view(avg_pool.size()[0], -1)
        avg_pool_bck = self.bottleneck(avg_pool)

        avg_pool_bck = avg_pool_bck.unsqueeze(2).unsqueeze(3)

        out = avg_pool_bck.repeat(1, 1, kernel[0], kernel[1])
        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size):
        super(SpatialAttention, self).__init__()
        self.kernel_size = kernel_size

        assert kernel_size % 2 == 1, 'Odd kernel size required'
        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=kernel_size,
                              padding=int((kernel_size - 1) / 2))
        # batch-norm

    def forward(self, x):

        pool = self.agg_channel(x, 'avg')

        conv = self.conv(pool)
        # batch-norm ????????????????????????????????????????????
        conv = conv.repeat(1, x.size()[1], 1, 1)
        return conv

    @staticmethod
    def agg_channel(x, pool='max'):
        b, c, h, w = x.size()
        x = x.view(b, c, h * w)
        x = x.permute(0, 2, 1)
        if pool == 'max':
            x = functional.max_pool1d(x, c)
        elif pool == 'avg':
            x = functional.avg_pool1d(x, c)
        elif pool == 'min':
            x = -functional.max_pool1d(-x, c)
        x = x.permute(0, 2, 1)
        x = x.view(b, 1, h, w)
        return x


class CBAM(nn.Module):

    def __init__(self, channels, reduction_ratio: int = 2, spatial: bool | None = True):
        super(CBAM, self).__init__()
        self.channels = channels
        self.reduction_ratio = reduction_ratio
        self.kernel_size = 7

        self.channel_attention = ChannelAttention(channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(self.kernel_size) if spatial is True else None

    def forward(self, f, height, width):
        B, N, channels = f.shape
        f = f.permute(0, 2, 1).reshape(B, channels, height, width)
        chan_att = self.channel_attention(f)
        fp = chan_att * f
        if self.spatial_attention is not None:
            fp = self.spatial_attention(fp) * fp
        fpp = fp
        fpp = fpp.reshape(B, channels, N).permute(0, 2, 1)
        return fpp


class ChannelPriorQueries(nn.Module):
    def __init__(
            self,
            dim: int,
            name: str = 'channel_prior',
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=2,
            stride=2,
            padding=0
        )
        self.conv2 = nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=2,
            stride=2,
            padding=0
        )
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        query = self.conv1(x)
        query = self.upsample(query)
        query = self.conv2(query)
        query = self.upsample(query)

        return query


class MultiHeadAttention(nn.Module):
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

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(
            self,
            x,
            H,
            W
    ):

        B, N, C = x.shape
        # Number of Batches  (B)
        # images' feature (N) = H*W
        # Channels     (C)

        # reshape features for n_head
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            # x_: change from (B, N, C) -> (B, C, N) -> (B, C, H, W)
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
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


class Attention_dec(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_scale=None,
            attn_drop=0.,
            proj_drop=0.,
            sr_ratio=1
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.task_query = ChannelPriorQueries(dim=dim)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(
                in_channels=dim,
                out_channels=dim,
                kernel_size=sr_ratio,
                stride=sr_ratio
            )
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):

        B, N, C = x.shape
        x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
        task_q = self.task_query(x_).reshape(B, C, -1).permute(0, 2, 1)
        q = self.q(task_q).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        q = torch.nn.functional.interpolate(q, size=(v.shape[2], v.shape[3]))

        #  Dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        #  Efficient attention
        # kv = (k.transpose(-2, -1) @ v)
        # attn = (q @ kv) * self.scale
        # x = attn.transpose(1, 2).reshape(B, N, C)
        # x = self.proj(x)
        # x = self.proj_drop(x)
        return x


class MFFN(nn.Module):
    def __init__(self, dim, FFN_expand=2, norm_layer='WithBias'):
        super(MFFN, self).__init__()
        self.fc = nn.Linear(
            in_features=dim,
            out_features=dim
        )
        self.conv1 = nn.Conv2d(dim, dim * FFN_expand, 1)
        self.conv33 = nn.Conv2d(dim * FFN_expand, dim * FFN_expand, 3, 1, 1, groups=dim * FFN_expand)
        self.conv55 = nn.Conv2d(dim * FFN_expand, dim * FFN_expand, 5, 1, 2, groups=dim * FFN_expand)
        self.sg = nn.SimpleGate()
        self.conv4 = nn.Conv2d(dim, dim, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            # m.weight.data.trunc_(std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                # nn.init.constant_(m.bias,0)
                m.bias.data.__contains__(0)
        elif isinstance(m, nn.LayerNorm):
            m.bias.data.__contains__(0)
            m.weight.data.__contains__(1.0)
            # nn.init.constant_(m.bias,0)
            # nn.init.constant_(m.weight,1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        x1 = self.conv1(x)
        x33 = self.conv33(x1)
        x55 = self.conv55(x1)
        x = x1 + x33 + x55
        x = self.sg(x)
        x = self.conv4(x)
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
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
            norm_layer=partial(nn.LayerNorm),
            sr_ratio=1
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = MultiHeadAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio
        )

        self.drop_path = nn.DropPath(drop_path) if drop_path > 0. else nn.Identity()
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


class Block_dec(nn.Module):

    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.,
            qkv_bias=False,
            qk_scale=None,
            drop=0.,
            attn_drop=0.,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=partial(nn.LayerNorm),
            sr_ratio=1
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_dec(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio
        )

        self.drop_path = nn.DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        # self.mlp = FFN(
        #  in_dim=dim,
        #  out_dim=dim,
        #  hidden_conv_dims=mlp_hidden_dim,
        #  hidden_dims=mlp_hidden_dim,
        #  activation=act_layer,
        #  dropout=drop
        # )
        self.ffn = MFFN(
            dim=dim,
        )

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.ffn(self.norm2(x), H, W))
        return x

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements Transweather models.

./run.sh transweather none none train 100 sice-zerodce all vision/enhance/derain no last
"""

from __future__ import annotations

__all__ = [
    "Transweather",
]

import math
from functools import partial
from typing import Any

import kornia
from torchvision.models import vgg16, VGG16_Weights

from mon import core, nn
from mon.core import _callable
from mon.globals import MODELS
from mon.nn import ContradictChannelLoss, PerceptualL1Loss
from mon.vision.enhance.derain import base
from .decoder import Block_dec
# from mon.vision.feature import OPEmbedder
from .encoder import Transformer_Block

console = core.console


# region Loss

class Loss(nn.Loss):
    
    def __init__(
        self,
        weight_smooth   : float = 1,
        weight_perL1    : float = 0.1,
        weight_ccp      : float = 2,
        reduction       : str   = "mean",
        verbose         : bool  = False,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.verbose        = verbose
        self.weight_smooth  = weight_smooth
        self.weight_perL1   = weight_perL1
        self.weight_ccp     = weight_ccp
        self.loss_smooth    = nn.SmoothL1Loss(reduction=reduction)
        self.loss_perL1     = PerceptualL1Loss(vgg16(weights=VGG16_Weights.DEFAULT).features[:16], reduction=reduction)
        self.loss_ccp       = ContradictChannelLoss(reduction=reduction)
    
    def __str__(self) -> str:
        return f"loss"
    
    def forward(
        self,
        pred    : torch.Tensor | list[torch.Tensor],
        target  : torch.Tensor | list[torch.Tensor],
        **_
    ) -> tuple[torch.Tensor, torch.Tensor]:
        loss_smooth = self.loss_smooth(input=pred, target=target)           if self.weight_smooth  > 0 else 0
        loss_perL1  = self.loss_perL1(input=pred, target=target)      if self.weight_perL1 > 0 else 0
        loss_ccp    = self.loss_ccp(input=pred, target=target)        if self.weight_ccp > 0 else 0
        
        loss = (
              self.weight_smooth    * loss_smooth
            + self.weight_perL1     * loss_perL1
            + self.weight_ccp       * loss_ccp
        )
        
        if self.verbose:
            console.log(f"{self.loss_smooth.__str__():<30} : {loss_smooth}")
            console.log(f"{self.loss_perL1.__str__():<30} : {loss_perL1}")
            console.log(f"{self.loss_ccp.__str__():<30}: {loss_ccp}")
        return loss, pred
        
# endregion


# region embedder

class OPEmbedder(nn.ConvLayerParsingMixin, nn.Module):
    """HOG (Histogram of Oriented Gradients) feature embedder.
    
    Args:
        win_size: The window size should be chosen based on the size of the
            objects being tracked. A smaller window size is suitable for
            tracking small objects, while a larger window size is suitable for
            larger objects. Default: ``(64, 128)``.
        block_size: The block size should be chosen based on the level of detail
            required for tracking. A larger block size can capture more global
            features, while a smaller block size can capture more local
            features. Default: ``(16, 16)``.
        block_stride: The block stride should be chosen based on the speed of
            the objects being tracked. A smaller block stride can provide more
            accurate tracking, but may also require more computation. Defaults
            to ``(8, 8)``.
        cell_size: The cell size should be chosen based on the texture and
            structure of the objects being tracked. A smaller cell size can
            capture more detailed features, while a larger cell size can capture
            more general features. Default: ``(8, 8)``.
        nbins: The number of orientation bins should be chosen based on the
            complexity of the gradient orientations in the images. More
            orientation bins can provide more detailed information about the
            orientations, but may also increase the dimensionality of the
            feature vector and require more computation. Default: ``9``.
        
    See Also:
        - :class:`mon.vision.model.embedding.base.Embedder`.
        - :class:`cv2.HOGDescriptor`.
    """
    
    def __init__(
        self,
        img_size:   int=224,
        patch_size: int=7,
        stride:  int=4,
        in_channels:int=3,
        nbins       : int       = 9,embed_dim:  int=768,
        *args, **kwargs
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size//patch_size) * (img_size//patch_size)
        self.proj = nn.Conv2d(
            in_channels     = in_channels,
            out_channels    = embed_dim,
            kernel_size     = patch_size,
            stride          = stride,
            padding         = patch_size // 2
        )
        self.normalization = nn.LayerNorm(embed_dim)
        self.apply(self.init_weights)
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            m.weight.data.trunc_()
            # torch.nn.init.trunc_normal_(m.weight, )
        if isinstance(m, nn.Linear) and m.bias is not None:
            m.bias.data.zero_()
            # torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            m.bias.data.zero_()
            m.weight.data.fill_(1.0)
            # torch.nn.init.constant_(m.bias, 0)
            # torch.nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        
    def forward(self, images: torch.Tensor, normalization: bool = False) -> torch.Tensor:
        """Extract features in the images.

        Args:
            indexes: A :class:`list` of image indexes.
            images: Images of shape :math:`[N, C, H, W]`.
            normalization: Whether to normalize the features.

        Returns:
           A 2D :class:`list` of feature vectors.
        """
        # print(images.get_device())
        # self.proj.to(images.get_device())
        images = self.proj(images)
        N, C, H, W = images.shape
        images = images.flatten(2).transpose(1, 2) # N, HW, C
        if normalization:
            images = self.normalization(images)
        return images, H, W
# endregion

# region Module

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['ChannelAttention']

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
        self.middle_layer_size = int(self.channels/ float(self.reduction_ratio))

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
        self.middle_layer_size = int(self.channels/ float(self.reduction_ratio))

        self.bottleneck = Bottleneck(self.channels, self.reduction_ratio, 3)


    def forward(self, x):
        kernel = (x.size()[2], x.size()[3])
        avg_pool = F.avg_pool2d(x, kernel )
        avg_pool = avg_pool.view(avg_pool.size()[0], -1)
        avg_pool_bck = self.bottleneck(avg_pool)

        avg_pool_bck = avg_pool_bck.unsqueeze(2).unsqueeze(3)

        out = avg_pool_bck.repeat(1,1,kernel[0], kernel[1])
        return out

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size):
        super(SpatialAttention, self).__init__()
        self.kernel_size = kernel_size

        assert kernel_size % 2 == 1, 'Odd kernel size required'
        self.conv = nn.Conv2d(in_channels = 1, out_channels = 1, kernel_size = kernel_size, padding= int((kernel_size-1)/2))
        # batchnorm


    def forward(self, x):

        pool = self.agg_channel(x, 'avg')

        conv = self.conv(pool)
        # batchnorm ????????????????????????????????????????????
        conv = conv.repeat(1,x.size()[1],1,1)
        return conv

    def agg_channel(self, x, pool = 'max'):
        b,c,h,w = x.size()
        x = x.view(b, c, h*w)
        x = x.permute(0,2,1)
        if pool == 'max':
            x = F.max_pool1d(x,c)
        elif pool == 'avg':
            x = F.avg_pool1d(x,c)
        elif pool == 'min':
            x = -F.max_pool1d(-x,c)
        x = x.permute(0,2,1)
        x = x.view(b,1,h,w)
        return x

class CBAM(nn.Module):

    def __init__(self, channels, reduction_ratio: int =2, spatial: bool | None = True):
        super(CBAM, self).__init__()
        self.channels = channels
        self.reduction_ratio = reduction_ratio
        self.kernel_size = 7

        self.channel_attention = ChannelAttention(channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(self.kernel_size) if spatial is True else None

    def forward(self, f, H, W):
        B, N, C = f.shape
        f = f.permute(0, 2, 1).reshape(B, C, H, W)
        chan_att = self.channel_attention(f)
        fp = chan_att * f
        if self.spatial_attention is not None:
            fp = self.spatial_attention(fp) * fp
        fpp = fp 
        fpp = fpp.reshape(B, C, N).permute(0, 2, 1)
        return fpp    

# endregion

# region Model
@MODELS.register(name="transweather")
class Transweather(base.DerainingModel):
    """Transweather model.
    
    See Also: :class:`mon.vision.enhance.derain.base.DerainingModel`
    """
    class Encoder(nn.Module):
        def __init__(
                self,
                img_size:       int=224,
                patch_size:     int=[7, 3, 3, 3],
                in_chans:       int=3,
                num_classes:    int=1000,
                embed_dims:     list=[64, 128, 320, 512],
                num_heads:      list=[1, 2, 4, 4],
                mlp_ratios:     list=[2, 2, 2, 2],
                qkv_bias:       bool=True,
                drop_rate:      float=0.,
                attn_drop_rate: float=0.,
                drop_path_rate: float=0.1,
                norm_layer      = partial(nn.LayerNorm, eps=1e-6),
                depths:         list=[2, 2, 2, 2],
                sr_ratios:      list=[4, 2, 2, 1]
            ):
            super().__init__()
            self.num_classes = num_classes
            self.depths = depths
            self.embed_dims = embed_dims

            # region Embedding
            # region patch embedding definitions
            self.patch_embed1 = OPEmbedder(
                img_size=img_size,
                patch_size=patch_size[0], 
                stride=int((patch_size[0] + 1) / 2), 
                in_channels=in_chans,
                embed_dim=embed_dims[0]
            )
            self.patch_embed2 = OPEmbedder(
                img_size=img_size // 4,
                patch_size=patch_size[1],
                stride=int((patch_size[1] + 1) / 2),
                in_channels=embed_dims[0],
                embed_dim=embed_dims[1]
            )
            self.patch_embed3 = OPEmbedder(
                img_size=img_size // 8,
                patch_size=patch_size[2],
                stride=int((patch_size[2] + 1) / 2),
                in_channels=embed_dims[1],
                embed_dim=embed_dims[2]
            )
            self.patch_embed4 = OPEmbedder(
                img_size=img_size // 16,
                patch_size=patch_size[3],
                stride=int((patch_size[3] + 1) / 2),
                in_channels=embed_dims[2],
                embed_dim=embed_dims[3]
            )
            # endregion
            # region Intra-patch embedding definitions
            self.mini_patch_embed1 = OPEmbedder(
                img_size=img_size // 4,
                patch_size=patch_size[1],
                stride=int((patch_size[1] + 1) / 2),
                in_channels=embed_dims[0],
                embed_dim=embed_dims[1]
            )
            self.mini_patch_embed2 = OPEmbedder(
                img_size=img_size // 8,
                patch_size=patch_size[2],
                stride=int((patch_size[2] + 1) / 2),
                in_channels=embed_dims[1],
                embed_dim=embed_dims[2]
            )
            self.mini_patch_embed3 = OPEmbedder(
                img_size=img_size // 16,
                patch_size=patch_size[3],
                stride=int((patch_size[3] + 1) / 2),
                in_channels=embed_dims[2],
                embed_dim=embed_dims[3]
            )
            # endregion
            # endregion
            # region Enoder
            # region Main encoder
            # region block 1
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
            cur = 0

            self.block1 = nn.ModuleList([Transformer_Block(
                dim=embed_dims[0],
                num_heads=num_heads[0],
                mlp_ratio=mlp_ratios[0],
                qkv_bias=qkv_bias,

                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[cur + i],
                norm_layer=norm_layer,
                sr_ratio=sr_ratios[0])
                for i in range(depths[0])]
            )
            self.norm1 = norm_layer(embed_dims[0])
            # endregion
            # region block 2
            cur += depths[0]
            self.block2 = nn.ModuleList([Transformer_Block(
                dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
                sr_ratio=sr_ratios[1])
                for i in range(depths[1])])
            self.norm2 = norm_layer(embed_dims[1])
            # endregion
            # region block 3
            cur += depths[1]
            self.block3 = nn.ModuleList([Transformer_Block(
                dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
                sr_ratio=sr_ratios[2])
                for i in range(depths[2])])
            self.norm3 = norm_layer(embed_dims[2])
            # endregion
            # region block 4
            cur += depths[2]
            self.block4 = nn.ModuleList([Transformer_Block(
                dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
                sr_ratio=sr_ratios[3])
                for i in range(depths[3])])
            self.norm4 = norm_layer(embed_dims[3])
            # endregion
            # endregion
            # region Intra-patch encoder
            # region block 1
            self.patch_block1 = nn.ModuleList([CBAM(
            channels=embed_dims[1], reduction_ratio=2, spatial=True)
            for _ in range(1)])
            self.pnorm1 = norm_layer(embed_dims[1])
            # endregion
            # region block 2
            self.patch_block2 = nn.ModuleList([CBAM(
            channels=embed_dims[2], reduction_ratio=2, spatial=True)
            for _ in range(1)])
            self.pnorm2 = norm_layer(embed_dims[2])
            # endregion
            # region block 3
            self.patch_block3 = nn.ModuleList([CBAM(
            channels=embed_dims[3], reduction_ratio=2, spatial=True)
            for _ in range(1)])
            self.pnorm3 = norm_layer(embed_dims[3])
            # endregion
            # endregion
            # endregion

            self.apply(self.init_weights)

        def init_weights(self, m):
            if isinstance(m, nn.Linear):
                m.weight.data.trunc_()
                # torch.nn.init.trunc_normal_(m.weight, )
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()
                # torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                m.bias.data.zero_()
                m.weight.data.fill_(1.0)
                # torch.nn.init.constant_(m.bias, 0)
                # torch.nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    m.bias.data.zero_()
        
        def reset_drop_path(self, drop_path_rate):
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
            cur = 0
            for i in range(self.depths[0]):
                self.block1[i].drop_path.drop_prob = dpr[cur + i] # type: ignore

            cur += self.depths[0]
            for i in range(self.depths[1]):
                self.block2[i].drop_path.drop_prob = dpr[cur + i] # type: ignore

            cur += self.depths[1]
            for i in range(self.depths[2]):
                self.block3[i].drop_path.drop_prob = dpr[cur + i] # type: ignore

            cur += self.depths[2]
            for i in range(self.depths[3]):
                self.block4[i].drop_path.drop_prob = dpr[cur + i] # type: ignore

        def forward_features(self, x):
            B = x.shape[0]
            outs = []
            # stage 1
            x1, H1, W1 = self.patch_embed1(
                images=x,
                normalization=True
            )
            x2, H2, W2 = self.mini_patch_embed1(
                x1.permute(0,2,1).reshape(B,self.embed_dims[0],H1,W1),
                normalization=True
            )

            for i, blk in enumerate(self.block1):
                x1 = blk(x1, H1, W1)
            x1 = self.norm1(x1)
            x1 = x1.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()

            for i, blk in enumerate(self.patch_block1):
                x2 = blk(x2, H2, W2)
            x2 = self.pnorm1(x2)
            x2 = x2.reshape(B, H2, W2, -1).permute(0, 3, 1, 2).contiguous()

            outs.append(x1)

            # stage 2
            x1, H1, W1 = self.patch_embed2(
                images=x1,
                normalization=True
            )
            x1 = x1.permute(0,2,1).reshape(B,self.embed_dims[1],H1,W1)+x2
            x2, H2, W2 = self.mini_patch_embed2(
                images=x1,
                normalization=True
            )

            x1 = x1.view(x1.shape[0],x1.shape[1],-1).permute(0,2,1)

            for i, blk in enumerate(self.block2):
                x1 = blk(x1, H1, W1)
            x1 = self.norm2(x1)
            x1 = x1.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x1)

            for i, blk in enumerate(self.patch_block2):
                x2 = blk(x2, H2, W2)
            x2 = self.pnorm2(x2)
            x2 = x2.reshape(B, H2, W2, -1).permute(0, 3, 1, 2).contiguous()


            # stage 3
            x1, H1, W1 = self.patch_embed3(
                images=x1,
                normalization=True
            )
            x1 = x1.permute(0,2,1).reshape(B,self.embed_dims[2],H1,W1)+x2
            x2, H2, W2 = self.mini_patch_embed3(
                images=x1,
                normalization=True
            )

            x1 = x1.view(x1.shape[0],x1.shape[1],-1).permute(0,2,1)

            for i, blk in enumerate(self.block3):
                x1 = blk(x1, H1, W1)
            x1 = self.norm3(x1)
            x1 = x1.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x1)

            for i, blk in enumerate(self.patch_block3):
                x2 = blk(x2, H2, W2)
            x2 = self.pnorm3(x2)
            x2 = x2.reshape(B, H2, W2, -1).permute(0, 3, 1, 2).contiguous()

            # stage 4
            x1, H1, W1 = self.patch_embed4(
                images=x1,
                normalization=True
            )
            x1 = x1.permute(0,2,1).reshape(B,self.embed_dims[3],H1,W1)+x2

            x1 = x1.view(x1.shape[0],x1.shape[1],-1).permute(0,2,1)

            for i, blk in enumerate(self.block4):
                x1 = blk(x1, H1, W1)
            x1 = self.norm4(x1)
            x1 = x1.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x1)

            return outs
        
        def forward(self, x):
            x = self.forward_features(x)
            return x

    class Decoder(nn.Module):
        def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_chans=3,
            num_classes=1000,
            embed_dims=[64, 128, 320, 512],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.1,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[3, 4, 6, 3],
            sr_ratios=[8, 4, 2, 1]
        ):
            super().__init__()
            self.num_classes = num_classes
            self.depths = depths
            self.embed_dims = embed_dims

            # patch_embed
            self.patch_embed1 = OPEmbedder(img_size=img_size//16, patch_size=3, stride=2, in_channels=embed_dims[3], embed_dim=embed_dims[3])

            # transformer decoder
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
            cur = 0
            self.block1 = nn.ModuleList([Block_dec(
                dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
                sr_ratio=sr_ratios[3])
                for i in range(depths[0])])
            self.norm1 = norm_layer(embed_dims[3])
            cur += depths[0]

            self.apply(self.init_weights)

        def init_weights(self, m):
            if isinstance(m, nn.Linear):
                m.weight.data.trunc_()
                # torch.nn.init.trunc_normal_(m.weight, )
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()
                # torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                m.bias.data.zero_()
                m.weight.data.fill_(1.0)
                # torch.nn.init.constant_(m.bias, 0)
                # torch.nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    m.bias.data.zero_()

        def forward_features(self, x):
            #x=x[3]
            B = x.shape[0]
            outs = []

            # stage 1
            x, H, W = self.patch_embed1(
                images=x,
                normalization=True
            )
            for i, blk in enumerate(self.block1):
                x = blk(x, H, W)
            x = self.norm1(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)

            return outs

        def forward(self, x):
            x = self.forward_features(x)
            # x = self.head(x)

            return x
    
    class convprojection(nn.Module):
        class ResidualBlock(torch.nn.Module):
            def __init__(self, channels):
                super().__init__()
                #Residual Block
                self.residual_block = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3,
                        stride=1, padding='same'),
                nn.ReLU(),
                nn.Conv2d(channels, channels, kernel_size=3,
                        stride=1, padding='same'))
                self.relu = nn.ReLU()

            def forward(self, x):
                out = self.residual_block(x)
                out = out + x
                return self.relu(out)
        
        def __init__(self, path=None, **kwargs):
            super().__init__()

            self.convd32x = nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1)
            self.convd16x = nn.ConvTranspose2d(512, 320, kernel_size=4, stride=2, padding=1)
            self.dense_4 = nn.Sequential(self.ResidualBlock(320))
            self.convd8x = nn.ConvTranspose2d(320, 128, kernel_size=4, stride=2, padding=1)
            self.dense_3 = nn.Sequential(self.ResidualBlock(128))
            self.convd4x = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
            self.dense_2 = nn.Sequential(self.ResidualBlock(64))
            self.convd2x = nn.ConvTranspose2d(64, 16, kernel_size=4, stride=2, padding=1)
            self.dense_1 = nn.Sequential(self.ResidualBlock(16))
            self.convd1x = nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1)
            self.conv_output = nn.Conv2d(8, 3, kernel_size=3, stride=1, padding=1)

            self.active = nn.Tanh()  

        def forward(self,x1,x2):

            res32x = self.convd32x(x2[0])

            if x1[3].shape[3] != res32x.shape[3] and x1[3].shape[2] != res32x.shape[2]:
                p2d = (0,-1,0,-1)
                res32x = F.pad(res32x,p2d,"constant",0)

            elif x1[3].shape[3] != res32x.shape[3] and x1[3].shape[2] == res32x.shape[2]:
                p2d = (0,-1,0,0)
                res32x = F.pad(res32x,p2d,"constant",0)
            elif x1[3].shape[3] == res32x.shape[3] and x1[3].shape[2] != res32x.shape[2]:
                p2d = (0,0,0,-1)
                res32x = F.pad(res32x,p2d,"constant",0)

            res16x = res32x + x1[3]
            res16x = self.convd16x(res16x) 

            if x1[2].shape[3] != res16x.shape[3] and x1[2].shape[2] != res16x.shape[2]:
                p2d = (0,-1,0,-1)
                res16x = F.pad(res16x,p2d,"constant",0)
            elif x1[2].shape[3] != res16x.shape[3] and x1[2].shape[2] == res16x.shape[2]:
                p2d = (0,-1,0,0)
                res16x = F.pad(res16x,p2d,"constant",0)
            elif x1[2].shape[3] == res16x.shape[3] and x1[2].shape[2] != res16x.shape[2]:
                p2d = (0,0,0,-1)
                res16x = F.pad(res16x,p2d,"constant",0)

            res8x = self.dense_4(res16x) + x1[2]
            res8x = self.convd8x(res8x) 
            res4x = self.dense_3(res8x) + x1[1]
            res4x = self.convd4x(res4x)
            res2x = self.dense_2(res4x) + x1[0]
            res2x = self.convd2x(res2x)
            x = res2x
            x = self.dense_1(x)
            x = self.convd1x(x)

            return x

    _zoo: dict = {
        "transweather-rain100l": {
            "path"        : "transweather-rain100l.pt",
            "num_channels": 32,
        },
    }
    
    def __init__(
        self,
        config       : Any                = None,
        loss         : Any                = Loss(),
        variant      :         str | None = None,
        num_channels : int   | str        = 32,
        scale_factor : float | str        = 1.0,
        num_iters    : int   | str        = 8,
        unsharp_sigma: int   | str | None = None,
        *args, **kwargs
    ):
        self.name          = "transweather"
        super().__init__(
            config = config,
            loss   = loss,
            *args, **kwargs
        )
        variant            = core.to_int()
        self.variant       = f"{variant:04d}" if isinstance(variant, int) else None
        self.num_channels  = core.to_int(num_channels)    or 32
        self.scale_factor  = core.to_float(scale_factor)  or 1.0
        self.num_iters     = core.to_int(num_iters)       or 8
        self.unsharp_sigma = core.to_float(unsharp_sigma) or None
        self.previous      = None

        if variant is None:  # Default model
            self.upsample   = nn.UpsamplingBilinear2d(self.scale_factor)

            self.enc        = self.Encoder()
            self.dec        = self.Decoder()
            self.convtail   = self.convprojection()
            self.clean      = nn.Conv2d(8, 3, kernel_size=3, stride=1, padding=1)
            self.act     = nn.Tanh()

            self.loss         = Loss(
                weight_smooth   = 1,
                weight_perL1    = 0.1,
                weight_ccp      = 2,
                reduction       = "mean",
            )
            self.apply(self._init_weights)
        else:
            pass #self.config_model_variant()

    # def config_model_variant(self):
    #     """Config the model based on ``self.variant``.
    #     Mainly used in ablation study.
    #     """
    #     # self.gamma         = 2.8
    #     # self.num_iters     = 9
    #     # self.unsharp_sigma = 2.5
    #     self.previous      = None
    #     self.out_channels  = 3

    #     # Variant code: [aa][l][i]
    #     # i: inference mode
    #     if self.variant[3] == "0":
    #         self.out_channels = 3
    #     elif self.variant[3] == "1":
    #         self.out_channels = 3
    #     elif self.variant[3] == "2":
    #         self.out_channels = 3
    #     elif self.variant[3] == "3":
    #         self.out_channels = 3
    #     else:
    #         raise ValueError

    #     # Variant code: [aa][l][i]
    #     # aa: architecture
    #     if self.variant[0:2] == "00":  # Zero-DCE (baseline)
    #         self.conv1    = nn.Conv2d(self.channels,         self.num_channels, 3, 1, 1, bias=True)
    #         self.conv2    = nn.Conv2d(self.num_channels,     self.num_channels, 3, 1, 1, bias=True)
    #         self.conv3    = nn.Conv2d(self.num_channels,     self.num_channels, 3, 1, 1, bias=True)
    #         self.conv4    = nn.Conv2d(self.num_channels,     self.num_channels, 3, 1, 1, bias=True)
    #         self.conv5    = nn.Conv2d(self.num_channels * 2, self.num_channels, 3, 1, 1, bias=True)
    #         self.conv6    = nn.Conv2d(self.num_channels * 2, self.num_channels, 3, 1, 1, bias=True)
    #         self.conv7    = nn.Conv2d(self.num_channels * 2, self.out_channels, 3, 1, 1, bias=True)
    #         self.attn     = nn.Identity()
    #         self.act      = nn.ReLU(inplace=True)
    #         self.upsample = nn.UpsamplingBilinear2d(self.scale_factor)
    #         self.apply(self.init_weights)
    #     elif self.variant[0:2] == "01":  # Zero-DCE++ (baseline)
    #         self.conv1    = nn.DSConv2d(self.channels,         self.num_channels, 3, 1, 1, bias=True)
    #         self.conv2    = nn.DSConv2d(self.num_channels,     self.num_channels, 3, 1, 1, bias=True)
    #         self.conv3    = nn.DSConv2d(self.num_channels,     self.num_channels, 3, 1, 1, bias=True)
    #         self.conv4    = nn.DSConv2d(self.num_channels,     self.num_channels, 3, 1, 1, bias=True)
    #         self.conv5    = nn.DSConv2d(self.num_channels * 2, self.num_channels, 3, 1, 1, bias=True)
    #         self.conv6    = nn.DSConv2d(self.num_channels * 2, self.num_channels, 3, 1, 1, bias=True)
    #         self.conv7    = nn.DSConv2d(self.num_channels * 2, self.out_channels, 3, 1, 1, bias=True)
    #         self.attn     = nn.Identity()
    #         self.act      = nn.ReLU(inplace=True)
    #         self.upsample = nn.UpsamplingBilinear2d(self.scale_factor)
    #         self.apply(self.init_weights)
    #     elif self.variant[0:2] == "02":
    #         self.conv1    = nn.BSConv2dS(self.channels,         self.num_channels, 3, 1, 1, bias=True)
    #         self.conv2    = nn.BSConv2dS(self.num_channels,     self.num_channels, 3, 1, 1, bias=True)
    #         self.conv3    = nn.BSConv2dS(self.num_channels,     self.num_channels, 3, 1, 1, bias=True)
    #         self.conv4    = nn.BSConv2dS(self.num_channels,     self.num_channels, 3, 1, 1, bias=True)
    #         self.conv5    = nn.BSConv2dS(self.num_channels * 2, self.num_channels, 3, 1, 1, bias=True)
    #         self.conv6    = nn.BSConv2dS(self.num_channels * 2, self.num_channels, 3, 1, 1, bias=True)
    #         self.conv7    = nn.BSConv2dS(self.num_channels * 2, self.out_channels, 3, 1, 1, bias=True)
    #         self.attn     = nn.Identity()
    #         self.act      = nn.ReLU(inplace=True)
    #         self.upsample = nn.UpsamplingBilinear2d(self.scale_factor)
    #         self.apply(self.init_weights)
    #     elif self.variant[0:2] == "03":
    #         self.conv1    = nn.BSConv2dU(self.channels,         self.num_channels, 3, 1, 1, bias=True)
    #         self.conv2    = nn.BSConv2dU(self.num_channels,     self.num_channels, 3, 1, 1, bias=True)
    #         self.conv3    = nn.BSConv2dU(self.num_channels,     self.num_channels, 3, 1, 1, bias=True)
    #         self.conv4    = nn.BSConv2dU(self.num_channels,     self.num_channels, 3, 1, 1, bias=True)
    #         self.conv5    = nn.BSConv2dU(self.num_channels * 2, self.num_channels, 3, 1, 1, bias=True)
    #         self.conv6    = nn.BSConv2dU(self.num_channels * 2, self.num_channels, 3, 1, 1, bias=True)
    #         self.conv7    = nn.BSConv2dU(self.num_channels * 2, self.out_channels, 3, 1, 1, bias=True)
    #         self.attn     = nn.Identity()
    #         self.act      = nn.ReLU(inplace=True)
    #         self.upsample = nn.UpsamplingBilinear2d(self.scale_factor)
    #         self.apply(self.init_weights)
    #     #
    #     elif self.variant[0:2] == "10":
    #         self.conv1    = nn.Conv2d(self.channels,         self.num_channels, 3, 1, 1, bias=True)
    #         self.conv2    = nn.Conv2d(self.num_channels,     self.num_channels, 3, 1, 1, bias=True)
    #         self.conv3    = nn.Conv2d(self.num_channels,     self.num_channels, 3, 1, 1, bias=True)
    #         self.conv4    = nn.Conv2d(self.num_channels,     self.num_channels, 3, 1, 1, bias=True)
    #         # Curve Enhancement Map (A)
    #         self.conv5    = nn.Conv2d(self.num_channels * 2, self.num_channels, 3, 1, 1, bias=True)
    #         self.conv6    = nn.Conv2d(self.num_channels * 2, self.num_channels, 3, 1, 1, bias=True)
    #         self.conv7    = nn.Conv2d(self.num_channels * 2, self.out_channels, 3, 1, 1, bias=True)
    #         # Guided Brightness Enhancement Map (G)
    #         self.conv8    = nn.Conv2d(self.num_channels * 2, self.num_channels, 3, 1, 1, bias=True)
    #         self.conv9    = nn.Conv2d(self.num_channels * 2, self.num_channels, 3, 1, 1, bias=True)
    #         self.conv10   = nn.Conv2d(self.num_channels * 2, 1, 3, 1, 1, bias=True)
    #         self.attn     = nn.Identity()
    #         self.act      = nn.ReLU(inplace=True)
    #         self.upsample = nn.UpsamplingBilinear2d(self.scale_factor)
    #         self.apply(self.init_weights)
    #     elif self.variant[0:2] == "11":
    #         self.conv1    = nn.Conv2d(self.channels,         self.num_channels, 3, 1, 1, bias=True)
    #         self.conv2    = nn.Conv2d(self.num_channels,     self.num_channels, 3, 1, 1, bias=True)
    #         self.conv3    = nn.Conv2d(self.num_channels,     self.num_channels, 3, 1, 1, bias=True)
    #         self.conv4    = nn.Conv2d(self.num_channels,     self.num_channels, 3, 1, 1, bias=True)
    #         self.conv5    = nn.Conv2d(self.num_channels,     self.num_channels, 3, 1, 1, bias=True)
    #         # Curve Enhancement Map (A)
    #         self.conv6    = nn.Conv2d(self.num_channels * 2, self.num_channels, 3, 1, 1, bias=True)
    #         self.conv7    = nn.Conv2d(self.num_channels * 2, self.num_channels, 3, 1, 1, bias=True)
    #         self.conv8    = nn.Conv2d(self.num_channels * 2, self.num_channels, 3, 1, 1, bias=True)
    #         self.conv9    = nn.Conv2d(self.num_channels * 2, self.out_channels, 3, 1, 1, bias=True)
    #         # Guided Brightness Enhancement Map (G)
    #         self.conv10   = nn.Conv2d(self.num_channels * 2, self.num_channels, 3, 1, 1, bias=True)
    #         self.conv11   = nn.Conv2d(self.num_channels * 2, self.num_channels, 3, 1, 1, bias=True)
    #         self.conv12   = nn.Conv2d(self.num_channels * 2, self.num_channels, 3, 1, 1, bias=True)
    #         self.conv13   = nn.Conv2d(self.num_channels * 2, 1, 3, 1, 1, bias=True)
    #         self.attn     = nn.Identity()
    #         self.act      = nn.ReLU(inplace=True)
    #         self.upsample = nn.UpsamplingBilinear2d(self.scale_factor)
    #         self.apply(self.init_weights)
    #     #
    #     elif self.variant[0:2] == "20":
    #         self.conv1    = nn.Conv2d(self.channels,     self.num_channels, 3, 1, 1, bias=True)
    #         self.conv2    = nn.Conv2d(self.num_channels, self.num_channels, 3, 1, 1, bias=True)
    #         self.conv3    = nn.Conv2d(self.num_channels, self.num_channels, 3, 1, 1, bias=True)
    #         self.conv4    = nn.Conv2d(self.num_channels, self.num_channels, 3, 1, 1, bias=True)
    #         # knot points
    #         self.conv5    = nn.Conv2d(self.num_channels, self.num_channels, 3, 1, 1, bias=True)
    #         self.conv6    = nn.Conv2d(self.num_channels, self.num_channels, 3, 1, 1, bias=True)
    #         self.conv7    = nn.Conv2d(self.num_channels, self.num_channels, 3, 1, 1, bias=True)
    #         self.conv8    = nn.Conv2d(self.num_channels, 24, 3, 1, 1, bias=True)
    #         # curve parameter
    #         self.conv9    = nn.Conv2d(self.num_channels * 2, self.num_channels, 3, 1, 1, bias=True)
    #         self.conv10   = nn.Conv2d(self.num_channels * 2, self.num_channels, 3, 1, 1, bias=True)
    #         self.conv11   = nn.Conv2d(self.num_channels * 2, 1, 3, 1, 1, bias=True)
    #         self.attn     = nn.Identity()
    #         self.act      = nn.ReLU(inplace=True)
    #         self.upsample = nn.UpsamplingBilinear2d(self.scale_factor)
    #         self.pool     = nn.MaxPool2d(2, 1)
    #         self.apply(self.init_weights)
    #     else:
    #         raise ValueError

    #     # Variant code: [aa][l][i]
    #     # l: loss function
    #     weight_tvA = 1600 if self.out_channels == 3 else 200
    #     if self.variant[2] == "0":  # Zero-DCE Loss
    #         # NOT WORKING: over-exposed artifacts, enhance noises
    #         self.loss = Loss(
    #             exp_patch_size  = 16,
    #             exp_mean_val    = 0.6,
    #             spa_num_regions = 4,
    #             spa_patch_size  = 4,
    #             weight_bri      = 0,
    #             weight_col      = 5,
    #             weight_crl      = 0,
    #             weight_edge     = 0,
    #             weight_exp      = 10,
    #             weight_kl       = 0,
    #             weight_spa      = 1,
    #             weight_tvA      = weight_tvA,
    #             reduction       = "mean",
    #         )
    #     elif self.variant[2] == "1":  # New Loss
    #         self.loss = Loss(
    #             exp_patch_size  = 16,
    #             exp_mean_val    = 0.6,
    #             spa_num_regions = 8,
    #             spa_patch_size  = 4,
    #             weight_bri      = 0,
    #             weight_col      = 5,
    #             weight_crl      = 0,
    #             weight_edge     = 1,
    #             weight_exp      = 10,
    #             weight_kl       = 0.1,
    #             weight_spa      = 1,
    #             weight_tvA      = weight_tvA,
    #             reduction       = "mean",
    #         )
    #     elif self.variant[2] == "2":
    #         self.loss = Loss(
    #             exp_patch_size  = 16,
    #             exp_mean_val    = 0.6,
    #             spa_num_regions = 8,
    #             spa_patch_size  = 4,
    #             weight_bri      = 0,
    #             weight_col      = 5,
    #             weight_crl      = 0.1,
    #             weight_edge     = 1,
    #             weight_exp      = 10,
    #             weight_kl       = 0.1,
    #             weight_spa      = 1,
    #             weight_tvA      = weight_tvA,
    #             reduction       = "mean",
    #         )
    #     elif self.variant[2] == "9":
    #         self.loss  = Loss(
    #             exp_patch_size  = 16,   # 16
    #             exp_mean_val    = 0.6,  # 0.6
    #             spa_num_regions = 8,    # 8
    #             spa_patch_size  = 4,    # 4
    #             weight_bri      = 10,   # 10
    #             weight_col      = 5,    # 5
    #             weight_crl      = 0.1,  # 0.1
    #             weight_edge     = 1,    # 1
    #             weight_exp      = 10,   # 10
    #             weight_kl       = 0.1,  # 0.1
    #             weight_spa      = 1,    # 1
    #             weight_tvA      = weight_tvA,  # weight_tvA,
    #             reduction       = "mean",
    #         )
    #     else:
    #         raise ValueError

    @property
    def config_dir(self) -> core.Path:
        return core.Path(__file__).absolute().parent / "config"
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            m.weight.data.trunc_()
            # torch.nn.init.trunc_normal_(m.weight, )
        if isinstance(m, nn.Linear) and m.bias is not None:
            m.bias.data.zero_()
            # torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            m.bias.data.zero_()
            m.weight.data.fill_(1.0)
            # torch.nn.init.constant_(m.bias, 0)
            # torch.nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    
    def forward_loss(
        self,
        input : torch.Tensor,
        target: torch.Tensor | None,
        *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        pred  = self.forward(input=input, *args, **kwargs)
        loss, self.previous = self.loss(pred, target) 
        loss += self.regularization_loss(alpha=0.1)
        return pred, loss

    def forward(
        self,
        input    : torch.Tensor,
        augment  : _callable = None,
        profile  : bool      = False,
        out_index: int       = -1,
        *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass. This is the primary :meth:`forward` function of the
        model. It supports augmented inference. In this function, we perform
        test-time augmentation and pass the transformed input to
        :meth:`forward_once()`.

        Args:
            input: An input of shape :math`[B, C, H, W]`.
            augment: If ``True``, perform test-time augmentation. Default:
                ``False``.
            profile: If ``True``, Measure processing time. Default: ``False``.
            out_index: Return specific layer's output from :param:`out_index`.
                Default: -1 means the last layer.

        Return:
            Predictions.
        """
        if augment:
            # For now just forward the input. Later, we will implement the
            # test-time augmentation.
            if self.variant is not None:
                pass # return self.forward_once_variant(input=input, profile=profile, *args, **kwargs)
            else:
                return self.forward_once(input=input, profile=profile, *args, **kwargs)
        else:
            if self.variant is not None:
                pass # return self.forward_once_variant(input=input, profile=profile, *args, **kwargs)
            else:
                return self.forward_once(input=input, profile=profile, *args, **kwargs)

    def forward_once(
        self,
        input    : torch.Tensor,
        profile  : bool = False,
        out_index: int = -1,
        *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
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

        # Downsampling
        x_down = x
        if self.scale_factor != 1:
            x_down = F.interpolate(x, scale_factor=1 / self.scale_factor, mode="bilinear")

        # Enhancement
        f1 = self.enc(x_down)
        f2 = self.dec(f1[3])
        f3 = self.convtail(f1,f2)
        y = self.act(self.clean(f3))
        
        # Upsampling
        if self.scale_factor != 1:
            a = self.upsample(a)

        
        # Unsharp masking
        if self.unsharp_sigma is not None:
            y = kornia.filters.unsharp_mask(y, (3, 3), (self.unsharp_sigma, self.unsharp_sigma))

        return y

    # def forward_once_variant(
    #     self,
    #     input    : torch.Tensor,
    #     profile  : bool = False,
    #     out_index: int  = -1,
    #     *args, **kwargs
    # ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    #     """Forward pass once. Implement the logic for a single forward pass. Mainly used for ablation study.

    #     Args:
    #         input: An input of shape :math:`[N, C, H, W]`.
    #         profile: Measure processing time. Default: ``False``.
    #         out_index: Return specific layer's output from :param:`out_index`.
    #             Default: ``-1`` means the last layer.

    #     Return:
    #         Predictions.
    #     """
    #     x = input

    #     # Downsampling
    #     x_down = x
    #     if self.scale_factor != 1:
    #         x_down = F.interpolate(x, scale_factor=1 / self.scale_factor, mode="bilinear")

    #     # Variant code: [aa][l][e]
    #     if self.variant[0:2] in ["10", "12"]:
    #         f1  = self.act(self.conv1(x_down))
    #         f2  = self.act(self.conv2(f1))
    #         f3  = self.act(self.conv3(f2))
    #         f4  = self.act(self.conv4(f3))
    #         f4  = self.attn(f4)
    #         # Curve Enhancement Map (A)
    #         f5  = self.act(self.conv5(torch.cat([f3, f4], dim=1)))
    #         f6  = self.act(self.conv6(torch.cat([f2, f5], dim=1)))
    #         a   =   F.tanh(self.conv7(torch.cat([f1, f6], dim=1)))
    #         # Guided Brightness Enhancement Map (GBEM)
    #         f8  = self.act(self.conv8(torch.cat([f3, f4], dim=1)))
    #         f9  = self.act(self.conv9(torch.cat([f2, f8], dim=1)))
    #         g   =  F.tanh(self.conv10(torch.cat([f1, f9], dim=1)))
    #     elif self.variant[0:2] in ["11"]:
    #         f1  = self.act(self.conv1(x_down))
    #         f2  = self.act(self.conv2(f1))
    #         f3  = self.act(self.conv3(f2))
    #         f4  = self.act(self.conv4(f3))
    #         f5  = self.act(self.conv5(f4))
    #         f5  = self.attn(f5)
    #         # Curve Enhancement Map (A)
    #         f6  = self.act(self.conv6(torch.cat([f4, f5], dim=1)))
    #         f7  = self.act(self.conv7(torch.cat([f3, f6], dim=1)))
    #         f8  = self.act(self.conv8(torch.cat([f2, f7], dim=1)))
    #         a   =   F.tanh(self.conv9(torch.cat([f1, f8], dim=1)))
    #         # Guided Brightness Enhancement Map (GBEM)
    #         f9  = self.act(self.conv10(torch.cat([f4,  f5], dim=1)))
    #         f10 = self.act(self.conv11(torch.cat([f3,  f9], dim=1)))
    #         f11 = self.act(self.conv12(torch.cat([f2, f10], dim=1)))
    #         g   =   F.tanh(self.conv13(torch.cat([f1, f11], dim=1)))
    #     elif self.variant[0:2] in ["20"]:
    #         f1  = self.act(self.conv1(x_down))
    #         f2  = self.act(self.conv2(f1))
    #         f3  = self.act(self.conv3(f2))
    #         f4  = self.act(self.conv4(f3))
    #         # knot points
    #         f5  = self.pool(self.pool(self.act(self.conv5(f4))))
    #         f6  = self.pool(self.pool(self.act(self.conv6(f5))))
    #         f7  = self.pool(self.pool(self.act(self.conv7(f6))))
    #         k   = F.adaptive_avg_pool2d(self.conv8(f7), (1, 1))
    #         # curve parameters
    #         f9  = self.act(self.conv9(torch.cat([f3, f4],   dim=1)))
    #         f10 = self.act(self.conv10(torch.cat([f2, f9],  dim=1)))
    #         a   =   F.tanh(self.conv11(torch.cat([f1, f10], dim=1)))
    #     else:
    #         f1  = self.act(self.conv1(x_down))
    #         f2  = self.act(self.conv2(f1))
    #         f3  = self.act(self.conv3(f2))
    #         f4  = self.act(self.conv4(f3))
    #         f4  = self.attn(f4)
    #         f5  = self.act(self.conv5(torch.cat([f3, f4], dim=1)))
    #         f6  = self.act(self.conv6(torch.cat([f2, f5], dim=1)))
    #         a   =   F.tanh(self.conv7(torch.cat([f1, f6], dim=1)))

    #     # Upsampling
    #     if self.scale_factor != 1:
    #         a = self.upsample(a)

    #     # Enhancement
    #     if "1" in self.variant[0:1]:
    #         if self.out_channels == 3:
    #             y = x
    #             for _ in range(self.num_iters):
    #                 b = y * (1 - g)
    #                 d = y * g
    #                 y = b + d + a * (torch.pow(d, 2) - d)
    #         else:
    #             y = x
    #             A = torch.split(a, 3, dim=1)
    #             for i in range(self.num_iters):
    #                 b = y * (1 - g)
    #                 d = y * g
    #                 y = b + d + A[i] * (torch.pow(d, 2) - d)
    #     # Piece-wise
    #     elif "2" in self.variant[0:1]:
    #         K = torch.split(k, 3, dim=1)
    #         # A = torch.split(a, 3, dim=1)
    #         y = x  # K[0]
    #         for m in range(0, 7):
    #             k_m1 = K[m + 1]
    #             k_m  = K[m]
    #             # a_m  = A[m]
    #             S    = y
    #             for n in range(0, 4):
    #                 S = S + a * (torch.pow(S, 2) - S)
    #             y = y + (k_m1 - k_m) * S
    #     # Default
    #     elif self.variant[3] == "0":
    #         if self.out_channels == 3:
    #             y = x
    #             for _ in range(self.num_iters):
    #                 y = y + a * (torch.pow(y, 2) - y)
    #         else:
    #             y = x
    #             A = torch.split(a, 3, dim=1)
    #             for i in range(self.num_iters):
    #                 y = y + A[i] * (torch.pow(y, 2) - y)
    #     # Global G
    #     elif self.variant[3] == "1":
    #         if self.out_channels == 3:
    #             y = x
    #             for _ in range(self.num_iters):
    #                 b = y * (1 - g)
    #                 d = y * g
    #                 y = b + d + a * (torch.pow(d, 2) - d)
    #         else:
    #             y = x
    #             A = torch.split(a, 3, dim=1)
    #             for i in range(self.num_iters):
    #                 b = y * (1 - g)
    #                 d = y * g
    #                 y = b + d + A[i] * (torch.pow(d, 2) - d)
    #     # Global G Inference Only
    #     elif self.variant[3] == "2":
    #         if self.out_channels == 3:
    #             if self.phase == ModelPhase.TRAINING:
    #                 y = x
    #                 for _ in range(self.num_iters):
    #                     y = y + a * (torch.pow(y, 2) - y)
    #             else:
    #                 y = x
    #                 for _ in range(self.num_iters):
    #                     b = y * (1 - g)
    #                     d = y * g
    #                     y = b + d + a * (torch.pow(d, 2) - d)
    #         else:
    #             if self.phase == ModelPhase.TRAINING:
    #                 y = x
    #                 A = torch.split(a, 3, dim=1)
    #                 for i in range(self.num_iters):
    #                     y = y + A[i] * (torch.pow(y, 2) - y)
    #             else:
    #                 y = x
    #                 A = torch.split(a, 3, dim=1)
    #                 for i in range(self.num_iters):
    #                     b = y * (1 - g)
    #                     d = y * g
    #                     y = b + d + A[i] * (torch.pow(d, 2) - d)
    #     # Iterative G Inference Only
    #     elif self.variant[3] == "3":
    #         if self.out_channels == 3:
    #             if self.phase == ModelPhase.TRAINING:
    #                 y = x
    #                 for _ in range(self.num_iters):
    #                     y = y + a * (torch.pow(y, 2) - y)
    #             else:
    #                 y = x
    #                 for _ in range(self.num_iters):
    #                     b = y * (1 - g)
    #                     d = y * g
    #                     y = b + d + a * (torch.pow(d, 2) - d)
    #         else:
    #             if self.phase == ModelPhase.TRAINING:
    #                 y = x
    #                 A = torch.split(a, 3, dim=1)
    #                 for i in range(self.num_iters):
    #                     y = y + A[i] * (torch.pow(y, 2) - y)
    #             else:
    #                 y = x
    #                 A = torch.split(a, 3, dim=1)
    #                 for i in range(self.num_iters):
    #                     b = y * (1 - g)
    #                     d = y * g
    #                     y = b + d + A[i] * (torch.pow(d, 2) - d)

    #     # Unsharp masking
    #     if self.unsharp_sigma is not None:
    #         y = kornia.filters.unsharp_mask(y, (3, 3), (self.unsharp_sigma, self.unsharp_sigma))

    #     #
    #     if "1" in self.variant[0:1]:
    #         return a, g, y
    #     return a, y

    def regularization_loss(self, alpha: float = 0.1):
        loss = 0.0
        for sub_module in [
            self.enc, self.dec, self.convtail, self.clean
        ]:
            if hasattr(sub_module, "regularization_loss"):
                loss += sub_module.regularization_loss()
        return alpha * loss

# endregion

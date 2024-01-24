#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements Transweather models.

./bin/enhance/run.sh derain train rain13k rain13k transweather none none vision/enhance/derain 100
./bin/enhance/run.sh derain train rain13k rain13k transweather transweather_b none vision/enhance/derain 100
"""

from __future__ import annotations

__all__ = [
    "Transweather",
    "TransweatherB",
]

from functools import partial
from typing import Any

import kornia
import torch
from torchvision.models import VGG16_Weights
from torchvision.models import vgg16

import mon
from mon import core, nn
from mon.globals import MODELS, LAYERS
from mon.nn import functional as F
from mon.vision.enhance.derain import base
from mon.core.typing import _callable, _size_2_t
from mon.nn.loss import PerceptualL1Loss, ContradictChannelLoss
from mon.vision.feature import OPEmbedder
from mon.vision.enhance.derain.modules.transweather_layers import Block_dec, Transformer_Block, CBAM

math = core.math
console = core.console
_current_dir = core.Path(__file__).absolute().parent


# region Loss
class Loss(nn.Loss):

    def __init__(
            self,
            weight_smooth: float = 1,
            weight_perL1: float = 0.1,
            weight_ccp: float = 2,
            reduction: str = "mean",
            verbose: bool = False,
            *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.verbose = verbose
        self.weight_smooth = weight_smooth
        self.weight_perL1 = weight_perL1
        self.weight_ccp = weight_ccp
        self.loss_smooth = nn.SmoothL1Loss(reduction=reduction)
        self.loss_perL1 = PerceptualL1Loss(vgg16(weights=VGG16_Weights.DEFAULT).features[:16], reduction=reduction)
        self.loss_ccp = ContradictChannelLoss(reduction=reduction)

    def __str__(self) -> str:
        return f"loss"

    def forward(
            self,
            pred: torch.Tensor | list[torch.Tensor],
            target: torch.Tensor | list[torch.Tensor],
            **_
    ) -> tuple[torch.Tensor, torch.Tensor]:
        loss_smooth = self.loss_smooth(input=pred, target=target) if self.weight_smooth > 0 else 0
        loss_perL1 = self.loss_perL1(input=pred, target=target) if self.weight_perL1 > 0 else 0
        loss_ccp = self.loss_ccp(input=pred, target=target) if self.weight_ccp > 0 else 0

        loss = (
                self.weight_smooth * loss_smooth
                + self.weight_perL1 * loss_perL1
                + self.weight_ccp * loss_ccp
        )

        if self.verbose:
            console.log(f"{self.loss_smooth.__str__():<30} : {loss_smooth}")
            console.log(f"{self.loss_perL1.__str__():<30} : {loss_perL1}")
            console.log(f"{self.loss_ccp.__str__():<30}: {loss_ccp}")
        return loss, pred
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
                img_size: _size_2_t = 224,
                patch_size: list[int] = [7, 3, 3, 3],
                in_chans: int = 3,
                embed_dims: list[int] = [64, 128, 320, 512],
                num_heads: list[int] = [1, 2, 4, 4],
                mlp_ratios: list[int] = [2, 2, 2, 2],
                qkv_bias: bool = True,
                drop_rate: float = 0.,
                attn_drop_rate: float = 0.,
                drop_path_rate: float = 0.1,
                norm_layer: object = partial(nn.LayerNorm, eps=1e-6),
                depths: list[int] = [2, 2, 2, 2],
                sr_ratios: list[int] = [4, 2, 2, 1]
        ):
            super().__init__()
            self.num_stages = len(depths)
            self.depths = depths
            self.embed_dims = embed_dims

            # region Embedding
            self.patch_embed = nn.Sequential

            self.patch_embed = nn.ModuleList()
            self.patch_embed.append(OPEmbedder(
                image_size=img_size[0],
                patch_size=patch_size[0],
                stride=int((patch_size[0] + 1) / 2),
                in_channels=in_chans,
                embed_dim=embed_dims[0]
            ))
            for stage in range(1, self.num_stages):
                self.patch_embed.append(OPEmbedder(
                    image_size=img_size[0] // (2 ** (stage + 1)),
                    patch_size=patch_size[stage],
                    stride=int((patch_size[stage] + 1) / 2),
                    in_channels=embed_dims[stage - 1],
                    embed_dim=embed_dims[stage]
                ))
            self.mini_patch_embed = nn.ModuleList(
                [OPEmbedder(
                    image_size=img_size[0] // (2 ** (stage + 2)),
                    patch_size=patch_size[stage + 1],
                    stride=int((patch_size[stage + 1] + 1) / 2),
                    in_channels=embed_dims[stage],
                    embed_dim=embed_dims[stage + 1]
                )
                    for stage in range(self.num_stages - 1)]
            )
            # endregion
            # region Transformer encoder
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
            cur = 0

            self.block = nn.ModuleList()
            self.norm = nn.ModuleList()
            for stage in range(self.num_stages):
                self.block.append(nn.ModuleList(
                    [
                        Transformer_Block(
                            dim=embed_dims[stage],
                            num_heads=num_heads[stage],
                            mlp_ratio=mlp_ratios[stage],
                            qkv_bias=qkv_bias,

                            drop=drop_rate,
                            attn_drop=attn_drop_rate,
                            drop_path=dpr[cur + i],
                            norm_layer=norm_layer,
                            sr_ratio=sr_ratios[stage])
                        for i in range(depths[stage])
                    ]
                ))
                self.norm.append(norm_layer(embed_dims[stage]))
                cur += depths[stage]
            # endregion
            # region Intra-patch encoder
            self.patch_block = nn.ModuleList()
            self.pnorm = nn.ModuleList()
            for stage in range(self.num_stages - 1):
                self.patch_block.append(nn.ModuleList(
                    [
                        CBAM(
                            channels=embed_dims[stage + 1],
                            reduction_ratio=2,
                            spatial=True)
                        for _ in range(self.depths[stage + 1])
                    ]
                ))
                self.pnorm.append(norm_layer(embed_dims[stage + 1]))
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
                self.block1[i].drop_path.drop_prob = dpr[cur + i]  # type: ignore

            cur += self.depths[0]
            for i in range(self.depths[1]):
                self.block2[i].drop_path.drop_prob = dpr[cur + i]  # type: ignore

            cur += self.depths[1]
            for i in range(self.depths[2]):
                self.block3[i].drop_path.drop_prob = dpr[cur + i]  # type: ignore

            cur += self.depths[2]
            for i in range(self.depths[3]):
                self.block4[i].drop_path.drop_prob = dpr[cur + i]  # type: ignore

        def forward_features(self, x):
            B = x.shape[0]
            outs = []
            x1 = x
            x2 = 0
            for stage in range(self.num_stages - 1):
                x1, H1, W1 = self.patch_embed[stage].embed(
                    images=x1,
                    norm=True
                )
                x2, H2, W2 = self.mini_patch_embed[stage].embed(
                    x1.permute(0, 2, 1).reshape(B, self.embed_dims[stage], H1, W1) + x2,
                    norm=True
                )

                for i, blk in enumerate(self.block[stage]):
                    x1 = blk(x1, H1, W1)
                x1 = self.norm[stage](x1)
                x1 = x1.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()

                for i, blk in enumerate(self.patch_block[stage]):
                    x2 = blk(x2, H2, W2)
                x2 = self.pnorm[stage](x2)
                x2 = x2.reshape(B, H2, W2, -1).permute(0, 3, 1, 2).contiguous()

                outs.append(x1)
            x1, H1, W1 = self.patch_embed[-1].embed(
                images=x1,
                norm=True
            )
            x1 = x1.permute(0, 2, 1).reshape(B, self.embed_dims[-1], H1, W1) + x2

            x1 = x1.view(x1.shape[0], x1.shape[1], -1).permute(0, 2, 1)

            for i, blk in enumerate(self.block[-1]):
                x1 = blk(x1, H1, W1)
            x1 = self.norm[-1](x1)
            x1 = x1.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x1)

            return outs

        def forward(self, x):
            x = self.forward_features(x)
            return x

    class Decoder(nn.Module):
        def __init__(
                self,
                img_size: _size_2_t = 256,
                patch_size=16,
                in_chans=3,
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
            self.depths = depths
            self.embed_dims = embed_dims

            # patch_embed
            self.patch_embed = OPEmbedder(
                image_size=img_size[0] // 16,
                patch_size=patch_size,
                stride=int((patch_size + 1) / 2),
                in_channels=embed_dims[-1],
                embed_dim=embed_dims[-1]
            )

            # transformer decoder
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
            cur = 0
            self.block = nn.ModuleList(
                [
                    Block_dec(
                        dim=embed_dims[-1], num_heads=num_heads[-1], mlp_ratio=mlp_ratios[-1], qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
                        sr_ratio=sr_ratios[-1]
                    )
                    for i in range(depths[-1])
                ])
            self.norm = norm_layer(embed_dims[-1])

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
            # x=x[3]
            B = x.shape[0]
            outs = []

            # stage 1
            x, H, W = self.patch_embed.embed(
                images=x,
                norm=True
            )
            for i, blk in enumerate(self.block):
                x = blk(x, H, W)
            x = self.norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)

            return outs

        def forward(self, x):
            x = self.forward_features(x)
            # x = self.head(x)

            return x

    class ConvProjection(nn.Module):
        class ResidualBlock(torch.nn.Module):
            def __init__(self, channels):
                super().__init__()
                # Residual Block
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

        def forward(self, x1, x2):

            res32x = self.convd32x(x2[0])

            if x1[3].shape[3] != res32x.shape[3] and x1[3].shape[2] != res32x.shape[2]:
                p2d = (0, -1, 0, -1)
                res32x = F.pad(res32x, p2d, "constant", 0)

            elif x1[3].shape[3] != res32x.shape[3] and x1[3].shape[2] == res32x.shape[2]:
                p2d = (0, -1, 0, 0)
                res32x = F.pad(res32x, p2d, "constant", 0)
            elif x1[3].shape[3] == res32x.shape[3] and x1[3].shape[2] != res32x.shape[2]:
                p2d = (0, 0, 0, -1)
                res32x = F.pad(res32x, p2d, "constant", 0)

            res16x = res32x + x1[3]
            res16x = self.convd16x(res16x)

            if x1[2].shape[3] != res16x.shape[3] and x1[2].shape[2] != res16x.shape[2]:
                p2d = (0, -1, 0, -1)
                res16x = F.pad(res16x, p2d, "constant", 0)
            elif x1[2].shape[3] != res16x.shape[3] and x1[2].shape[2] == res16x.shape[2]:
                p2d = (0, -1, 0, 0)
                res16x = F.pad(res16x, p2d, "constant", 0)
            elif x1[2].shape[3] == res16x.shape[3] and x1[2].shape[2] != res16x.shape[2]:
                p2d = (0, 0, 0, -1)
                res16x = F.pad(res16x, p2d, "constant", 0)

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

    zoo = {
        "transweather-rain100l": {
            "path": "transweather-rain100l.pt",
            "channels": 3,
        },
    }

    def __init__(
            self,
            image_size: _size_2_t = 256,
            channels: int = 3,
            embed_dims: list[int] = [64, 128, 320, 512],
            loss: Any = Loss(),
            enc_config=None,
            dec_config=None,
            weights: Any = None,
            name: str = "transweather",
            *args, **kwargs
    ):
        dec_config = dec_config if dec_config is not None else {}
        enc_config = enc_config if enc_config is not None else {}

        self.name = name
        super().__init__(
            channels=channels,
            weights=weights,
            name=name,
            loss=loss,
            *args, **kwargs
        )
        self.channels = mon.to_int(channels) or 3
        self.previous = None

        self.enc = self.Encoder(
            img_size=image_size,
            patch_size=enc_config.get("patch_size", [7, 3, 3, 3]),
            in_chans=channels,
            embed_dims=embed_dims,
            num_heads=enc_config.get("num_heads", [1, 2, 4, 4]),
            mlp_ratios=enc_config.get("mlp_ratios", [2, 2, 2, 2]),
            qkv_bias=enc_config.get("qkv_bias", True),
            drop_rate=enc_config.get("drop_rate", 0.),
            attn_drop_rate=enc_config.get("attn_drop_rate", 0.),
            drop_path_rate=enc_config.get("drop_path_rate", 0.1),
            depths=enc_config.get("depths", [2, 2, 2, 2]),
            sr_ratios=enc_config.get("sr_ratios", [4, 2, 2, 1])
        )
        self.dec = self.Decoder(
            img_size=image_size,
            patch_size=dec_config.get("patch_size", 3),
            in_chans=channels,
            embed_dims=embed_dims,
            num_heads=dec_config.get("num_heads", [1, 2, 5, 8]),
            mlp_ratios=dec_config.get("mlp_ratios", [4, 4, 4, 4]),
            qkv_bias=dec_config.get("qkv_bias", True),
            qk_scale=dec_config.get("qk_scale", None),
            drop_rate=dec_config.get("drop_rate", 0.),
            attn_drop_rate=dec_config.get("attn_drop_rate", 0.),
            drop_path_rate=dec_config.get("drop_path_rate", 0.1),
            depths=dec_config.get("depths", [3, 4, 6, 3]),
            sr_ratios=dec_config.get("sr_ratios", [8, 4, 2, 1])
        )
        self.convtail = self.ConvProjection()
        self.clean = nn.Conv2d(8, 3, kernel_size=3, stride=1, padding=1)
        self.act = nn.Tanh()

        self.loss = Loss(
            weight_smooth=1,
            weight_perL1=0.1,
            weight_ccp=2,
            reduction="mean",
        )
        self.apply(self.init_weights)

    @property
    def config_dir(self) -> core.Path:
        return core.Path(__file__).absolute().parent / "config"

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

    def forward_loss(
            self,
            input: torch.Tensor,
            target: torch.Tensor | None,
            *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
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
        loss, self.previous = self.loss(pred, target)
        loss += self.regularization_loss(alpha=0.1)
        return pred, loss

    def forward(
            self,
            input: torch.Tensor,
            augment: bool = False,
            profile: bool = False,
            out_index: int = -1,
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
            return self.forward_once(input=input, profile=profile, *args, **kwargs)
        else:
            return self.forward_once(input=input, profile=profile, *args, **kwargs)

    def forward_once(
            self,
            input: torch.Tensor,
            profile: bool = False,
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
        # if self.scale_factor != 1:
        #     x_down = F.interpolate(x, scale_factor=1 / self.scale_factor, mode="bilinear")

        # Enhancement
        f1 = self.enc(x_down)
        f2 = self.dec(f1[-1])
        f3 = self.convtail(f1, f2)
        y = self.act(self.clean(f3))

        # Upsampling
        # if self.scale_factor != 1:
        #     y = self.upsample(y)

        # Unsharp masking
        # if self.unsharp_sigma is not None:
        #     y = kornia.filters.unsharp_mask(y, (3, 3), (self.unsharp_sigma, self.unsharp_sigma))

        return y

    def regularization_loss(self, alpha: float = 0.1):
        loss = 0.0
        for sub_module in [
            self.enc, self.dec, self.convtail, self.clean
        ]:
            if hasattr(sub_module, "regularization_loss"):
                loss += sub_module.regularization_loss()
        return alpha * loss


@MODELS.register(name="transweather_b")
class TransweatherB(Transweather):
    """Transweather model.

    See Also: :class:`mon.vision.enhance.derain.base.DerainingModel`
    """

    zoo = {
    }

    def __init__(
            self,
            weights: Any = None,
            name: str = "transweather",
            variant: str = "transweather_b",
            *args, **kwargs
    ):
        self.name = name
        super().__init__(
            embed_dims=[64, 128, 320, 512],
            loss=Loss(),
            enc_config={
                "patch_size": [7, 3, 3, 3],
                "num_heads": [1, 2, 4, 4],
                "mlp_ratios": [2, 2, 2, 2],
                "qkv_bias": True,
                "drop_rate": 0.1,
                "attn_drop_rate": 0.1,
                "drop_path_rate": 0.1,
                "depths": [3, 8, 6, 3],
                "sr_ratios": [4, 2, 2, 1]
            },
            dec_config={
                "patch_size": 3,
                "num_heads": [1, 2, 5, 8],
                "mlp_ratios": [4, 4, 4, 4],
                "qkv_bias": True,
                "qk_scale": None,
                "drop_rate": 0.1,
                "attn_drop_rate": 0.1,
                "drop_path_rate": 0.1,
                "depths": [3, 4, 6, 3],
                "sr_ratios": [8, 4, 2, 1]
            },
            weights=weights,
            name=name,
            variant=variant,
            *args, **kwargs
        )

# endregion

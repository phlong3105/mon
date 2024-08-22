#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements ViT (Vision Transformer) models."""

from __future__ import annotations

__all__ = [
    "ViT",
    "ViT_B_16",
    "ViT_B_32",
    "ViT_H_14",
    "ViT_L_16",
    "ViT_L_32",
    "VisionTransformer",
]

import functools
import math
from abc import ABC
from collections import OrderedDict
from typing import Any, NamedTuple

import torch

from mon import core, nn
from mon.core import _callable
from mon.globals import MODELS, Scheme
from mon.vision.classify import base

console = core.console


# region Module

class ConvStemConfig(NamedTuple):
    out_channels    : int
    kernel_size     : int
    stride          : int
    norm_layer      : _callable = nn.BatchNorm2d
    activation_layer: _callable = nn.ReLU


class MLPBlock(nn.MLP):
    """Transformer MLP block."""

    _version = 2

    def __init__(self, in_dim: int, mlp_dim: int, dropout: float):
        super().__init__(
            in_channels      = in_dim,
            hidden_channels  = [mlp_dim, in_dim],
            activation_layer = nn.GELU,
            inplace          = None,
            dropout          = dropout,
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias:
                    torch.nn.init.normal_(m.bias, std=1e-6)

    def _load_from_state_dict(
        self,
        state_dict     : dict,
        prefix         : str,
        local_metadata : dict,
        strict         : bool,
        missing_keys   : list[str],
        unexpected_keys: list[str],
        error_msgs     : list[str],
    ):
        version = local_metadata.get("version", None)
        if version is None or version < 2:
            # Replacing legacy MLPBlock with MLP. See https://github.com/pytorch/vision/pull/6053
            for i in range(2):
                for type in ["weight", "bias"]:
                    old_key = f"{prefix}linear_{i+1}.{type}"
                    new_key = f"{prefix}{3*i}.{type}"
                    if old_key in state_dict:
                        state_dict[new_key] = state_dict.pop(old_key)

        super()._load_from_state_dict(
            state_dict      = state_dict,
            prefix          = prefix,
            local_metadata  = local_metadata,
            strict          = strict,
            missing_keys    = missing_keys,
            unexpected_keys = unexpected_keys,
            error_msgs      = error_msgs,
        )


class EncoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(
        self,
        num_heads        : int,
        hidden_dim       : int,
        mlp_dim          : int,
        dropout          : float,
        attention_dropout: float,
        norm_layer       : _callable = functools.partial(nn.LayerNorm, eps = 1e-6),
        *args, **kwargs
    ):
        super().__init__()
        self.num_heads = num_heads
        # Attention block
        self.ln_1           = norm_layer(hidden_dim)
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)
        self.dropout        = nn.Dropout(dropout)
        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp  = MLPBlock(hidden_dim, mlp_dim, dropout)

    def forward(self, input: torch.Tensor) ->  torch.Tensor:
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        x = self.ln_1(input)
        x, _ = self.self_attention(x, x, x, need_weights=False)
        x = self.dropout(x)
        x = x + input
        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y


class Encoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    def __init__(
        self,
        seq_length       : int,
        num_layers       : int,
        num_heads        : int,
        hidden_dim       : int,
        mlp_dim          : int,
        dropout          : float,
        attention_dropout: float,
        norm_layer       : _callable = functools.partial(nn.LayerNorm, eps = 1e-6),
        *args, **kwargs
    ):
        super().__init__()
        # Note that batch_size is on the first dim because we have
        # batch_first=True in nn.MultiAttention() by default
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))  # from BERT
        self.dropout       = nn.Dropout(dropout)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = EncoderBlock(
                num_heads         = num_heads,
                hidden_dim        = hidden_dim,
                mlp_dim           = mlp_dim,
                dropout           = dropout,
                attention_dropout = attention_dropout,
                norm_layer        = norm_layer,
            )
        self.layers = nn.Sequential(layers)
        self.ln     = norm_layer(hidden_dim)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        input = input + self.pos_embedding
        return self.ln(self.layers(self.dropout(input)))

# endregion


# region Model

class VisionTransformer(base.ImageClassificationModel, ABC):
    """Vision Transformer as per https://arxiv.org/abs/2010.11929."""
    
    arch   : str  = "vit"
    schemes: list[Scheme] = [Scheme.SUPERVISED]
    zoo    : dict = {}
    
    def __init__(
        self,
        image_size         : int,
        patch_size         : int,
        num_layers         : int,
        num_heads          : int,
        hidden_dim         : int,
        mlp_dim            : int,
        dropout            : float      = 0.0,
        attention_dropout  : float      = 0.0,
        in_channels        : int        = 3,
        num_classes        : int        = 1000,
        representation_size: int | None = None,
        norm_layer         : _callable = functools.partial(nn.LayerNorm, eps = 1e-6),
        conv_stem_configs  : list[ConvStemConfig] | None    = None,
        weights            : Any = None,
        *args, **kwargs
    ):
        super().__init__(
            in_channels = in_channels,
            num_classes = num_classes,
            weights     = weights,
            *args, **kwargs
        )
        assert image_size % patch_size == 0, "Input shape indivisible by patch size!"
        self.image_size          = image_size
        self.patch_size          = patch_size
        self.hidden_dim          = hidden_dim
        self.mlp_dim             = mlp_dim
        self.attention_dropout   = attention_dropout
        self.dropout             = dropout
        self.representation_size = representation_size
        self.norm_layer          = norm_layer
        
        if conv_stem_configs:
            # As per https://arxiv.org/abs/2106.14881
            seq_proj = nn.Sequential()
            prev_channels = 3
            for i, conv_stem_layer_config in enumerate(conv_stem_configs):
                seq_proj.add_module(
                    f"conv_bn_relu_{i}",
                    nn.Conv2dNormAct(
                        in_channels      = prev_channels,
                        out_channels     = conv_stem_layer_config.out_channels,
                        kernel_size      = conv_stem_layer_config.kernel_size,
                        stride           = conv_stem_layer_config.stride,
                        norm_layer       = conv_stem_layer_config.norm_layer,
                        activation_layer = conv_stem_layer_config.activation_layer,
                    ),
                )
                prev_channels = conv_stem_layer_config.out_channels
            seq_proj.add_module(
                "conv_last", nn.Conv2d(in_channels=prev_channels, out_channels=hidden_dim, kernel_size=1)
            )
            self.conv_proj: nn.Module = seq_proj
        else:
            self.conv_proj = nn.Conv2d(
                in_channels  = self.in_channels,
                out_channels = hidden_dim,
                kernel_size  = patch_size,
                stride       = patch_size,
            )

        seq_length = (image_size // patch_size) ** 2

        # Add a class token
        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        seq_length      += 1

        self.encoder = Encoder(
            seq_length        = seq_length,
            num_layers        = num_layers,
            num_heads         = num_heads,
            hidden_dim        = hidden_dim,
            mlp_dim           = mlp_dim,
            dropout           = dropout,
            attention_dropout = attention_dropout,
            norm_layer        = norm_layer,
        )
        self.seq_length = seq_length

        heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
        if representation_size is None:
            heads_layers["head"]       = nn.Linear(hidden_dim, self.num_classes)
        else:
            heads_layers["pre_logits"] = nn.Linear(hidden_dim, representation_size)
            heads_layers["act"]        = nn.Tanh()
            heads_layers["head"]       = nn.Linear(representation_size, self.num_classes)

        self.heads = nn.Sequential(heads_layers)

        if isinstance(self.conv_proj, nn.Conv2d):
            # Init the patchify stem
            fan_in = self.conv_proj.in_channels * self.conv_proj.kernel_size[0] * self.conv_proj.kernel_size[1]
            torch.nn.init.trunc_normal_(self.conv_proj.weight, std=math.sqrt(1 / fan_in))
            if self.conv_proj.bias:
                torch.nn.init.zeros_(self.conv_proj.bias)
        elif self.conv_proj.conv_last and isinstance(self.conv_proj.conv_last, nn.Conv2d):
            # Init the last 1x1 conv of the conv stem
            torch.nn.init.normal_(
                self.conv_proj.conv_last.weight, mean=0.0, std=math.sqrt(2.0 / self.conv_proj.conv_last.out_channels)
            )
            if self.conv_proj.conv_last.bias:
                torch.nn.init.zeros_(self.conv_proj.conv_last.bias)
        
        if hasattr(self.heads, "pre_logits") and isinstance(self.heads.pre_logits, nn.Linear):
            fan_in = self.heads.pre_logits.in_features
            torch.nn.init.trunc_normal_(self.heads.pre_logits.weight, std=math.sqrt(1 / fan_in))
            torch.nn.init.zeros_(self.heads.pre_logits.bias)

        if isinstance(self.heads.head, nn.Linear):
            torch.nn.init.zeros_(self.heads.head.weight)
            torch.nn.init.zeros_(self.heads.head.bias)
        
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)

    def init_weights(self, model: nn.Module):
        pass
    
    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        p = self.patch_size
        assert h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!"
        assert w == self.image_size, f"Wrong image width!  Expected {self.image_size} but got {w}!"
        n_h = h // p
        n_w = w // p
        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.conv_proj(x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.hidden_dim, n_h * n_w)
        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension.
        x = x.permute(0, 2, 1)
        return x
    
    def forward(self, datapoint: dict, *args, **kwargs) -> dict:
        self.assert_datapoint(datapoint)
        x = datapoint.get("image")
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]
        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = self.encoder(x)
        # Classifier "token" as used by standard language architectures
        x = x[:, 0]
        y = self.heads(x)
        return {"logits": y}


ViT = VisionTransformer


@MODELS.register(name="vit_b_16", arch="vit")
class ViT_B_16(VisionTransformer):
    """vit_b_16 architecture from `An Image is Worth 16x16 Words: Transformers
    for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`__.
    """
    
    zoo: dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/vit_b_16-c867db91.pth",
            "path"       : "vit/vit_b_16/imagenet1k_v1/vit_b_16_imagenet1k_v1.pth",
            "num_classes": 1000,
            "map": {},
        },
        "imagenet1k_swag_e2e_v1": {
            "url"        : "https://download.pytorch.org/models/vit_b_16_swag-9ac1b537.pth",
            "path"       : "vit/vit_b_16_swag/imagenet1k_v1/vit_b_16_swag_imagenet1k_v1.pth",
            "num_classes": 1000,
            "map": {},
        },
        "imagenet1k_swag_linear_v1": {
            "url"        : "https://download.pytorch.org/models/vit_b_16_lc_swag-4e70ced5.pth",
            "path"       : "vit/vit_b_16_lc_swag/imagenet1k_v1/vit_b_16_lc_swag_imagenet1k_v1.pth",
            "num_classes": 1000,
            "map": {},
        },
    }
    
    def __init__(self, args, **kwargs):
        image_size = kwargs.pop("image_size", 224)
        super().__init__(
            name       = "vit_b_16",
            image_size = image_size,
            patch_size = 16,
            num_layers = 12,
            num_heads  = 12,
            hidden_dim = 768,
            mlp_dim    = 3072,
            *args, **kwargs
        )


@MODELS.register(name="vit_b_32", arch="vit")
class ViT_B_32(VisionTransformer):
    """vit_b_32 architecture from `An Image is Worth 16x16 Words: Transformers
    for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`__.
    """
    
    zoo: dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/vit_b_32-d86f8d99.pth",
            "path"       : "vit/vit_b_32/imagenet1k_v1/vit_b_32_imagenet1k_v1.pth",
            "num_classes": 1000,
            "map": {},
        },
    }
    
    def __init__(self, args, **kwargs):
        image_size = kwargs.pop("image_size", 224)
        super().__init__(
            name       = "vit_b_32",
            image_size = image_size,
            patch_size = 32,
            num_layers = 12,
            num_heads  = 12,
            hidden_dim = 768,
            mlp_dim    = 3072,
            *args, **kwargs
        )


@MODELS.register(name="vit_l_16", arch="vit")
class ViT_L_16(VisionTransformer):
    """vit_l_16 architecture from `An Image is Worth 16x16 Words: Transformers
    for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`__.
    """
    
    zoo: dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/vit_l_16-852ce7e3.pth",
            "path"       : "vit/vit_l_16/imagenet1k_v1/vit_l_16_imagenet1k_v1.pth",
            "num_classes": 1000,
            "map": {},
        },
        "imagenet1k_swag_e2e_v1": {
            "url"        : "https://download.pytorch.org/models/vit_l_16_swag-4f3808c9.pth",
            "path"       : "vit/vit_l_16_swag/imagenet1k_v1/vit_l_16_swag_imagenet1k_v1.pth",
            "num_classes": 1000,
            "map": {},
        },
        "imagenet1k_swag_linear_v1": {
            "url"        : "https://download.pytorch.org/models/vit_l_16_lc_swag-4d563306.pth",
            "path"       : "vit/vit_l_16_lc_swag/imagenet1k_v1/vit_l_16_lc_swag_imagenet1k_v1.pth",
            "num_classes": 1000,
            "map": {},
        },
    }
    
    def __init__(self, args, **kwargs):
        image_size = kwargs.pop("image_size", 224)
        super().__init__(
            name       = "vit_l_16",
            image_size = image_size,
            patch_size = 16,
            num_layers = 24,
            num_heads  = 16,
            hidden_dim = 1024,
            mlp_dim    = 4096,
            *args, **kwargs
        )


@MODELS.register(name="vit_l_32", arch="vit")
class ViT_L_32(VisionTransformer):
    """vit_l_32 architecture from `An Image is Worth 16x16 Words: Transformers
    for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`__.
    """
    
    zoo: dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/vit_l_32-c7638314.pth",
            "path"       : "vit/vit_l_32/imagenet1k_v1/vit_l_32_imagenet1k_v1.pth",
            "num_classes": 1000,
            "map": {},
        },
    }
    
    def __init__(self, args, **kwargs):
        image_size = kwargs.pop("image_size", 224)
        super().__init__(
            name       = "vit_l_32",
            image_size = image_size,
            patch_size = 32,
            num_layers = 24,
            num_heads  = 16,
            hidden_dim = 1024,
            mlp_dim    = 4096,
            *args, **kwargs
        )
      

@MODELS.register(name="vit_h_14", arch="vit")
class ViT_H_14(VisionTransformer):
    """vit_h_14 architecture from `An Image is Worth 16x16 Words: Transformers
    for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`__.
    """
    
    zoo: dict = {
        "imagenet1k_swag_e2e_v1": {
            "url"        : "https://download.pytorch.org/models/vit_h_14_swag-80465313.pth",
            "path"       : "vit/vit_h_14_swag/imagenet1k_v1/vit_h_14_swag_imagenet1k_v1.pth",
            "num_classes": 1000,
            "map": {},
        },
        "imagenet1k_swag_linear_v1": {
            "url"        : "https://download.pytorch.org/models/vit_h_14_lc_swag-c1eb923e.pth",
            "path"       : "vit/vit_h_14_lc_swag/imagenet1k_v1/vit_h_14_lc_swag_imagenet1k_v1.pth",
            "num_classes": 1000,
            "map": {},
        },
    }
    
    def __init__(self, args, **kwargs):
        image_size = kwargs.pop("image_size", 224)
        super().__init__(
            name       = "vit_h_14",
            image_size = image_size,
            patch_size = 14,
            num_layers = 32,
            num_heads  = 16,
            hidden_dim = 1280,
            mlp_dim    = 5120,
            *args, **kwargs
        )
        
# endregion

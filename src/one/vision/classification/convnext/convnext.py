#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""ConvNeXt models.
"""

from __future__ import annotations

from functools import partial
from typing import Optional
from typing import Union

import torch
import torch.nn.functional as F
from torch import nn
from torch import Size
from torch import Tensor
from torch.nn.init import trunc_normal_

from one.core import BACKBONES
from one.core import Indexes
from one.core import ListOrTupleAnyT
from one.core import MODELS
from one.core import Pretrained
from one.core import Tensors
from one.nn import DropPath
from one.vision.classification.image_classifier import ImageClassifier

__all__ = [
    "ConvNeXt",
    "ConvNeXtTiny",
    "ConvNeXtSmall",
    "ConvNeXtBase",
    "ConvNeXtLarge",
    "ConvNeXtXLarge",
]


# MARK: - Modules

class LayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or
    channels_first. The ordering of the dimensions in the inputs. channels_last
    corresponds to inputs with shape (batch_size, height, width, channels)
    while channels_first corresponds to inputs with shape
    (batch_size, channels, height, width).
    """
    
    def __init__(
        self,
        normalized_shape: Union[int, list[int], Size],
        eps             : float = 1e-6,
        data_format     : str   = "channels_last"
    ):
        super().__init__()
        self.weight      = nn.Parameter(torch.ones(normalized_shape))
        self.bias        = nn.Parameter(torch.zeros(normalized_shape))
        self.eps         = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x: Tensor):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class Block(nn.Module):
    r"""ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch.
    
    Args:
        dim (int):
            Number of input channels.
        drop_path (float):
            Stochastic depth rate. Default: `0.0`.
        layer_scale_init_value (float):
            Init value for Layer Scale. Default: `1e-6`.
    """
    
    def __init__(
        self,
        dim                   : int,
        drop_path             : float = 0.0,
        layer_scale_init_value: float = 1e-6
    ):
        super().__init__()
        # depthwise conv
        self.dw_conv  = nn.Conv2d(
            dim, dim, kernel_size=(7, 7), padding=3, groups=dim
        )
        self.norm     = LayerNorm(dim, eps=1e-6)
        # pointwise/1x1 convs, implemented with linear layers
        self.pw_conv1 = nn.Linear(dim, 4 * dim)
        self.act      = nn.GELU()
        self.pw_conv2 = nn.Linear(4 * dim, dim)
        self.gamma    = nn.Parameter(
            layer_scale_init_value * torch.ones(dim),
            requires_grad=True
        ) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.dw_conv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pw_conv1(x)
        x = self.act(x)
        x = self.pw_conv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = x + self.drop_path(x)
        return x


# MARK: - ConvNeXt

cfgs = {
    "convnext_tiny": {
        "in_channels"           : 3,
        "out_channels"          : [96, 192, 384, 768],
        "depths"                : [3, 3, 9, 3],
        "drop_path_rate"        : 0.1,
        "layer_scale_init_value": 1e-6,
        "head_init_scale"       : 1.0,
    },
    "convnext_small": {
        "in_channels"           : 3,
        "out_channels"          : [96, 192, 384, 768],
        "depths"                : [3, 3, 27, 3],
        "drop_path_rate"        : 0.0,
        "layer_scale_init_value": 1e-6,
        "head_init_scale"       : 1.0,
    },
    "convnext_base": {
        "in_channels"           : 3,
        "out_channels"          : [128, 256, 512, 1024],
        "depths"                : [3, 3, 27, 3],
        "drop_path_rate"        : 0.0,
        "layer_scale_init_value": 1e-6,
        "head_init_scale"       : 1.0,
    },
    "convnext_large": {
        "in_channels"           : 3,
        "out_channels"          : [192, 384, 768, 1536],
        "depths"                : [3, 3, 27, 3],
        "drop_path_rate"        : 0.0,
        "layer_scale_init_value": 1e-6,
        "head_init_scale"       : 1.0,
    },
    "convnext_xlarge": {
        "in_channels"           : 3,
        "out_channels"          : [256, 512, 1024, 2048],
        "depths"                : [3, 3, 27, 3],
        "drop_path_rate"        : 0.0,
        "layer_scale_init_value": 1e-6,
        "head_init_scale"       : 1.0,
    },
}


@MODELS.register(name="convnext")
@BACKBONES.register(name="convnext")
class ConvNeXt(ImageClassifier):
    """ConvNeXt backbone.
    
    Args:
        basename (str, optional):
            Model basename. Default: `convnext`.
        name (str, optional):
            Model name. Default: `convnext`.
        num_classes (int, optional):
            Number of classes for classification. Default: `None`.
        out_indexes (Indexes):
            List of output tensors taken from specific layers' indexes.
            If `>= 0`, return the ith layer's output.
            If `-1`, return the final layer's output. Default: `-1`.
        pretrained (Pretrained):
            Use pretrained weights. If `True`, returns a model pre-trained on
            ImageNet. If `str`, load weights from saved file. Default: `True`.
            - If `True`, returns a model pre-trained on ImageNet.
            - If `str` and is a weight file(path), then load weights from
              saved file.
            - In each inherited model, `pretrained` can be a dictionary's
              key to get the corresponding local file or url of the weight.
    """
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        # Hyperparameters
        in_channels           : int                  = 3,
        out_channels          : ListOrTupleAnyT[int] = (96, 192, 384, 768),
        depths                : ListOrTupleAnyT[int] = (3, 3, 9, 3),
        drop_path_rate        : float                = 0.1,
        layer_scale_init_value: float                = 1e-6,
        head_init_scale       : Optional[float]      = 1.0,
        # BaseModel's args
        basename              : Optional[str]   = "convnext",
        name                  : Optional[str]   = "convnext",
        num_classes           : Optional[int]   = None,
        out_indexes           : Indexes         = -1,
        pretrained            : Pretrained      = False,
        *args, **kwargs
    ):
        super().__init__(
            basename    = basename,
            name        = name,
            num_classes = num_classes,
            out_indexes = out_indexes,
            pretrained  = pretrained,
            *args, **kwargs
        )
        # NOTE: Get Hyperparameters
        self.in_channels            = in_channels
        self.out_channels           = out_channels
        self.depths                 = depths
        self.drop_path_rate         = drop_path_rate
        self.layer_scale_init_value = layer_scale_init_value
        self.head_init_scale        = head_init_scale
        
        # Stem and 3 intermediate downsampling conv layers
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels[0], (4, 4), (4, 4)),
            LayerNorm(self.out_channels[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(self.out_channels[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(self.out_channels[i], self.out_channels[i + 1], (2, 2), (2, 2)),
            )
            self.downsample_layers.append(downsample_layer)

        # 4 feature resolution stages, each consisting of multiple residual blocks
        self.stages = nn.ModuleList()
        dp_rates    = [
            x.item() for x in torch.linspace(0, self.drop_path_rate, sum(self.depths))
        ]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=self.out_channels[i], drop_path=dp_rates[cur + j],
                        layer_scale_init_value=self.layer_scale_init_value)
                  for j in range(self.depths[i])]
            )
            self.stages.append(stage)
            cur += self.depths[i]
        
        # Norm layer of forward features
        norm_layer = partial(LayerNorm, eps=1e-6, data_format="channels_first")
        for i_layer in range(4):
            layer      = norm_layer(self.out_channels[i_layer])
            layer_name = f"norm{i_layer}"
            self.add_module(layer_name, layer)
        
        # NOTE: Classifier
        self.norm = nn.LayerNorm(self.out_channels[-1], eps=1e-6)
        self.head = self.create_classifier(self.out_channels[-1], self.num_classes)
        
        # NOTE: Load Pretrained
        if self.pretrained:
            self.load_pretrained()
        else:
            self.apply(self.init_weights)
            if not isinstance(self.head, nn.Identity):
                self.head.weight.data.mul_(self.head_init_scale)
                self.head.bias.data.mul_(self.head_init_scale)
        
        # NOTE: Alias
        self.classifier = self.head
    
    # MARK: Configure

    @staticmethod
    def create_classifier(
        num_features: int, num_classes: Optional[int]
    ) -> nn.Module:
        if num_classes and num_classes > 0:
            classifier = nn.Linear(num_features, num_classes)
        else:
            classifier = nn.Identity()
        return classifier
    
    def init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)
    
    # MARK: Forward Pass

    def forward_once(self, x: Tensor, *args, **kwargs) -> Tensor:
        """Forward pass once. Implement the logic for a single forward pass.

		Args:
			x (Tensor):
				Input of shape [B, C, H, W].

		Returns:
			yhat (Tensor):
				Predictions.
		"""
        x = self.forward_features(x)
        if isinstance(x, Tensor):
            # global average pooling, (N, C, H, W) -> (N, C)
            x = self.norm(x.mean([-2, -1]))
            x = self.classifier(x)
        return x

    def forward_features(
        self, x: Tensor, out_indexes: Optional[Indexes] = None
    ) -> Tensors:
        """Forward pass for features extraction.

        Args:
            x (Tensor):
                Input image.
            out_indexes (Indexes, optional):
                List of layers' indexes to extract features. This is called
                in `forward_features()` and is useful when the model
                is used as a component in another model.
                - If is a `tuple` or `list`, return an array of features.
                - If is a `int`, return only the feature from that layer's
                  index.
                - If is `-1`, return the last layer's output.
                Default: `None`.
        """
        out_indexes = self.out_indexes if out_indexes is None else out_indexes
    
        yhat = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if isinstance(out_indexes, (tuple, list)) and (i in out_indexes):
                norm_layer = getattr(self, f"norm{i}")
                output     = norm_layer(x)
                yhat.append(output)
            elif isinstance(out_indexes, int) and (i == out_indexes):
                norm_layer = getattr(self, f"norm{i}")
                output     = norm_layer(x)
                return output
            elif out_indexes is None or out_indexes == -1:
                yhat = x
        return yhat


# MARK: - ConvNeXtTiny

@MODELS.register(name="convnext_t")
@MODELS.register(name="convnext_tiny")
@BACKBONES.register(name="convnext_t")
@BACKBONES.register(name="convnext_tiny")
class ConvNeXtTiny(ConvNeXt):

    model_zoo = {
        "imagenet_1k": dict(
            path="https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
            file_name="convnext_tiny_imagenet_1k.pth", num_classes=1000,
        ),
    }
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        # BaseModel's args
        name       : Optional[str] = "convnext_tiny",
        out_indexes: Indexes       = -1,
        num_classes: Optional[int] = None,
        pretrained : Pretrained    = False,
        *args, **kwargs
    ):
        kwargs = cfgs["convnext_tiny"] | kwargs
        super().__init__(
            name        = name,
            out_indexes = out_indexes,
            num_classes = num_classes,
            pretrained  = pretrained,
            *args, **kwargs
        )


# MARK: - ConvNeXtSmall

@MODELS.register(name="convnext_s")
@MODELS.register(name="convnext_small")
@BACKBONES.register(name="convnext_s")
@BACKBONES.register(name="convnext_small")
class ConvNeXtSmall(ConvNeXt):

    model_zoo = {
        "imagenet_1k": dict(
            path="https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
            file_name="convnext_small_imagenet_1k.pth", num_classes=1000,
        ),
    }
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        # BaseModel's args
        name       : Optional[str] = "convnext_small",
        out_indexes: Indexes       = -1,
        num_classes: Optional[int] = None,
        pretrained : Pretrained    = False,
        *args, **kwargs
    ):
        kwargs = cfgs["convnext_small"] | kwargs
        super().__init__(
            name        = name,
            out_indexes = out_indexes,
            num_classes = num_classes,
            pretrained  = pretrained,
            *args, **kwargs
        )


# MARK: - ConvNeXtBase

@MODELS.register(name="convnext_b")
@MODELS.register(name="convnext_base")
@BACKBONES.register(name="convnext_b")
@BACKBONES.register(name="convnext_base")
class ConvNeXtBase(ConvNeXt):

    model_zoo = {
        "imagenet_1k": dict(
            path="https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
            file_name="convnext_base_imagenet_1k.pth", num_classes=1000,
        ),
        "imagenet_22k": dict(
            path="https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
            file_name="convnext_base_imagenet_22k.pth", num_classes=22000,
        ),
    }
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        # BaseModel's args
        name       : Optional[str] = "convnext_base",
        out_indexes: Indexes       = -1,
        num_classes: Optional[int] = None,
        pretrained : Pretrained    = False,
        *args, **kwargs
    ):
        kwargs = cfgs["convnext_base"] | kwargs
        super().__init__(
            name        = name,
            out_indexes = out_indexes,
            num_classes = num_classes,
            pretrained  = pretrained,
            *args, **kwargs
        )


# MARK: - ConvNeXtLarge

@MODELS.register(name="convnext_l")
@MODELS.register(name="convnext_large")
@BACKBONES.register(name="convnext_l")
@BACKBONES.register(name="convnext_large")
class ConvNeXtLarge(ConvNeXt):

    model_zoo = {
        "imagenet_1k": dict(
            path="https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
            file_name="convnext_large_imagenet_1k.pth", num_classes=1000,
        ),
        "imagenet_22k": dict(
            path="https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth",
            file_name="convnext_large_imagenet_22k.pth", num_classes=22000,
        ),
    }
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        # BaseModel's args
        name       : Optional[str] = "convnext_large",
        out_indexes: Indexes       = -1,
        num_classes: Optional[int] = None,
        pretrained : Pretrained    = False,
        *args, **kwargs
    ):
        kwargs = cfgs["convnext_large"] | kwargs
        super().__init__(
            name        = name,
            out_indexes = out_indexes,
            num_classes = num_classes,
            pretrained  = pretrained,
            *args, **kwargs
        )


# MARK: - ConvNeXtXLarge

@MODELS.register(name="convnext_xl")
@MODELS.register(name="convnext_xlarge")
@BACKBONES.register(name="convnext_xl")
@BACKBONES.register(name="convnext_xlarge")
class ConvNeXtXLarge(ConvNeXt):

    model_zoo = {
        "imagenet_22k": dict(
            path="https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth",
            file_name="convnext_xlarge_imagenet_22k.pth", num_classes=22000,
        ),
    }
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        # BaseModel's args
        name       : Optional[str] = "convnext_xlarge",
        out_indexes: Indexes       = -1,
        num_classes: Optional[int] = None,
        pretrained : Pretrained    = False,
        *args, **kwargs
    ):
        kwargs = cfgs["convnext_large"] | kwargs
        super().__init__(
            name        = name,
            out_indexes = out_indexes,
            num_classes = num_classes,
            pretrained  = pretrained,
            *args, **kwargs
        )

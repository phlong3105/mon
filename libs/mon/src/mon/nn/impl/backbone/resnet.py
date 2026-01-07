#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""ResNet backbones.

This module implements various ResNet backbones using PyTorch.
"""

__all__ = [
    "ResNeXt101_32X8D_Weights",
    "ResNeXt101_64X4D_Weights",
    "ResNeXt50_32X4D_Weights",
    "ResNet101_Weights",
    "ResNet152_Weights",
    "ResNet18_Weights",
    "ResNet34_Weights",
    "ResNet50_Weights",
    "Wide_ResNet101_2_Weights",
    "Wide_ResNet50_2_Weights",
    "resnet101",
    "resnet152",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnext101_32x8d",
    "resnext101_64x4d",
    "resnext50_32x4d",
    "wide_resnet101_2",
    "wide_resnet50_2",
]

from typing import Union

import torch
import torch.nn as nn
from torchvision.models._meta import _IMAGENET_CATEGORIES
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet

from mon.core import BACKBONES, Path, ROOT_DIR
from mon.core.dtypes import Weights, WeightsEnum

current_file = Path(__file__).absolute()
root_dir     = current_file.parents[0]


# ==============================================================================
# COMPONENTS (Building Blocks)
# ==============================================================================


# ==============================================================================
# BASE CLASSES & MIXINS (Behaviors)
# ==============================================================================

# --- Structural Bases ---
class ResNetBackBone(nn.Module):
    """ResNet backbone.

    References:
        - Paper: https://arxiv.org/abs/1512.03385
    """
    
    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        variant    : str,
        block      : type[Union[BasicBlock, Bottleneck]],
        layers     : list[int],
        weights    : WeightsEnum = None,
        out_indices: list        = None,
        *args, **kwargs
    ):
        """Initialize a new instance.
        
        Args:
            variant: Variant of the ResNet model.
            weights: Pre-trained weights to load.
            out_indices: List of layer indices to extract features from.
                If None, defaults to [4, 5, 6, 7].
            args: Additional positional arguments for the ResNet model.
            kwargs: Additional keyword arguments for the ResNet model
        """
        super().__init__()
        # Load the base model
        if weights is not None:
            kwargs["num_classes"] = weights.num_classes
        
        base_model = ResNet(block=block, layers=layers, *args, **kwargs)
        
        if weights is not None:
            base_model.load_state_dict(weights.get_state_dict())
        
        # Remove the global average pool and classifier head
        # For ResNet, we usually want the features before the final layers
        self.features     = nn.Sequential(*list(base_model.children())[:-2])
        self.out_indices  = out_indices or [4, 5, 6, 7]
        self.out_channels = self._get_out_channels(variant)
    
    def _get_out_channels(self, variant):
        # Bottleneck models (50+) have 4x more channels in the output of each stage
        if any(x in variant for x in ["50", "101", "152"]):
            return [256, 512, 1024, 2048]
        return [64, 128, 256, 512]
    
    # --- Callable & Context Manager ---
    def forward(self, x: torch.Tensor) -> list:
        """Forward pass.
        
        Args:
            x: Input tensor with dimensions (B, C, H, W) and values ranging
                from 0.0 to 1.0.
        
        Returns:
            A list of feature maps from the specified layers.
        """
        # If you need multi-scale features for a Neck (FPN):
        outputs = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.out_indices:
                outputs.append(x)
        return outputs
    

# ==============================================================================
# CONCRETE IMPLEMENTATIONS (Variants)
# ==============================================================================

# --- Pre-trained Weights ---
class ResNet18_Weights(WeightsEnum):
    
    IMAGENET1K_V1 = Weights(
        url         = "https://download.pytorch.org/models/resnet18-f37072fd.pth",
        path        = ROOT_DIR / "zoo/nn/backbone/resnet/resnet18/imagenet1k_v1/resnet18_imagenet1k_v1.pth",
        num_classes = 1000,
        transforms  = None,
        meta        = {
            "num_params": 11689512,
            "min_size"  : (1, 1),
            "categories": _IMAGENET_CATEGORIES,
            "recipe"    : "https://github.com/pytorch/vision/tree/main/references/classification#resnet",
            "_metrics"  : {
                "ImageNet-1K": {
                    "acc@1": 69.758,
                    "acc@5": 89.078,
                }
            },
            "_ops"      : 1.814,
            "_file_size": 44.661,
            "_docs"     : """These weights reproduce closely the results of the paper using a simple training recipe.""",
        }
    )
    DEFAULT = IMAGENET1K_V1


class ResNet34_Weights(WeightsEnum):
    
    IMAGENET1K_V1 = Weights(
        url         = "https://download.pytorch.org/models/resnet34-b627a593.pth",
        path        = ROOT_DIR / "zoo/nn/backbone/resnet/resnet34/imagenet1k_v1/resnet34_imagenet1k_v1.pth",
        num_classes = 1000,
        transforms  = None,
        meta        = {
            "num_params": 21797672,
            "min_size"  : (1, 1),
            "categories": _IMAGENET_CATEGORIES,
            "recipe"    : "https://github.com/pytorch/vision/tree/main/references/classification#resnet",
            "_metrics"  : {
                "ImageNet-1K": {
                    "acc@1": 73.314,
                    "acc@5": 91.420,
                }
            },
            "_ops"      : 3.664,
            "_file_size": 83.275,
            "_docs"     : """These weights reproduce closely the results of the paper using a simple training recipe.""",
        }
    )
    DEFAULT = IMAGENET1K_V1


class ResNet50_Weights(WeightsEnum):
    
    IMAGENET1K_V1 = Weights(
        url         = "https://download.pytorch.org/models/resnet50-0676ba61.pth",
        path        = ROOT_DIR / "zoo/nn/backbone/resnet/resnet50/imagenet1k_v1/resnet50_imagenet1k_v1.pth",
        num_classes = 1000,
        transforms  = None,
        meta        = {
            "num_params": 25557032,
            "min_size"  : (1, 1),
            "categories": _IMAGENET_CATEGORIES,
            "recipe"    : "https://github.com/pytorch/vision/tree/main/references/classification#resnet",
            "_metrics"  : {
                "ImageNet-1K": {
                    "acc@1": 76.130,
                    "acc@5": 92.862,
                }
            },
            "_ops"      : 4.089,
            "_file_size": 97.781,
            "_docs"     : """These weights reproduce closely the results of the paper using a simple training recipe.""",
        }
    )
    IMAGENET1K_V2 = Weights(
        url         = "https://download.pytorch.org/models/resnet50-11ad3fa6.pth",
        path        = ROOT_DIR / "zoo/nn/backbone/resnet/resnet50/imagenet1k_v2/resnet50_imagenet1k_v2.pth",
        num_classes = 1000,
        transforms  = None,
        meta        = {
            "num_params": 25557032,
            "min_size"  : (1, 1),
            "categories": _IMAGENET_CATEGORIES,
            "recipe"    : "https://github.com/pytorch/vision/issues/3995#issuecomment-1013906621",
            "_metrics"  : {
                "ImageNet-1K": {
                    "acc@1": 80.858,
                    "acc@5": 95.434,
                }
            },
            "_ops"      : 4.089,
            "_file_size": 97.79,
            "_docs"     : """
                These weights improve upon the results of the original paper by using TorchVision's `new training recipe
                <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
            """,
        }
    )
    DEFAULT = IMAGENET1K_V2


class ResNet101_Weights(WeightsEnum):
    
    IMAGENET1K_V1 = Weights(
        url         = "https://download.pytorch.org/models/resnet101-63fe2227.pth",
        path        = ROOT_DIR / "zoo/nn/backbone/resnet/resnet101/imagenet1k_v1/resnet101_imagenet1k_v1.pth",
        num_classes = 1000,
        transforms  = None,
        meta        = {
            "num_params": 44549160,
            "min_size"  : (1, 1),
            "categories": _IMAGENET_CATEGORIES,
            "recipe"    : "https://github.com/pytorch/vision/tree/main/references/classification#resnet",
            "_metrics"  : {
                "ImageNet-1K": {
                    "acc@1": 77.374,
                    "acc@5": 93.546,
                }
            },
            "_ops"      : 7.801,
            "_file_size": 170.511,
            "_docs"     : """These weights reproduce closely the results of the paper using a simple training recipe.""",
        }
    )
    IMAGENET1K_V2 = Weights(
        url         = "https://download.pytorch.org/models/resnet101-cd907fc2.pth",
        path        = ROOT_DIR / "zoo/nn/backbone/resnet/resnet101/imagenet1k_v2/resnet101_imagenet1k_v2.pth",
        num_classes = 1000,
        transforms  = None,
        meta        = {
            "num_params": 44549160,
            "min_size"  : (1, 1),
            "categories": _IMAGENET_CATEGORIES,
            "recipe"    : "https://github.com/pytorch/vision/issues/3995#new-recipe",
            "_metrics"  : {
                "ImageNet-1K": {
                    "acc@1": 81.886,
                    "acc@5": 95.780,
                }
            },
            "_ops"      : 7.801,
            "_file_size": 170.53,
            "_docs"     : """
                These weights improve upon the results of the original paper by using TorchVision's `new training recipe
                <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
            """,
        }
    )
    DEFAULT = IMAGENET1K_V2
    

class ResNet152_Weights(WeightsEnum):
    
    IMAGENET1K_V1 = Weights(
        url         = "https://download.pytorch.org/models/resnet152-394f9c45.pth",
        path        = ROOT_DIR / "zoo/nn/backbone/resnet/resnet152/imagenet1k_v1/resnet152_imagenet1k_v1.pth",
        num_classes = 1000,
        transforms  = None,
        meta        = {
            "num_params": 60192808,
            "min_size"  : (1, 1),
            "categories": _IMAGENET_CATEGORIES,
            "recipe"    : "https://github.com/pytorch/vision/tree/main/references/classification#resnet",
            "_metrics"  : {
                "ImageNet-1K": {
                    "acc@1": 78.312,
                    "acc@5": 94.046,
                }
            },
            "_ops"      : 11.514,
            "_file_size": 230.434,
            "_docs"     : """These weights reproduce closely the results of the paper using a simple training recipe.""",
        }
    )
    IMAGENET1K_V2 = Weights(
        url         = "https://download.pytorch.org/models/resnet152-f82ba261.pth",
        path        = ROOT_DIR / "zoo/nn/backbone/resnet/resnet152/imagenet1k_v2/resnet152_imagenet1k_v2.pth",
        num_classes = 1000,
        transforms  = None,
        meta        = {
            "num_params": 60192808,
            "min_size"  : (1, 1),
            "categories": _IMAGENET_CATEGORIES,
            "recipe"    : "https://github.com/pytorch/vision/issues/3995#new-recipe",
            "_metrics"  : {
                "ImageNet-1K": {
                    "acc@1": 82.284,
                    "acc@5": 96.002,
                }
            },
            "_ops"      : 11.514,
            "_file_size": 230.474,
            "_docs": """
                These weights improve upon the results of the original paper by using TorchVision's `new training recipe
                <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
            """,
        }
    )
    DEFAULT = IMAGENET1K_V2


class ResNeXt50_32X4D_Weights(WeightsEnum):
    
    IMAGENET1K_V1 = Weights(
        url         = "https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
        path        = ROOT_DIR / "zoo/nn/backbone/resnet/resnext50_32x4d/imagenet1k_v1/resnext50_32x4d_imagenet1k_v1.pth",
        num_classes = 1000,
        transforms  = None,
        meta        = {
            "num_params": 25028904,
            "min_size"  : (1, 1),
            "categories": _IMAGENET_CATEGORIES,
            "recipe"    : "https://github.com/pytorch/vision/tree/main/references/classification#resnext",
            "_metrics"  : {
                "ImageNet-1K": {
                    "acc@1": 77.618,
                    "acc@5": 93.698,
                }
            },
            "_ops"      : 4.23,
            "_file_size": 95.789,
            "_docs"     : """These weights reproduce closely the results of the paper using a simple training recipe.""",
        }
    )
    IMAGENET1K_V2 = Weights(
        url         = "https://download.pytorch.org/models/resnext50_32x4d-1a0047aa.pth",
        path        = ROOT_DIR / "zoo/nn/backbone/resnet/resnext50_32x4d/imagenet1k_v2/resnext50_32x4d_imagenet1k_v2.pth",
        num_classes = 1000,
        transforms  = None,
        meta        = {
            "num_params": 25028904,
            "min_size"  : (1, 1),
            "categories": _IMAGENET_CATEGORIES,
            "recipe"    : "https://github.com/pytorch/vision/issues/3995#new-recipe",
            "_metrics"  : {
                "ImageNet-1K": {
                    "acc@1": 81.198,
                    "acc@5": 95.340,
                }
            },
            "_ops"      : 4.23,
            "_file_size": 95.833,
            "_docs": """
                These weights improve upon the results of the original paper by using TorchVision's `new training recipe
                <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
            """,
        }
    )
    DEFAULT = IMAGENET1K_V2


class ResNeXt101_32X8D_Weights(WeightsEnum):
    
    IMAGENET1K_V1 = Weights(
        url         = "https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth",
        path        = ROOT_DIR / "zoo/nn/backbone/resnet/resnext101_32x8d/imagenet1k_v1/resnext101_32x8d_imagenet1k_v1.pth",
        num_classes = 1000,
        transforms  = None,
        meta        = {
            "num_params": 88791336,
            "min_size"  : (1, 1),
            "categories": _IMAGENET_CATEGORIES,
            "recipe"    : "https://github.com/pytorch/vision/tree/main/references/classification#resnext",
            "_metrics"  : {
                "ImageNet-1K": {
                    "acc@1": 79.312,
                    "acc@5": 94.526,
                }
            },
            "_ops"      : 16.414,
            "_file_size": 339.586,
            "_docs"     : """These weights reproduce closely the results of the paper using a simple training recipe.""",
        }
    )
    IMAGENET1K_V2 = Weights(
        url         = "https://download.pytorch.org/models/resnext101_32x8d-110c445d.pth",
        path        = ROOT_DIR / "zoo/nn/backbone/resnet/resnext101_32x8d/imagenet1k_v2/resnext101_32x8d_imagenet1k_v2.pth",
        num_classes = 1000,
        transforms  = None,
        meta        = {
            "num_params": 88791336,
            "min_size"  : (1, 1),
            "categories": _IMAGENET_CATEGORIES,
            "recipe"    : "https://github.com/pytorch/vision/issues/3995#new-recipe-with-fixres",
            "_metrics"  : {
                "ImageNet-1K": {
                    "acc@1": 82.834,
                    "acc@5": 96.228,
                }
            },
            "_ops"      : 16.414,
            "_file_size": 339.673,
            "_docs": """
                These weights improve upon the results of the original paper by using TorchVision's `new training recipe
                <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
            """,
        }
    )
    DEFAULT = IMAGENET1K_V2
    

class ResNeXt101_64X4D_Weights(WeightsEnum):
    
    IMAGENET1K_V1 = Weights(
        url         = "ttps://download.pytorch.org/models/resnext101_64x4d-173b62eb.pth",
        path        = ROOT_DIR / "zoo/nn/backbone/resnet/resnext101_64x4d/imagenet1k_v1/resnext101_64x4d_imagenet1k_v1.pth",
        num_classes = 1000,
        transforms  = None,
        meta        = {
            "num_params": 83455272,
            "min_size"  : (1, 1),
            "categories": _IMAGENET_CATEGORIES,
            "recipe"    : "https://github.com/pytorch/vision/pull/5935",
            "_metrics"  : {
                "ImageNet-1K": {
                    "acc@1": 83.246,
                    "acc@5": 96.454,
                }
            },
            "_ops"      : 15.46,
            "_file_size": 319.318,
            "_docs": """
                These weights were trained from scratch by using TorchVision's `new training recipe
                <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
            """,
        }
    )
    DEFAULT = IMAGENET1K_V1


class Wide_ResNet50_2_Weights(WeightsEnum):
    
    IMAGENET1K_V1 = Weights(
        url         = "https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth",
        path        = ROOT_DIR / "zoo/nn/backbone/resnet/wide_resnet50_2/imagenet1k_v1/wide_resnet50_2_imagenet1k_v1.pth",
        num_classes = 1000,
        transforms  = None,
        meta        = {
            "num_params": 68883240,
            "min_size"  : (1, 1),
            "categories": _IMAGENET_CATEGORIES,
            "recipe"    : "https://github.com/pytorch/vision/pull/912#issue-445437439",
            "_metrics"  : {
                "ImageNet-1K": {
                    "acc@1": 78.468,
                    "acc@5": 94.086,
                }
            },
            "_ops"      : 11.398,
            "_file_size": 131.82,
            "_docs"     : """These weights reproduce closely the results of the paper using a simple training recipe.""",
        }
    )
    IMAGENET1K_V2 = Weights(
        url         = "https://download.pytorch.org/models/wide_resnet50_2-9ba9bcbe.pth",
        path        = ROOT_DIR / "zoo/nn/backbone/resnet/wide_resnet50_2/imagenet1k_v2/wide_resnet50_2_imagenet1k_v2.pth",
        num_classes = 1000,
        transforms  = None,
        meta        = {
            "num_params": 68883240,
            "min_size"  : (1, 1),
            "categories": _IMAGENET_CATEGORIES,
            "recipe"    : "https://github.com/pytorch/vision/issues/3995#new-recipe-with-fixres",
            "_metrics"  : {
                "ImageNet-1K": {
                    "acc@1": 81.602,
                    "acc@5": 95.758,
                }
            },
            "_ops"      : 11.398,
            "_file_size": 263.124,
            "_docs": """
                These weights improve upon the results of the original paper by using TorchVision's `new training recipe
                <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
            """,
        }
    )
    DEFAULT = IMAGENET1K_V2


class Wide_ResNet101_2_Weights(WeightsEnum):
    
    IMAGENET1K_V1 = Weights(
        url         = "https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth",
        path        = ROOT_DIR / "zoo/nn/backbone/resnet/wide_resnet101_2/imagenet1k_v1/wide_resnet101_2_imagenet1k_v1.pth",
        num_classes = 1000,
        transforms  = None,
        meta        = {
            "num_params": 126886696,
            "min_size"  : (1, 1),
            "categories": _IMAGENET_CATEGORIES,
            "recipe"    : "https://github.com/pytorch/vision/pull/912#issue-445437439",
            "_metrics"  : {
                "ImageNet-1K": {
                    "acc@1": 78.848,
                    "acc@5": 94.284,
                }
            },
            "_ops"      : 22.753,
            "_file_size": 242.896,
            "_docs"     : """These weights reproduce closely the results of the paper using a simple training recipe.""",
        }
    )
    IMAGENET1K_V2 = Weights(
        url         = "https://download.pytorch.org/models/wide_resnet101_2-d733dc28.pth",
        path        = ROOT_DIR / "zoo/nn/backbone/resnet/wide_resnet50_2/imagenet1k_v2/wide_resnet50_2_imagenet1k_v2.pth",
        num_classes = 1000,
        transforms  = None,
        meta        = {
            "num_params": 126886696,
            "min_size"  : (1, 1),
            "categories": _IMAGENET_CATEGORIES,
            "recipe"    : "https://github.com/pytorch/vision/issues/3995#new-recipe",
            "_metrics"  : {
                "ImageNet-1K": {
                    "acc@1": 82.510,
                    "acc@5": 96.020,
                }
            },
            "_ops"      : 22.753,
            "_file_size": 484.747,
            "_docs": """
                These weights improve upon the results of the original paper by using TorchVision's `new training recipe
                <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
            """,
        }
    )
    DEFAULT = IMAGENET1K_V2
    

# --- Model Variants ---
@BACKBONES.register(name="resnet18")
def resnet18(weights: WeightsEnum | str = ResNet18_Weights.DEFAULT, out_indices: list = None, **kwargs):
    return ResNetBackBone(
        variant     = "resnet18",
        block       = BasicBlock,
        layers      = [2, 2, 2, 2],
        weights     = ResNet18_Weights(weights),
        out_indices = out_indices,
        **kwargs
    )


@BACKBONES.register(name="resnet34")
def resnet34(weights: WeightsEnum | str = ResNet34_Weights.DEFAULT, out_indices: list = None, **kwargs):
    return ResNetBackBone(
        variant     = "resnet34",
        block       = BasicBlock,
        layers      = [3, 4, 6, 3],
        weights     = ResNet34_Weights(weights),
        out_indices = out_indices,
        **kwargs
    )


@BACKBONES.register(name="resnet50")
def resnet50(weights: WeightsEnum | str = ResNet50_Weights.DEFAULT, out_indices: list = None, **kwargs):
    return ResNetBackBone(
        variant     = "resnet50",
        block       = Bottleneck,
        layers      = [3, 4, 6, 3],
        weights     = ResNet50_Weights(weights),
        out_indices = out_indices,
        **kwargs
    )


@BACKBONES.register(name="resnet101")
def resnet101(weights: WeightsEnum | str = ResNet101_Weights.DEFAULT, out_indices: list = None, **kwargs):
    return ResNetBackBone(
        variant     = "resnet101",
        block       = Bottleneck,
        layers      = [3, 4, 23, 3],
        weights     = ResNet101_Weights(weights),
        out_indices = out_indices,
        **kwargs
    )


@BACKBONES.register(name="resnet152")
def resnet152(weights: WeightsEnum | str = ResNet152_Weights.DEFAULT, out_indices: list = None, **kwargs):
    return ResNetBackBone(
        variant     = "resnet152",
        block       = Bottleneck,
        layers      = [3, 8, 36, 3],
        weights     = ResNet152_Weights(weights),
        out_indices = out_indices,
        **kwargs
    )


@BACKBONES.register(name="resnext50_32x4d")
def resnext50_32x4d(weights: WeightsEnum | str = ResNeXt50_32X4D_Weights.DEFAULT, out_indices: list = None, **kwargs):
    kwargs["groups"]          = 32
    kwargs["width_per_group"] = 4
    return ResNetBackBone(
        variant     = "resnext50_32x4d",
        block       = Bottleneck,
        layers      = [3, 4, 6, 3],
        weights     = ResNeXt50_32X4D_Weights(weights),
        out_indices = out_indices,
        **kwargs
    )


@BACKBONES.register(name="resnext101_32x8d")
def resnext101_32x8d(weights: WeightsEnum | str = ResNeXt101_32X8D_Weights.DEFAULT, out_indices: list = None, **kwargs):
    kwargs["groups"]          = 32
    kwargs["width_per_group"] = 4
    return ResNetBackBone(
        variant     = "resnext101_32x8d",
        block       = Bottleneck,
        layers      = [3, 4, 23, 3],
        weights     = ResNeXt101_32X8D_Weights(weights),
        out_indices = out_indices,
        **kwargs
    )


@BACKBONES.register(name="resnext101_64x4d")
def resnext101_64x4d(weights: WeightsEnum | str = ResNeXt101_64X4D_Weights.DEFAULT, out_indices: list = None, **kwargs):
    kwargs["groups"]          = 64
    kwargs["width_per_group"] = 4
    return ResNetBackBone(
        variant     = "resnext101_64x4d",
        block       = Bottleneck,
        layers      = [3, 4, 23, 3],
        weights     = ResNeXt101_64X4D_Weights(weights),
        out_indices = out_indices,
        **kwargs
    )


@BACKBONES.register(name="wide_resnet50_2")
def wide_resnet50_2(weights: WeightsEnum | str = Wide_ResNet50_2_Weights.DEFAULT, out_indices: list = None, **kwargs):
    kwargs["width_per_group"] = 64 * 2
    return ResNetBackBone(
        variant     = "wide_resnet50_2",
        block       = Bottleneck,
        layers      = [3, 4, 6, 3],
        weights     = Wide_ResNet50_2_Weights(weights),
        out_indices = out_indices,
        **kwargs
    )


@BACKBONES.register(name="wide_resnet101_2")
def wide_resnet101_2(weights: WeightsEnum | str = Wide_ResNet101_2_Weights.DEFAULT, out_indices: list = None, **kwargs):
    kwargs["width_per_group"] = 64 * 2
    return ResNetBackBone(
        variant     = "wide_resnet101_2",
        block       = Bottleneck,
        layers      = [3, 4, 23, 3],
        weights     = Wide_ResNet101_2_Weights(weights),
        out_indices = out_indices,
        **kwargs
    )


# ==============================================================================
# DEBUGGING
# ==============================================================================

if __name__ == "__main__":
    model_ = resnet18(weights="default")
    x = torch.ones(1, 3, 224, 224)
    y = model_(x)
    # print(model_.features)
    # print(x)
    print(y)

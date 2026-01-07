#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""VGG backbones.

This module implements various VGG backbones using PyTorch.
"""

__all__ = [
    "VGG11_BN_Weights",
    "VGG11_Weights",
    "VGG13_BN_Weights",
    "VGG13_Weights",
    "VGG16_BN_Weights",
    "VGG16_Weights",
    "VGG19_BN_Weights",
    "VGG19_Weights",
    "vgg11",
    "vgg11_bn",
    "vgg13",
    "vgg13_bn",
    "vgg16",
    "vgg16_bn",
    "vgg19",
    "vgg19_bn",
]

import torch
import torch.nn as nn
from torchvision.models._meta import _IMAGENET_CATEGORIES
from torchvision.models.vgg import cfgs, make_layers, VGG

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
class VGGBackBone(nn.Module):
    """VGG backbone."""
    
    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        cfg        : str,
        batch_norm : bool,
        weights    : WeightsEnum,
        out_indices: list = None,
        *args, **kwargs
    ):
        """Initialize a new instance.
        
        Args:
            weights: Pre-trained weights to load.
            out_indices: List of layer indices to extract features from.
                If None, defaults to [6, 13, 23, 33, 43].
            args: Additional positional arguments for the ResNet model.
            kwargs: Additional keyword arguments for the ResNet model
        """
        super().__init__()
        # Load the base model
        if weights is not None:
            kwargs["init_weights"] = False
            kwargs["num_classes"]  = weights.num_classes
        
        base_model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), *args, **kwargs)
        
        if weights is not None:
            base_model.load_state_dict(weights.get_state_dict())
        
        # In torchvision, VGG already has a 'features' block
        self.features     = base_model.features
        self.out_indices  = out_indices or [6, 13, 23, 33, 43]
        self.out_channels = [64, 128, 256, 512, 512]
    
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
class VGG11_Weights(WeightsEnum):
    
    IMAGENET1K_V1 = Weights(
        url         = "https://download.pytorch.org/models/vgg11-8a719046.pth",
        path        = ROOT_DIR / "zoo/nn/backbone/vgg/vgg11/imagenet1k_v1/vgg11_imagenet1k_v1.pth",
        num_classes = 1000,
        transforms  = None,
        meta        = {
            "num_params": 132863336,
            "min_size"  : (32, 32),
            "categories": _IMAGENET_CATEGORIES,
            "recipe"    : "https://github.com/pytorch/vision/tree/main/references/classification#alexnet-and-vgg",
            "_metrics"  : {
                "ImageNet-1K": {
                    "acc@1": 69.020,
                    "acc@5": 88.628,
                }
            },
            "_ops"      : 7.609,
            "_file_size": 506.84,
            "_docs"     : """These weights were trained from scratch by using a simplified training recipe.""",
        }
    )
    DEFAULT = IMAGENET1K_V1
    

class VGG11_BN_Weights(WeightsEnum):
    
    IMAGENET1K_V1 = Weights(
        url         = "https://download.pytorch.org/models/vgg11_bn-6002323d.pth",
        path        = ROOT_DIR / "zoo/nn/backbone/vgg/vgg11_bn/imagenet1k_v1/vgg11_bn_imagenet1k_v1.pth",
        num_classes = 1000,
        transforms  = None,
        meta        = {
            "num_params": 132868840,
            "min_size"  : (32, 32),
            "categories": _IMAGENET_CATEGORIES,
            "recipe"    : "https://github.com/pytorch/vision/tree/main/references/classification#alexnet-and-vgg",
            "_metrics"  : {
                "ImageNet-1K": {
                    "acc@1": 70.370,
                    "acc@5": 89.810,
                }
            },
            "_ops"      : 7.609,
            "_file_size": 506.881,
            "_docs"     : """These weights were trained from scratch by using a simplified training recipe.""",
        }
    )
    DEFAULT = IMAGENET1K_V1


class VGG13_Weights(WeightsEnum):
    
    IMAGENET1K_V1 = Weights(
        url         = "https://download.pytorch.org/models/vgg13-19584684.pth",
        path        = ROOT_DIR / "zoo/nn/backbone/vgg/vgg13/imagenet1k_v1/vgg13_imagenet1k_v1.pth",
        num_classes = 1000,
        transforms  = None,
        meta        = {
            "num_params": 133047848,
            "min_size"  : (32, 32),
            "categories": _IMAGENET_CATEGORIES,
            "recipe"    : "https://github.com/pytorch/vision/tree/main/references/classification#alexnet-and-vgg",
            "_metrics"  : {
                "ImageNet-1K": {
                    "acc@1": 69.928,
                    "acc@5": 89.246,
                }
            },
            "_ops"      : 11.308,
            "_file_size": 507.545,
            "_docs"     : """These weights were trained from scratch by using a simplified training recipe.""",
        }
    )
    DEFAULT = IMAGENET1K_V1


class VGG13_BN_Weights(WeightsEnum):
    
    IMAGENET1K_V1 = Weights(
        url         = "https://download.pytorch.org/models/vgg13_bn-abd245e5.pth",
        path        = ROOT_DIR / "zoo/nn/backbone/vgg/vgg13_bn/imagenet1k_v1/vgg13_bn_imagenet1k_v1.pth",
        num_classes = 1000,
        transforms  = None,
        meta        = {
            "num_params": 133053736,
            "min_size"  : (32, 32),
            "categories": _IMAGENET_CATEGORIES,
            "recipe"    : "https://github.com/pytorch/vision/tree/main/references/classification#alexnet-and-vgg",
            "_metrics"  : {
                "ImageNet-1K": {
                    "acc@1": 71.586,
                    "acc@5": 90.374,
                }
            },
            "_ops"      : 11.308,
            "_file_size": 507.59,
            "_docs"     : """These weights were trained from scratch by using a simplified training recipe.""",
        }
    )
    DEFAULT = IMAGENET1K_V1


class VGG16_Weights(WeightsEnum):
    
    IMAGENET1K_V1 = Weights(
        url         = "https://download.pytorch.org/models/vgg16-397923af.pth",
        path        = ROOT_DIR / "zoo/nn/backbone/vgg/vgg16/imagenet1k_v1/vgg16_imagenet1k_v1.pth",
        num_classes = 1000,
        transforms  = None,
        meta        = {
            "num_params": 138357544,
            "min_size"  : (32, 32),
            "categories": _IMAGENET_CATEGORIES,
            "recipe"    : "https://github.com/pytorch/vision/tree/main/references/classification#alexnet-and-vgg",
            "_metrics"  : {
                "ImageNet-1K": {
                    "acc@1": 71.592,
                    "acc@5": 90.382,
                }
            },
            "_ops"      : 15.47,
            "_file_size": 527.796,
            "_docs"     : """These weights were trained from scratch by using a simplified training recipe.""",
        }
    )
    DEFAULT = IMAGENET1K_V1


class VGG16_BN_Weights(WeightsEnum):
    
    IMAGENET1K_V1 = Weights(
        url         = "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth",
        path        = ROOT_DIR / "zoo/nn/backbone/vgg/vgg16_bn/imagenet1k_v1/vgg16_bn_imagenet1k_v1.pth",
        num_classes = 1000,
        transforms  = None,
        meta        = {
            "num_params": 138365992,
            "min_size"  : (32, 32),
            "categories": _IMAGENET_CATEGORIES,
            "recipe"    : "https://github.com/pytorch/vision/tree/main/references/classification#alexnet-and-vgg",
            "_metrics"  : {
                "ImageNet-1K": {
                    "acc@1": 73.360,
                    "acc@5": 91.516,
                }
            },
            "_ops"      : 15.47,
            "_file_size": 527.866,
            "_docs"     : """These weights were trained from scratch by using a simplified training recipe.""",
        }
    )
    DEFAULT = IMAGENET1K_V1


class VGG19_Weights(WeightsEnum):
    
    IMAGENET1K_V1 = Weights(
        url         = "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth",
        path        = ROOT_DIR / "zoo/nn/backbone/vgg/vgg16/imagenet1k_v1/vgg16_imagenet1k_v1.pth",
        num_classes = 1000,
        transforms  = None,
        meta        = {
            "num_params": 143667240,
            "min_size"  : (32, 32),
            "categories": _IMAGENET_CATEGORIES,
            "recipe"    : "https://github.com/pytorch/vision/tree/main/references/classification#alexnet-and-vgg",
            "_metrics"  : {
                "ImageNet-1K": {
                    "acc@1": 72.376,
                    "acc@5": 90.876,
                }
            },
            "_ops"      : 19.632,
            "_file_size": 548.051,
            "_docs"     : """These weights were trained from scratch by using a simplified training recipe.""",
        }
    )
    DEFAULT = IMAGENET1K_V1


class VGG19_BN_Weights(WeightsEnum):
    
    IMAGENET1K_V1 = Weights(
        url         = "https://download.pytorch.org/models/vgg19_bn-c79401a0.pth",
        path        = ROOT_DIR / "zoo/nn/backbone/vgg/vgg16_bn/imagenet1k_v1/vgg16_bn_imagenet1k_v1.pth",
        num_classes = 1000,
        transforms  = None,
        meta        = {
            "num_params": 143678248,
            "min_size"  : (32, 32),
            "categories": _IMAGENET_CATEGORIES,
            "recipe"    : "https://github.com/pytorch/vision/tree/main/references/classification#alexnet-and-vgg",
            "_metrics"  : {
                "ImageNet-1K": {
                    "acc@1": 74.218,
                    "acc@5": 91.842,
                }
            },
            "_ops"      : 19.632,
            "_file_size": 548.143,
            "_docs"     : """These weights were trained from scratch by using a simplified training recipe.""",
        }
    )
    DEFAULT = IMAGENET1K_V1


# --- Model Variants ---
@BACKBONES.register(name="vgg11")
def vgg11(weights: WeightsEnum | str = VGG11_Weights.DEFAULT, out_indices: list = None, **kwargs):
    return VGGBackBone(
        cfg         = "A",
        batch_norm  = False,
        weights     = VGG11_Weights(weights),
        out_indices = out_indices,
        **kwargs
    )


@BACKBONES.register(name="vgg11_bn")
def vgg11_bn(weights: WeightsEnum | str = VGG11_BN_Weights.DEFAULT, out_indices: list = None, **kwargs):
    return VGGBackBone(
        cfg         = "A",
        batch_norm  = True,
        weights     = VGG11_BN_Weights(weights),
        out_indices = out_indices,
        **kwargs
    )


@BACKBONES.register(name="vgg13")
def vgg13(weights: WeightsEnum | str = VGG13_Weights.DEFAULT, out_indices: list = None, **kwargs):
    return VGGBackBone(
        cfg         = "B",
        batch_norm  = False,
        weights     = VGG13_Weights(weights),
        out_indices = out_indices,
        **kwargs
    )


@BACKBONES.register(name="vgg13_bn")
def vgg13_bn(weights: WeightsEnum | str = VGG13_BN_Weights.DEFAULT, out_indices: list = None, **kwargs):
    return VGGBackBone(
        cfg         = "B",
        batch_norm  = True,
        weights     = VGG13_BN_Weights(weights),
        out_indices = out_indices,
        **kwargs
    )


@BACKBONES.register(name="vgg16")
def vgg16(weights: WeightsEnum | str = VGG16_Weights.DEFAULT, out_indices: list = None, **kwargs):
    return VGGBackBone(
        cfg         = "D",
        batch_norm  = False,
        weights     = VGG16_Weights(weights),
        out_indices = out_indices,
        **kwargs
    )


@BACKBONES.register(name="vgg16_bn")
def vgg16_bn(weights: WeightsEnum | str = VGG16_BN_Weights.DEFAULT, out_indices: list = None, **kwargs):
    return VGGBackBone(
        cfg         = "D",
        batch_norm  = True,
        weights     = VGG16_BN_Weights(weights),
        out_indices = out_indices,
        **kwargs
    )


@BACKBONES.register(name="vgg19")
def vgg19(weights: WeightsEnum | str = VGG19_Weights.DEFAULT, out_indices: list = None, **kwargs):
    return VGGBackBone(
        cfg         = "E",
        batch_norm  = False,
        weights     = VGG19_Weights(weights),
        out_indices = out_indices,
        **kwargs
    )


@BACKBONES.register(name="vgg19_bn")
def vgg19_bn(weights: WeightsEnum | str = VGG19_BN_Weights.DEFAULT, out_indices: list = None, **kwargs):
    return VGGBackBone(
        cfg         = "E",
        batch_norm  = True,
        weights     = VGG19_BN_Weights(weights),
        out_indices = out_indices,
        **kwargs
    )


# ==============================================================================
# DEBUGGING
# ==============================================================================

if __name__ == "__main__":
    model_ = vgg11(weights="default")
    x = torch.ones(1, 3, 224, 224)
    y = model_(x)
    # print(model_.features)
    # print(x)
    print(y)

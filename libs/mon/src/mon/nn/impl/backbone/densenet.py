#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""DenseNet backbones.

This module implements various VGG backbones using PyTorch.
"""

__all__ = [
    "DenseNet121_Weights",
    "DenseNet161_Weights",
    "DenseNet169_Weights",
    "DenseNet201_Weights",
    "densenet121",
    "densenet161",
    "densenet169",
    "densenet201",
]

import torch
import torch.nn as nn
from torchvision.models._meta import _IMAGENET_CATEGORIES
from torchvision.models.densenet import DenseNet

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
class DenseNetBackBone(nn.Module):
    """DenseNet backbone."""
    
    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        variant          : str,
        growth_rate      : int,
        block_config     : tuple[int, int, int, int],
        num_init_features: int,
        weights          : WeightsEnum = None,
        out_indices      : list        = None,
        *args, **kwargs
    ):
        """Initialize a new instance.
        
        Args:
            variant: Variant of DenseNet to use.
            growth_rate: How many filters to add each layer (`k` in paper).
            block_config: How many layers in each pooling block.
            num_init_features: The number of filters to learn in the first
                convolution layer.
            weights: Pre-trained weights to load.
            out_indices: List of layer indices to extract features from.
                If None, defaults to [3, 5, 7, 11].
            args: Additional positional arguments for the ResNet model.
            kwargs: Additional keyword arguments for the ResNet model
        """
        super().__init__()
        # Load the base model
        if weights is not None:
            kwargs["num_classes"] = weights.num_classes
            
        base_model = DenseNet(
            growth_rate       = growth_rate,
            block_config      = block_config,
            num_init_features = num_init_features,
            *args, **kwargs
        )
        
        if weights is not None:
            base_model.load_state_dict(weights.get_state_dict())
        
        # In torchvision, DenseNet already has a 'features' block
        self.features     = base_model.features
        self.out_indices  = out_indices or [3, 5, 7, 11]
        self.out_channels = self._get_out_channels(variant)
    
    def _get_out_channels(self, variant):
        mapping = {
            "densenet121": [64, 256, 512, 1024],
            "densenet161": [96, 384, 768, 2208],
            "densenet169": [64, 256, 512, 1664],
            "densenet201": [64, 256, 512, 1920],
        }
        return mapping.get(variant, [64, 256, 512, 1024])
    
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
class DenseNet121_Weights(WeightsEnum):
    
    IMAGENET1K_V1 = Weights(
        url         = "https://download.pytorch.org/models/densenet121-a639ec97.pth",
        path        = ROOT_DIR / "zoo/nn/backbone/densenet/densenet121/imagenet1k_v1/densenet121_imagenet1k_v1.pth",
        num_classes = 1000,
        transforms  = None,
        meta        = {
            "num_params": 7978856,
            "min_size"  : (29, 29),
            "categories": _IMAGENET_CATEGORIES,
            "recipe"    : "https://github.com/pytorch/vision/pull/116",
            "_metrics"  : {
                "ImageNet-1K": {
                    "acc@1": 74.434,
                    "acc@5": 91.972,
                }
            },
            "_ops"      : 2.834,
            "_file_size": 30.845,
            "_docs"     : """These weights are ported from LuaTorch.""",
        }
    )
    DEFAULT = IMAGENET1K_V1


class DenseNet161_Weights(WeightsEnum):
    
    IMAGENET1K_V1 = Weights(
        url         = "https://download.pytorch.org/models/densenet161-8d451a50.pth",
        path        = ROOT_DIR / "zoo/nn/backbone/densenet/densenet161/imagenet1k_v1/densenet161_imagenet1k_v1.pth",
        num_classes = 1000,
        transforms  = None,
        meta        = {
            "num_params": 28681000,
            "min_size"  : (29, 29),
            "categories": _IMAGENET_CATEGORIES,
            "recipe"    : "https://github.com/pytorch/vision/pull/116",
            "_metrics"  : {
                "ImageNet-1K": {
                    "acc@1": 77.138,
                    "acc@5": 93.560,
                }
            },
            "_ops"      : 7.728,
            "_file_size": 110.369,
            "_docs"     : """These weights are ported from LuaTorch.""",
        }
    )
    DEFAULT = IMAGENET1K_V1


class DenseNet169_Weights(WeightsEnum):
    
    IMAGENET1K_V1 = Weights(
        url         = "https://download.pytorch.org/models/densenet169-b2777c0a.pth",
        path        = ROOT_DIR / "zoo/nn/backbone/densenet/densenet169/imagenet1k_v1/densenet169_imagenet1k_v1.pth",
        num_classes = 1000,
        transforms  = None,
        meta        = {
            "num_params": 14149480,
            "min_size"  : (29, 29),
            "categories": _IMAGENET_CATEGORIES,
            "recipe"    : "https://github.com/pytorch/vision/pull/116",
            "_metrics"  : {
                "ImageNet-1K": {
                    "acc@1": 75.600,
                    "acc@5": 92.806,
                }
            },
            "_ops"      : 3.36,
            "_file_size": 54.708,
            "_docs"     : """These weights are ported from LuaTorch.""",
        }
    )
    DEFAULT = IMAGENET1K_V1


class DenseNet201_Weights(WeightsEnum):
    
    IMAGENET1K_V1 = Weights(
        url         = "https://download.pytorch.org/models/densenet201-c1103571.pth",
        path        = ROOT_DIR / "zoo/nn/backbone/densenet/densenet201/imagenet1k_v1/densenet201_imagenet1k_v1.pth",
        num_classes = 1000,
        transforms  = None,
        meta        = {
            "num_params": 20013928,
            "min_size"  : (29, 29),
            "categories": _IMAGENET_CATEGORIES,
            "recipe"    : "https://github.com/pytorch/vision/pull/116",
            "_metrics"  : {
                "ImageNet-1K": {
                    "acc@1": 76.896,
                    "acc@5": 93.370,
                }
            },
            "_ops"      : 4.291,
            "_file_size": 77.373,
            "_docs"     : """These weights are ported from LuaTorch.""",
        }
    )
    DEFAULT = IMAGENET1K_V1
    

# --- Model Variants ---
@BACKBONES.register(name="densenet121")
def densenet121(weights: WeightsEnum | str = DenseNet121_Weights.DEFAULT, out_indices: list = None, **kwargs):
    return DenseNetBackBone(
        variant           = "densenet121",
        growth_rate       = 32,
        block_config      = (6, 12, 24, 16),
        num_init_features = 64,
        weights           = DenseNet121_Weights(weights),
        out_indices       = out_indices,
        **kwargs
    )


@BACKBONES.register(name="densenet161")
def densenet161(weights: WeightsEnum | str = DenseNet161_Weights.DEFAULT, out_indices: list = None, **kwargs):
    return DenseNetBackBone(
        variant           = "densenet161",
        growth_rate       = 48,
        block_config      = (6, 12, 36, 24),
        num_init_features = 96,
        weights           = DenseNet161_Weights(weights),
        out_indices       = out_indices,
        **kwargs
    )


@BACKBONES.register(name="densenet169")
def densenet169(weights: WeightsEnum | str = DenseNet169_Weights.DEFAULT, out_indices: list = None, **kwargs):
    return DenseNetBackBone(
        variant           = "densenet169",
        growth_rate       = 32,
        block_config      = (6, 12, 32, 32),
        num_init_features = 64,
        weights           = DenseNet169_Weights(weights),
        out_indices       = out_indices,
        **kwargs
    )


@BACKBONES.register(name="densenet201")
def densenet201(weights: WeightsEnum | str = DenseNet201_Weights.DEFAULT, out_indices: list = None, **kwargs):
    return DenseNetBackBone(
        variant           = "densenet201",
        growth_rate       = 32,
        block_config      = (6, 12, 48, 32),
        num_init_features = 64,
        weights           = DenseNet201_Weights(weights),
        out_indices       = out_indices,
        **kwargs
    )


# ==============================================================================
# DEBUGGING
# ==============================================================================

if __name__ == "__main__":
    model_ = densenet121(weights="default")
    x = torch.ones(1, 3, 224, 224)
    y = model_(x)
    # print(model_.features)
    # print(x)
    print(y)

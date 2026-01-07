#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""ConvNeXt backbones.

This module implements various ConvNeXt backbones using PyTorch.
"""

__all__ = [
    "ConvNeXt_Base_Weights",
    "ConvNeXt_Large_Weights",
    "ConvNeXt_Small_Weights",
    "ConvNeXt_Tiny_Weights",
    "convnext_base",
    "convnext_large",
    "convnext_small",
    "convnext_tiny",
]

import torch
import torch.nn as nn
from torchvision.models._meta import _IMAGENET_CATEGORIES
from torchvision.models.convnext import CNBlockConfig, ConvNeXt

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
class ConvNeXtBackBone(nn.Module):
    """ConvNeXt backbone."""
    
    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        variant              : str,
        block_setting        : list[CNBlockConfig],
        stochastic_depth_prob: float,
        weights              : WeightsEnum = None,
        out_indices          : list        = None,
        *args, **kwargs
    ):
        """Initialize a new instance.
        
        Args:
            variant: Variant of DenseNet to use.
            block_setting: A list of block settings.
            stochastic_depth_prob: Probability of an element to be zeroed.
            weights: Pre-trained weights to load.
            out_indices: List of layer indices to extract features from.
                If None, defaults to [1, 3, 5, 7].
            args: Additional positional arguments for the ResNet model.
            kwargs: Additional keyword arguments for the ResNet model
        """
        super().__init__()
        # Load the base model
        if weights is not None:
            kwargs["num_classes"] = weights.num_classes
            
        base_model = ConvNeXt(
            block_setting         = block_setting,
            stochastic_depth_prob = stochastic_depth_prob,
            *args, **kwargs
        )
        
        if weights is not None:
            base_model.load_state_dict(weights.get_state_dict())
        
        # In torchvision, ConvNeXt already has a 'features' block
        self.features     = base_model.features
        self.out_indices  = out_indices or [1, 3, 5, 7]
        self.out_channels = self._get_out_channels(variant)
    
    def _get_out_channels(self, variant):
        mapping = {
            "convnext_tiny":  [96, 192, 384, 768],
            "convnext_small": [96, 192, 384, 768],
            "convnext_base":  [128, 256, 512, 1024],
            "convnext_large": [192, 384, 768, 1536],
        }
        return mapping.get(variant, [96, 192, 384, 768])
    
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
class ConvNeXt_Tiny_Weights(WeightsEnum):
    
    IMAGENET1K_V1 = Weights(
        url         = "https://download.pytorch.org/models/convnext_tiny-983f1562.pth",
        path        = ROOT_DIR / "zoo/nn/backbone/convnext/convnext_tiny/imagenet1k_v1/convnext_tiny_imagenet1k_v1.pth",
        num_classes = 1000,
        transforms  = None,
        meta        = {
            "num_params": 28589128,
            "min_size"  : (32, 32),
            "categories": _IMAGENET_CATEGORIES,
            "recipe"    : "https://github.com/pytorch/vision/tree/main/references/classification#convnext",
            "_metrics"  : {
                "ImageNet-1K": {
                    "acc@1": 82.520,
                    "acc@5": 96.146,
                }
            },
            "_ops"      : 4.456,
            "_file_size": 109.119,
            "_docs"     : """
                These weights improve upon the results of the original paper by using a modified version of TorchVision's
                `new training recipe
                <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
            """,
        }
    )
    DEFAULT = IMAGENET1K_V1


class ConvNeXt_Small_Weights(WeightsEnum):
    
    IMAGENET1K_V1 = Weights(
        url         = "https://download.pytorch.org/models/convnext_small-0c510722.pth",
        path        = ROOT_DIR / "zoo/nn/backbone/convnext/convnext_small/imagenet1k_v1/convnext_small_imagenet1k_v1.pth",
        num_classes = 1000,
        transforms  = None,
        meta        = {
            "num_params": 50223688,
            "min_size"  : (32, 32),
            "categories": _IMAGENET_CATEGORIES,
            "recipe"    : "https://github.com/pytorch/vision/tree/main/references/classification#convnext",
            "_metrics"  : {
                "ImageNet-1K": {
                    "acc@1": 83.616,
                    "acc@5": 96.650,
                }
            },
            "_ops"      : 8.684,
            "_file_size": 191.703,
            "_docs"     : """
                These weights improve upon the results of the original paper by using a modified version of TorchVision's
                `new training recipe
                <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
            """,
        }
    )
    DEFAULT = IMAGENET1K_V1


class ConvNeXt_Base_Weights(WeightsEnum):
    
    IMAGENET1K_V1 = Weights(
        url         = "https://download.pytorch.org/models/convnext_base-6075fbad.pth",
        path        = ROOT_DIR / "zoo/nn/backbone/convnext/convnext_base/imagenet1k_v1/convnext_base_imagenet1k_v1.pth",
        num_classes = 1000,
        transforms  = None,
        meta        = {
            "num_params": 88591464,
            "min_size"  : (32, 32),
            "categories": _IMAGENET_CATEGORIES,
            "recipe"    : "https://github.com/pytorch/vision/tree/main/references/classification#convnext",
            "_metrics"  : {
                "ImageNet-1K": {
                    "acc@1": 84.062,
                    "acc@5": 96.870,
                }
            },
            "_ops"      : 15.355,
            "_file_size": 338.064,
            "_docs"     : """
                These weights improve upon the results of the original paper by using a modified version of TorchVision's
                `new training recipe
                <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
            """,
        }
    )
    DEFAULT = IMAGENET1K_V1


class ConvNeXt_Large_Weights(WeightsEnum):
    
    IMAGENET1K_V1 = Weights(
        url         = "https://download.pytorch.org/models/convnext_large-ea097f82.pth",
        path        = ROOT_DIR / "zoo/nn/backbone/convnext/convnext_large/imagenet1k_v1/convnext_large_imagenet1k_v1.pth",
        num_classes = 1000,
        transforms  = None,
        meta        = {
            "num_params": 197767336,
            "min_size"  : (32, 32),
            "categories": _IMAGENET_CATEGORIES,
            "recipe"    : "https://github.com/pytorch/vision/tree/main/references/classification#convnext",
            "_metrics"  : {
                "ImageNet-1K": {
                    "acc@1": 84.414,
                    "acc@5": 96.976,
                }
            },
            "_ops"      : 34.361,
            "_file_size": 754.537,
            "_docs"     : """
                These weights improve upon the results of the original paper by using a modified version of TorchVision's
                `new training recipe
                <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
            """,
        }
    )
    DEFAULT = IMAGENET1K_V1


# --- Model Variants ---
@BACKBONES.register(name="convnext_tiny")
def convnext_tiny(weights: WeightsEnum | str = ConvNeXt_Tiny_Weights.DEFAULT, out_indices: list = None, **kwargs):
    block_setting = [
        CNBlockConfig(96, 192, 3),
        CNBlockConfig(192, 384, 3),
        CNBlockConfig(384, 768, 9),
        CNBlockConfig(768, None, 3),
    ]
    stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.1)
    return ConvNeXtBackBone(
        variant               = "convnext_tiny",
        block_setting         = block_setting,
        stochastic_depth_prob = stochastic_depth_prob,
        weights               = ConvNeXt_Tiny_Weights(weights),
        out_indices           = out_indices,
        **kwargs
    )


@BACKBONES.register(name="convnext_small")
def convnext_small(weights: WeightsEnum | str = ConvNeXt_Small_Weights.DEFAULT, out_indices: list = None, **kwargs):
    block_setting = [
        CNBlockConfig(96, 192, 3),
        CNBlockConfig(192, 384, 3),
        CNBlockConfig(384, 768, 27),
        CNBlockConfig(768, None, 3),
    ]
    stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.4)
    return ConvNeXtBackBone(
        variant               = "convnext_small",
        block_setting         = block_setting,
        stochastic_depth_prob = stochastic_depth_prob,
        weights               = ConvNeXt_Small_Weights(weights),
        out_indices           = out_indices,
        **kwargs
    )


@BACKBONES.register(name="convnext_base")
def convnext_base(weights: WeightsEnum | str = ConvNeXt_Base_Weights.DEFAULT, out_indices: list = None, **kwargs):
    block_setting = [
        CNBlockConfig(128, 256, 3),
        CNBlockConfig(256, 512, 3),
        CNBlockConfig(512, 1024, 27),
        CNBlockConfig(1024, None, 3),
    ]
    stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.5)
    return ConvNeXtBackBone(
        variant               = "convnext_base",
        block_setting         = block_setting,
        stochastic_depth_prob = stochastic_depth_prob,
        weights               = ConvNeXt_Base_Weights(weights),
        out_indices           = out_indices,
        **kwargs
    )


@BACKBONES.register(name="convnext_large")
def convnext_large(weights: WeightsEnum | str = ConvNeXt_Large_Weights.DEFAULT, out_indices: list = None, **kwargs):
    block_setting = [
        CNBlockConfig(192, 384, 3),
        CNBlockConfig(384, 768, 3),
        CNBlockConfig(768, 1536, 27),
        CNBlockConfig(1536, None, 3),
    ]
    stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.5)
    return ConvNeXtBackBone(
        variant               = "convnext_large",
        block_setting         = block_setting,
        stochastic_depth_prob = stochastic_depth_prob,
        weights               = ConvNeXt_Large_Weights(weights),
        out_indices           = out_indices,
        **kwargs
    )


# ==============================================================================
# DEBUGGING
# ==============================================================================

if __name__ == "__main__":
    model_ = convnext_tiny(weights="default")
    x = torch.ones(1, 3, 224, 224)
    y = model_(x)
    # print(model_.features)
    # print(x)
    print(y)

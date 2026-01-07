#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""MobileNetV2 backbones.

This module implements various MobileNetV2 backbones using PyTorch.
"""

__all__ = [
    "MobileNet_V2_Weights",
    "mobilenet_v2",
]

import torch
import torch.nn as nn
from torchvision.models._meta import _IMAGENET_CATEGORIES
from torchvision.models.mobilenetv2 import MobileNetV2

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
class MobileNetV2BackBone(nn.Module):
    """MobileNetV2 backbone."""
    
    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        weights    : WeightsEnum = None,
        out_indices: list        = None,
        *args, **kwargs
    ):
        """Initialize a new instance.
        
        Args:
            weights: Pre-trained weights to load.
            out_indices: List of layer indices to extract features from.
                If None, defaults to [3, 6, 13, 18].
            args: Additional positional arguments for the ResNet model.
            kwargs: Additional keyword arguments for the ResNet model
        """
        super().__init__()
        # Load the base model
        if weights is not None:
            kwargs["num_classes"] = weights.num_classes
        
        base_model = MobileNetV2(*args, **kwargs)
        
        if weights is not None:
            base_model.load_state_dict(weights.get_state_dict())
        
        # In torchvision, MobileNetV2 already has a 'features' block
        self.features     = base_model.features
        self.out_indices  = out_indices or [3, 6, 13, 18]
        self.out_channels = [24, 32, 96, 1280]
    
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
class MobileNet_V2_Weights(WeightsEnum):
    
    IMAGENET1K_V1 = Weights(
        url         = "https://download.pytorch.org/models/mobilenet_v2-b0353104.pth",
        path        = ROOT_DIR / "zoo/nn/backbone/mobilenet/mobilenet_v2/imagenet1k_v1/mobilenet_v2_imagenet1k_v1.pth",
        num_classes = 1000,
        transforms  = None,
        meta        = {
            "num_params": 3504872,
            "min_size"  : (1, 1),
            "categories": _IMAGENET_CATEGORIES,
            "recipe"    : "https://github.com/pytorch/vision/tree/main/references/classification#mobilenetv2",
            "_metrics"  : {
                "ImageNet-1K": {
                    "acc@1": 71.878,
                    "acc@5": 90.286,
                }
            },
            "_ops"      : 0.301,
            "_file_size": 13.555,
            "_docs"     : """These weights reproduce closely the results of the paper using a simple training recipe.""",
        }
    )
    IMAGENET1K_V2 = Weights(
        url         = "https://download.pytorch.org/models/mobilenet_v2-7ebf99e0.pth",
        path        = ROOT_DIR / "zoo/nn/backbone/mobilenet/mobilenet_v2/imagenet1k_v2/mobilenet_v2_imagenet1k_v2.pth",
        num_classes = 1000,
        transforms  = None,
        meta        = {
            "num_params": 3504872,
            "min_size"  : (1, 1),
            "categories": _IMAGENET_CATEGORIES,
            "recipe"    : "https://github.com/pytorch/vision/issues/3995#new-recipe-with-reg-tuning",
            "_metrics"  : {
                "ImageNet-1K": {
                    "acc@1": 72.154,
                    "acc@5": 90.822,
                }
            },
            "_ops"      : 0.301,
            "_file_size": 13.598,
            "_docs"     : """
                These weights improve upon the results of the original paper by using a modified version of TorchVision's
                `new training recipe
                <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
            """,
        }
    )
    DEFAULT = IMAGENET1K_V2


# --- Model Variants ---
@BACKBONES.register(name="mobilenet_v2")
def mobilenet_v2(weights: WeightsEnum | str = MobileNet_V2_Weights.DEFAULT, out_indices: list = None, **kwargs):
    return MobileNetV2BackBone(
        weights     = MobileNet_V2_Weights(weights),
        out_indices = out_indices,
        **kwargs
    )


# ==============================================================================
# DEBUGGING
# ==============================================================================

if __name__ == "__main__":
    model_ = mobilenet_v2(weights="default")
    x = torch.ones(1, 3, 224, 224)
    y = model_(x)
    # print(model_.features)
    # print(x)
    print(y)

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""AlexNet backbones.

This module implements various AlexNet backbones using PyTorch.
"""

__all__ = [
    "AlexNet_Weights",
    "alexnet",
]

import torch
import torch.nn as nn
from torchvision.models._meta import _IMAGENET_CATEGORIES
from torchvision.models.alexnet import AlexNet

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
class AlexNetBackBone(nn.Module):
    """AlexNet backbone."""
    
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
                If None, defaults to [2, 5, 8, 10, 12].
            args: Additional positional arguments for the ResNet model.
            kwargs: Additional keyword arguments for the ResNet model
        """
        super().__init__()
        # Load the base model
        if weights is not None:
            kwargs["num_classes"] = weights.num_classes
            
        base_model = AlexNet(*args, **kwargs)
        
        if weights is not None:
            base_model.load_state_dict(weights.get_state_dict())
        
        # In torchvision, AlexNet already has a 'features' block
        self.features     = base_model.features
        self.out_indices  = out_indices or [2, 5, 8, 10, 12]
        self.out_channels = [64, 192, 384, 256, 256]
        
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
class AlexNet_Weights(WeightsEnum):
    
    IMAGENET1K_V1 = Weights(
        url         = "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth",
        path        = ROOT_DIR / "zoo/nn/backbone/alexnet/alexnet/imagenet1k_v1/alexnet_imagenet1k_v1.pth",
        num_classes = 1000,
        transforms  = None,
        meta        = {
            "num_params": 61100840,
            "min_size"  : (63, 63),
            "categories": _IMAGENET_CATEGORIES,
            "recipe"    : "https://github.com/pytorch/vision/tree/main/references/classification#alexnet-and-vgg",
            "_metrics"  : {
                "ImageNet-1K": {
                    "acc@1": 56.522,
                    "acc@5": 79.066,
                }
            },
            "_ops"      : 0.714,
            "_file_size": 233.087,
            "_docs"     : """These weights reproduce closely the results of the paper using a simplified training recipe.""",
        }
    )
    DEFAULT = IMAGENET1K_V1


# --- Model Variants ---
@BACKBONES.register(name="resnet18")
def alexnet(weights: WeightsEnum | str = AlexNet_Weights.DEFAULT, out_indices: list = None, **kwargs):
    return AlexNetBackBone(
        weights     = AlexNet_Weights(weights),
        out_indices = out_indices,
        **kwargs
    )


# ==============================================================================
# DEBUGGING
# ==============================================================================

if __name__ == "__main__":
    model_ = alexnet(weights="default")
    x = torch.ones(1, 3, 224, 224)
    y = model_(x)
    # print(model_.features)
    # print(x)
    print(y)

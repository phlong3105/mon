#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""MobileNetV3 backbones.

This module provides various MobileNetV3 backbones using PyTorch.
"""

from __future__ import annotations

__all__ = [
    "MobileNet_V3_Large_Weights",
    "MobileNet_V3_Small_Weights",
    "mobilenet_v3_large",
    "mobilenet_v3_small",
]

import torch
import torch.nn as nn
from torchvision.models._meta import _IMAGENET_CATEGORIES
from torchvision.models.mobilenetv3 import (
    _mobilenet_v3_conf,
    InvertedResidualConfig,
    MobileNetV3,
)

from mon.core import BACKBONES, MLType, Path, Task, WEIGHTS, ZOO_DIR
from mon.core.dtypes import Weights, WeightsEnum
from ...base import RegistrableMixin

current_file = Path(__file__).normalize()
current_dir  = current_file.parents[0]


# ==============================================================================
# region BASE CLASSES & MIXINS
# ==============================================================================

# --- Base Classes ---

class MobileNetV3BackBone(nn.Module, RegistrableMixin):
    """MobileNetV3 backbone.

    Attributes:
        features (torch.nn.Sequential): The feature extraction layers.
        out_indices (list): List of layer indices to extract features from.
        out_channels (list): List of output channels for each extracted layer.
    """

    _arch     : str          = "mobilenet"
    _name     : str          = None
    _tasks    : list[Task]   = [Task.BACKBONE]
    _mltypes  : list[MLType] = []
    _model_dir: Path         = current_dir

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        name                     : str,
        inverted_residual_setting: list[InvertedResidualConfig],
        last_channel             : int,
        weights                  : WeightsEnum | None = None,
        out_indices              : list | None        = None,
        *args, **kwargs
    ):
        """Initialize a new instance.

        Args:
            name: Variant of MobileNetV3 to use.
            inverted_residual_setting: A list of InvertedResidualConfig to
                construct blocks.
            last_channel: Channel dimension of the last convolutional layer.
            weights: Pre-trained weights to load.
            out_indices: List of layer indices to extract features from.
            args: Additional positional arguments for the ResNet model.
            kwargs: Additional keyword arguments for the ResNet model
        """
        super().__init__(name=name, *args, **kwargs)

        # Load the base model
        if isinstance(weights, WeightsEnum):
            kwargs["num_classes"] = weights.num_classes

        base_model = MobileNetV3(
            inverted_residual_setting = inverted_residual_setting,
            last_channel              = last_channel,
            *args, **kwargs
        )

        if isinstance(weights, WeightsEnum):
            base_model.load_state_dict(weights.get_state_dict())

        # In torchvision, MobileNetV3 already has a 'features' block
        self.features = base_model.features

        if "large" in name:
            self.out_indices  = out_indices or [3, 6, 12, 15]
            self.out_channels = [24, 40, 112, 160]
        else:  # Small variant
            self.out_indices  = out_indices or [0, 3, 8, 11]
            self.out_channels = [16, 24, 48, 96]

    # --- Callable & Context Manager ---
    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Forward the input through the network.

        Args:
            x: Input tensor of shape (B, C, H, W) and values ranging
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


# --- Mixins ---


# endregion


# ==============================================================================
# region CONCRETE IMPLEMENTATIONS
# ==============================================================================

# --- Pre-trained Weights ---

@WEIGHTS.register(arch="mobilenet", name="mobilenet_v3_large")
class MobileNet_V3_Large_Weights(WeightsEnum):

    IMAGENET1K_V1 = Weights(
        url         = "https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth",
        path        = ZOO_DIR / "nn/backbone/mobilenet/mobilenet_v3_large/imagenet1k_v1/mobilenet_v3_large_imagenet1k_v1.pth",
        num_classes = 1000,
        transforms  = None,
        meta        = {
            "num_params": 5483032,
            "min_size"  : (1, 1),
            "categories": _IMAGENET_CATEGORIES,
            "recipe"    : "https://github.com/pytorch/vision/tree/main/references/classification#mobilenetv3-large--small",
            "_metrics"  : {
                "ImageNet-1K": {
                    "acc@1": 74.042,
                    "acc@5": 91.340,
                }
            },
            "_ops"      : 0.217,
            "_file_size": 21.114,
            "_docs"     : """These weights were trained from scratch by using a simple training recipe.""",
        }
    )
    IMAGENET1K_V2 = Weights(
        url         = "https://download.pytorch.org/models/mobilenet_v3_large-5c1a4163.pth",
        path        = ZOO_DIR / "nn/backbone/mobilenet/mobilenet_v3_large/imagenet1k_v2/mobilenet_v3_large_imagenet1k_v2.pth",
        num_classes = 1000,
        transforms  = None,
        meta        = {
            "num_params": 5483032,
            "min_size"  : (1, 1),
            "categories": _IMAGENET_CATEGORIES,
            "recipe"    : "https://github.com/pytorch/vision/issues/3995#new-recipe-with-reg-tuning",
            "_metrics"  : {
                "ImageNet-1K": {
                    "acc@1": 75.274,
                    "acc@5": 92.566,
                }
            },
            "_ops"      : 0.217,
            "_file_size": 21.107,
            "_docs"     : """
                These weights improve marginally upon the results of the original paper by using a modified version of
                TorchVision's `new training recipe
                <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
            """,
        }
    )
    DEFAULT = IMAGENET1K_V2


@WEIGHTS.register(arch="mobilenet", name="mobilenet_v3_small")
class MobileNet_V3_Small_Weights(WeightsEnum):

    IMAGENET1K_V1 = Weights(
        url         = "https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth",
        path        = ZOO_DIR / "nn/backbone/mobilenet/mobilenet_v3_small/imagenet1k_v1/mobilenet_v3_small_imagenet1k_v1.pth",
        num_classes = 1000,
        transforms  = None,
        meta        = {
            "num_params": 2542856,
            "min_size"  : (1, 1),
            "categories": _IMAGENET_CATEGORIES,
            "recipe"    : "https://github.com/pytorch/vision/tree/main/references/classification#mobilenetv3-large--small",
            "_metrics"  : {
                "ImageNet-1K": {
                    "acc@1": 67.668,
                    "acc@5": 87.402,
                }
            },
            "_ops"      : 0.057,
            "_file_size": 9.829,
            "_docs"     : """These weights improve upon the results of the original paper by using a simple training recipe.""",
        }
    )
    DEFAULT = IMAGENET1K_V1


# --- Model Variants ---

@BACKBONES.register(name="mobilenet_v3_large")
def mobilenet_v3_large(
    weights    : WeightsEnum | str | None = MobileNet_V3_Large_Weights.DEFAULT,
    out_indices: list | None = None,
    *args, **kwargs
):
    """Create a MobileNetV3 Large backbone.

    Args:
        weights: Pre-trained weights to load. Defaults to
            MobileNet_V3_Large_Weights.DEFAULT.
        out_indices: List of layer indices to extract features from. Defaults
            to None.
        args: Additional positional arguments for the ResNet model.
        kwargs: Additional keyword arguments for the ResNet model.

    Returns:
        A MobileNetV3 Large backbone model.
    """
    inverted_residual_setting, last_channel = _mobilenet_v3_conf("mobilenet_v3_large", **kwargs)
    return MobileNetV3BackBone(
        name         = "mobilenet_v3_large",
        inverted_residual_setting = inverted_residual_setting,
        last_channel = last_channel,
        weights      = MobileNet_V3_Large_Weights(weights),
        out_indices  = out_indices,
        *args, **kwargs
    )


@BACKBONES.register(name="mobilenet_v3_small")
def mobilenet_v3_small(
    weights    : WeightsEnum | str | None = MobileNet_V3_Small_Weights.DEFAULT,
    out_indices: list | None = None,
    *args, **kwargs
):
    """Create a MobileNetV3 Small backbone.

    Args:
        weights: Pre-trained weights to load. Defaults to
            MobileNet_V3_Small_Weights.DEFAULT.
        out_indices: List of layer indices to extract features from. Defaults
            to None.
        args: Additional positional arguments for the ResNet model.
        kwargs: Additional keyword arguments for the ResNet model.

    Returns:
        A MobileNetV3 Small backbone model.
    """
    inverted_residual_setting, last_channel = _mobilenet_v3_conf("mobilenet_v3_small", **kwargs)
    return MobileNetV3BackBone(
        name         = "mobilenet_v3_small",
        inverted_residual_setting = inverted_residual_setting,
        last_channel = last_channel,
        weights      = MobileNet_V3_Small_Weights(weights),
        out_indices  = out_indices,
        *args, **kwargs
    )

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    model_ = mobilenet_v3_large(weights="default")
    x = torch.ones(1, 3, 224, 224)
    y = model_(x)
    print(model_.features)
    print(x)
    print(y)

# endregion

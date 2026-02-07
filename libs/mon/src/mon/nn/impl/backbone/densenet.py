#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""DenseNet backbones.

This module provides various VGG backbones using PyTorch.
"""

from __future__ import annotations

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

from mon.core import BACKBONES, log, MLType, Path, Task, WEIGHTS, ZOO_DIR
from mon.core.types import Weights, WeightsEnum, WeightsType
from ...base import RegistrableMixin

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region BASE CLASSES & MIXINS
# ==============================================================================

# --- Base Classes ---

class DenseNetBackBone(nn.Module, RegistrableMixin):
    """DenseNet backbone."""

    arch: str = "densenet"
    name: str = None
    tasks: list[Task] = [Task.BACKBONE]
    mltypes: list[MLType] = []
    model_dir: Path = current_dir

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        name: str,
        growth_rate: int,
        block_config: tuple[int, int, int, int],
        num_init_features: int,
        weights: WeightsType | None = None,
        out_indices: list[int] | None = None,
        verbose: bool = True,
        *args, **kwargs
    ):
        """Initialize a new instance.

        Args:
            name (str): Name of the model variant.
            growth_rate (int): How many filters to add each layer (`k` in paper).
            block_config (tuple[int, int, int, int]): How many layers in each
                pooling block.
            num_init_features (int): The number of filters to learn in the
                first convolution layer.
            weights (WeightsType, optional): Pre-trained weights to load.
                Defaults to None.
            out_indices (list[int], optional): List of layer indices to extract
                features from. If None, defaults to [3, 5, 7, 11].
            verbose (bool): Verbosity mode. Defaults to True.
        """
        # Satisfy PyTorch's empty signature first.
        super().__init__()
        # Initialize RegistrableMixin
        RegistrableMixin.__init__(self, name=name)

        # Assign attributes
        self.verbose = verbose

        # Load the base model
        if isinstance(weights, WeightsType):
            kwargs["num_classes"] = weights.num_classes

        base_model = DenseNet(
            growth_rate=growth_rate,
            block_config=block_config,
            num_init_features=num_init_features,
            *args, **kwargs,
        )

        if isinstance(weights, WeightsType):
            base_model.load_state_dict(weights.state_dict())
            if self.verbose:
                log(f"Initialized '{name}' from weights: '{weights.path}'.")
        else:
            if self.verbose:
                log(f"Initialized '{name}' from scratch.")

        # In torchvision, DenseNet already has a 'features' block
        self.features = base_model.features
        self.out_indices = out_indices or [3, 5, 7, 11]
        self.out_channels = self._get_out_channels(variant=name)

    def _get_out_channels(self, variant: str) -> list[int]:
        """Get the number of output channels for each layer."""
        mapping = {
            "densenet121": [64, 256, 512, 1024],
            "densenet161": [96, 384, 768, 2208],
            "densenet169": [64, 256, 512, 1664],
            "densenet201": [64, 256, 512, 1920],
        }
        return mapping.get(variant, [64, 256, 512, 1024])

    # --- Callable & Context Manager ---
    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Forward the input through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W) and values
                ranging from 0.0 to 1.0.

        Returns:
            list[torch.Tensor]: List of feature maps from the specified layers.
        """
        # If you need multiscale features for a Neck (FPN):
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

@WEIGHTS.register(name="densenet121")
class DenseNet121_Weights(WeightsEnum):

    IMAGENET1K_V1 = Weights(
        path=ZOO_DIR / "nn/backbone/densenet/densenet121/imagenet1k_v1/densenet121_imagenet1k_v1.pth",
        url="https://download.pytorch.org/models/densenet121-a639ec97.pth",
        num_classes=1000,
        transforms=None,
        meta={
            "num_params": 7978856,
            "min_size": (29, 29),
            "categories": _IMAGENET_CATEGORIES,
            "recipe": "https://github.com/pytorch/vision/pull/116",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 74.434,
                    "acc@5": 91.972,
                },
            },
            "_ops": 2.834,
            "_file_size": 30.845,
            "_docs": """These weights are ported from LuaTorch.""",
        },
    )
    DEFAULT = IMAGENET1K_V1


@WEIGHTS.register(name="densenet161")
class DenseNet161_Weights(WeightsEnum):

    IMAGENET1K_V1 = Weights(
        path=ZOO_DIR / "nn/backbone/densenet/densenet161/imagenet1k_v1/densenet161_imagenet1k_v1.pth",
        url="https://download.pytorch.org/models/densenet161-8d451a50.pth",
        num_classes=1000,
        transforms=None,
        meta={
            "num_params": 28681000,
            "min_size": (29, 29),
            "categories": _IMAGENET_CATEGORIES,
            "recipe": "https://github.com/pytorch/vision/pull/116",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 77.138,
                    "acc@5": 93.560,
                },
            },
            "_ops": 7.728,
            "_file_size": 110.369,
            "_docs": """These weights are ported from LuaTorch.""",
        },
    )
    DEFAULT = IMAGENET1K_V1


@WEIGHTS.register(name="densenet169")
class DenseNet169_Weights(WeightsEnum):

    IMAGENET1K_V1 = Weights(
        path=ZOO_DIR / "nn/backbone/densenet/densenet169/imagenet1k_v1/densenet169_imagenet1k_v1.pth",
        url="https://download.pytorch.org/models/densenet169-b2777c0a.pth",
        num_classes=1000,
        transforms=None,
        meta={
            "num_params": 14149480,
            "min_size": (29, 29),
            "categories": _IMAGENET_CATEGORIES,
            "recipe": "https://github.com/pytorch/vision/pull/116",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 75.600,
                    "acc@5": 92.806,
                },
            },
            "_ops": 3.36,
            "_file_size": 54.708,
            "_docs": """These weights are ported from LuaTorch.""",
        },
    )
    DEFAULT = IMAGENET1K_V1


@WEIGHTS.register(name="densenet201")
class DenseNet201_Weights(WeightsEnum):

    IMAGENET1K_V1 = Weights(
        path=ZOO_DIR / "nn/backbone/densenet/densenet201/imagenet1k_v1/densenet201_imagenet1k_v1.pth",
        url="https://download.pytorch.org/models/densenet201-c1103571.pth",
        num_classes=1000,
        transforms=None,
        meta={
            "num_params": 20013928,
            "min_size": (29, 29),
            "categories": _IMAGENET_CATEGORIES,
            "recipe": "https://github.com/pytorch/vision/pull/116",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 76.896,
                    "acc@5": 93.370,
                },
            },
            "_ops": 4.291,
            "_file_size": 77.373,
            "_docs": """These weights are ported from LuaTorch.""",
        },
    )
    DEFAULT = IMAGENET1K_V1


# --- Model Variants ---

@BACKBONES.register(name="densenet121", metaclass=DenseNetBackBone)
def densenet121(
    weights: WeightsEnum | str = DenseNet121_Weights.DEFAULT,
    out_indices: list[int] | None = None,
    *args, **kwargs,
) -> DenseNetBackBone:
    """Create a DenseNet-121 backbone.

    Args:
        weights (WeightsEnum | str): Pre-trained weights to load.
            Defaults to DenseNet121_Weights.DEFAULT.
        out_indices (list[int], optional): List of layer indices to extract
            features from. Defaults to None.
    """
    return DenseNetBackBone(
        name="densenet121",
        growth_rate=32,
        block_config=(6, 12, 24, 16),
        num_init_features=64,
        weights=DenseNet121_Weights(weights),
        out_indices=out_indices,
        *args, **kwargs,
    )


@BACKBONES.register(name="densenet161", metaclass=DenseNetBackBone)
def densenet161(
    weights: WeightsEnum | str = DenseNet161_Weights.DEFAULT,
    out_indices: list[int] | None = None,
    *args, **kwargs,
) -> DenseNetBackBone:
    """Create a DenseNet-161 backbone.

    Args:
        weights (WeightsEnum | str): Pre-trained weights to load.
            Defaults to DenseNet161_Weights.DEFAULT.
        out_indices (list[int], optional): List of layer indices to extract
            features from. Defaults to None.
    """
    return DenseNetBackBone(
        name="densenet161",
        growth_rate=48,
        block_config=(6, 12, 36, 24),
        num_init_features=96,
        weights=DenseNet161_Weights(weights),
        out_indices=out_indices,
        *args, **kwargs,
    )


@BACKBONES.register(name="densenet169", metaclass=DenseNetBackBone)
def densenet169(
    weights: WeightsEnum | str = DenseNet169_Weights.DEFAULT,
    out_indices: list[int] | None = None,
    *args, **kwargs,
) -> DenseNetBackBone:
    """Create a DenseNet-169 backbone.

    Args:
        weights (WeightsEnum | str): Pre-trained weights to load.
            Defaults to DenseNet169_Weights.DEFAULT.
        out_indices (list[int], optional): List of layer indices to extract
            features from. Defaults to None.
    """
    return DenseNetBackBone(
        name="densenet169",
        growth_rate=32,
        block_config=(6, 12, 32, 32),
        num_init_features=64,
        weights=DenseNet169_Weights(weights),
        out_indices=out_indices,
        *args, **kwargs,
    )


@BACKBONES.register(name="densenet201", metaclass=DenseNetBackBone)
def densenet201(
    weights: WeightsEnum | str = DenseNet201_Weights.DEFAULT,
    out_indices: list[int] | None = None,
    *args, **kwargs,
) -> DenseNetBackBone:
    """Create a DenseNet-201 backbone.

    Args:
        weights (WeightsEnum | str): Pre-trained weights to load.
            Defaults to DenseNet201_Weights.DEFAULT.
        out_indices (list[int], optional): List of layer indices to extract
            features from. Defaults to None.
    """
    return DenseNetBackBone(
        name="densenet201",
        growth_rate=32,
        block_config=(6, 12, 48, 32),
        num_init_features=64,
        weights=DenseNet201_Weights(weights),
        out_indices=out_indices,
        *args, **kwargs,
    )


# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    model_ = densenet121(weights="default")
    x = torch.ones(1, 3, 224, 224)
    y = model_(x)
    print(model_.features)
    print(x)
    print(y)

# endregion

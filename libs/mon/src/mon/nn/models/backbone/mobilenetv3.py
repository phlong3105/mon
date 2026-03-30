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

from torch import nn, Tensor
from torchvision.models._meta import _IMAGENET_CATEGORIES
from torchvision.models.mobilenetv3 import (
    _mobilenet_v3_conf,
    InvertedResidualConfig,
    MobileNetV3,
)

from mon.core import (
    BACKBONES,
    is_weights_type,
    K,
    log,
    Path,
    Task,
    WEIGHTS,
    Weights,
    WeightsEnum,
    WeightsLike,
)
from mon.nn.models.base import ModelRegisterMixin

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region BASE CLASSES
# ==============================================================================

class MobileNetV3BackBone(ModelRegisterMixin, nn.Module):
    """MobileNetV3 backbone."""

    arch: str = "mobilenet"
    name: str = "mobilenet"
    tasks: list[Task] = [Task.BACKBONE]
    model_dir: Path = current_dir

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        name: str,
        inverted_residual_setting: list[InvertedResidualConfig],
        last_channel: int,
        weights: WeightsLike | None = None,
        out_indices: list[int] | None = None,
        verbose: bool = True,
        *args, **kwargs,
    ):
        """Initialize a new instance.

        Args:
            name (str): Name of the model variant.
            inverted_residual_setting: Network structure configuration.
            last_channel (int): Number of output channels for the last layer.
            weights (WeightsLike, optional): Pre-trained weights to load.
                Defaults to None.
            out_indices (list[int], optional): List of layer indices to extract
                features from.
            verbose (bool): Verbosity mode. Defaults to True.
        """
        # Satisfy PyTorch's empty signature first.
        super().__init__(name=name)
        # Initialize RegistrableMixin
        # ModelRegisterMixin.__init__(self, name=name)

        # Assign attributes
        self.verbose = verbose

        # Define model
        if is_weights_type(weights):
            kwargs["num_classes"] = weights.num_classes or kwargs["num_classes"]

        base_model = MobileNetV3(
            inverted_residual_setting=inverted_residual_setting,
            last_channel=last_channel,
            *args, **kwargs,
        )

        # Load weights
        if is_weights_type(weights):
            base_model.load_state_dict(weights.state_dict())
            if self.verbose:
                log(f"Initialized '{name}' from weights: '{weights.path}'.")
        else:
            if self.verbose:
                log(f"Initialized '{name}' from scratch.")

        # In torchvision, MobileNetV3 already has a 'features' block
        self.features = base_model.features

        if "large" in name:
            self.out_indices = out_indices or [3, 6, 12, 15]
            self.out_channels = [24, 40, 112, 160]
        else:  # Small variant
            self.out_indices = out_indices or [0, 3, 8, 11]
            self.out_channels = [16, 24, 48, 96]

    # --- Callable & Context Manager ---
    def forward(self, x: Tensor) -> list[Tensor]:
        """Forward the input through the network.

        Args:
            x (Tensor): Input tensor of shape (B, C, H, W) and values ranging
                from 0.0 to 1.0.

        Returns:
            list[Tensor]: List of feature maps from the specified layers.
        """
        # If you need multiscale features for a Neck (FPN):
        outputs = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.out_indices:
                outputs.append(x)
        return outputs

# endregion


# ==============================================================================
# region CONCRETE IMPLEMENTATIONS
# ==============================================================================

# --- Pre-trained Weights ---

@WEIGHTS.register(name="mobilenet_v3_large")
class MobileNet_V3_Large_Weights(WeightsEnum):

    IMAGENET1K_V1 = Weights(
        path=K.ZOO_ROOT / "backbone/mobilenet/mobilenet_v3_large/imagenet1k_v1/mobilenet_v3_large_imagenet1k_v1.pt",
        url=Path("https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth"),
        num_classes=1000,
        transforms=None,
        meta={
            "num_params": 5483032,
            "min_size": (1, 1),
            "categories": _IMAGENET_CATEGORIES,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#mobilenetv3-large--small",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 74.042,
                    "acc@5": 91.340,
                },
            },
            "_ops": 0.217,
            "_file_size": 21.114,
            "_docs": """These weights were trained from scratch by using a simple training recipe.""",
        },
    )
    IMAGENET1K_V2 = Weights(
        path=K.ZOO_ROOT / "backbone/mobilenet/mobilenet_v3_large/imagenet1k_v2/mobilenet_v3_large_imagenet1k_v2.pt",
        url=Path("https://download.pytorch.org/models/mobilenet_v3_large-5c1a4163.pth"),
        num_classes=1000,
        transforms=None,
        meta={
            "num_params": 5483032,
            "min_size": (1, 1),
            "categories": _IMAGENET_CATEGORIES,
            "recipe": "https://github.com/pytorch/vision/issues/3995#new-recipe-with-reg-tuning",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 75.274,
                    "acc@5": 92.566,
                },
            },
            "_ops": 0.217,
            "_file_size": 21.107,
            "_docs": """
                These weights improve marginally upon the results of the original paper by using a modified version of
                TorchVision's `new training recipe
                <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
            """,
        },
    )
    DEFAULT = IMAGENET1K_V2


@WEIGHTS.register(name="mobilenet_v3_small")
class MobileNet_V3_Small_Weights(WeightsEnum):

    IMAGENET1K_V1 = Weights(
        path=K.ZOO_ROOT / "backbone/mobilenet/mobilenet_v3_small/imagenet1k_v1/mobilenet_v3_small_imagenet1k_v1.pt",
        url=Path("https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth"),
        num_classes=1000,
        transforms=None,
        meta={
            "num_params": 2542856,
            "min_size": (1, 1),
            "categories": _IMAGENET_CATEGORIES,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#mobilenetv3-large--small",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 67.668,
                    "acc@5": 87.402,
                },
            },
            "_ops": 0.057,
            "_file_size": 9.829,
            "_docs": """These weights improve upon the results of the original paper by using a simple training recipe.""",
        },
    )
    DEFAULT = IMAGENET1K_V1


# --- Model Variants ---

@BACKBONES.register(name="mobilenet_v3_large", metaclass=MobileNetV3BackBone)
def mobilenet_v3_large(
    weights: WeightsLike = "default",
    out_indices: list[int] | None = None,
    *args, **kwargs,
) -> MobileNetV3BackBone:
    """Create a MobileNetV3-Large backbone.

    Args:
        weights (WeightsLike, optional): Pre-trained weights to load.
            Defaults to "default".
        out_indices (list[int], optional): List of layer indices to extract
            features from. Defaults to None.
    """
    inverted_residual_setting, last_channel = _mobilenet_v3_conf(
        arch="mobilenet_v3_large",
        **kwargs
    )
    return MobileNetV3BackBone(
        name="mobilenet_v3_large",
        inverted_residual_setting=inverted_residual_setting,
        last_channel=last_channel,
        weights=MobileNet_V3_Large_Weights(weights),
        out_indices=out_indices,
        *args, **kwargs,
    )


@BACKBONES.register(name="mobilenet_v3_small", metaclass=MobileNetV3BackBone)
def mobilenet_v3_small(
    weights: WeightsLike = "default",
    out_indices: list[int] | None = None,
    *args, **kwargs,
) -> MobileNetV3BackBone:
    """Create a MobileNetV3 Small backbone.

    Args:
        weights (WeightsLike, optional): Pre-trained weights to load.
            Defaults to "default".
        out_indices (list[int], optional): List of layer indices to extract
            features from. Defaults to None.
    """
    inverted_residual_setting, last_channel = _mobilenet_v3_conf(
        arch="mobilenet_v3_small",
        **kwargs
    )
    return MobileNetV3BackBone(
        name="mobilenet_v3_small",
        inverted_residual_setting=inverted_residual_setting,
        last_channel=last_channel,
        weights=MobileNet_V3_Small_Weights(weights),
        out_indices=out_indices,
        *args, **kwargs,
    )

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion

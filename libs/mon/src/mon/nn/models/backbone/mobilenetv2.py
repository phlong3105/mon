#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""MobileNetV2 Backbones.

This module provides various MobileNetV2 backbones using PyTorch.
"""

from __future__ import annotations

__all__ = [
    "MobileNet_V2_Weights",
    "mobilenet_v2",
]

from torch import nn, Tensor
from torchvision.models._meta import _IMAGENET_CATEGORIES
from torchvision.models.mobilenetv2 import MobileNetV2

from mon.core import (
    BACKBONES,
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

class MobileNetV2BackBone(ModelRegisterMixin, nn.Module):
    """MobileNetV2 backbone."""

    arch: str = "mobilenet"
    name: str = "mobilenet"
    tasks: list[Task] = [Task.BACKBONE]
    model_dir: Path = current_dir

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        name: str,
        weights: WeightsLike | None = None,
        out_indices: list[int] | None = None,
        verbose: bool = True,
        *args, **kwargs,
    ):
        """Initialize a new instance.

        Args:
            name (str): Name of the model variant.
            weights (WeightsLike, optional): Pre-trained weights to load.
                Defaults to None.
            out_indices (list[int], optional): List of layer indices to extract
                features from. If None, defaults to [3, 6, 13, 18].
            verbose (bool): Verbosity mode. Defaults to True.
        """
        super().__init__(name=name)

        # Assign attributes
        self.verbose = verbose

        # Define model
        if isinstance(weights, WeightsEnum):
            kwargs["num_classes"] = weights.num_classes or kwargs["num_classes"]

        base_model = MobileNetV2(*args, **kwargs)

        # Load weights
        if isinstance(weights, WeightsEnum):
            base_model.load_state_dict(weights.state_dict())
            if self.verbose:
                log(f"Initialized '{name}' from weights: '{weights.path}'.")
        else:
            if self.verbose:
                log(f"Initialized '{name}' from scratch.")

        # In torchvision, MobileNetV2 already has a 'features' block
        self.features = base_model.features
        self.out_indices = out_indices or [3, 6, 13, 18]
        self.out_channels = [24, 32, 96, 1280]

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

@WEIGHTS.register(name="mobilenet_v2")
class MobileNet_V2_Weights(WeightsEnum):

    IMAGENET1K_V1 = Weights(
        path=K.ZOO_ROOT / "backbone/mobilenet/mobilenet_v2/imagenet1k_v1/mobilenet_v2_imagenet1k_v1.pt",
        url=Path("https://download.pytorch.org/models/mobilenet_v2-b0353104.pth"),
        num_classes=1000,
        transforms=None,
        meta={
            "num_params": 3504872,
            "min_size": (1, 1),
            "categories": _IMAGENET_CATEGORIES,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#mobilenetv2",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 71.878,
                    "acc@5": 90.286,
                },
            },
            "_ops": 0.301,
            "_file_size": 13.555,
            "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
        },
    )
    IMAGENET1K_V2 = Weights(
        path=K.ZOO_ROOT / "backbone/mobilenet/mobilenet_v2/imagenet1k_v2/mobilenet_v2_imagenet1k_v2.pt",
        url=Path("https://download.pytorch.org/models/mobilenet_v2-7ebf99e0.pth"),
        num_classes=1000,
        transforms=None,
        meta={
            "num_params": 3504872,
            "min_size": (1, 1),
            "categories": _IMAGENET_CATEGORIES,
            "recipe": "https://github.com/pytorch/vision/issues/3995#new-recipe-with-reg-tuning",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 72.154,
                    "acc@5": 90.822,
                },
            },
            "_ops": 0.301,
            "_file_size": 13.598,
            "_docs": """
                These weights improve upon the results of the original paper by using a modified version of TorchVision's
                `new training recipe
                <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
            """,
        },
    )
    DEFAULT = IMAGENET1K_V2


# --- Model Variants ---

@BACKBONES.register(name="mobilenet_v2", metaclass=MobileNetV2BackBone)
def mobilenet_v2(
    weights: WeightsLike = "default",
    out_indices: list[int] | None = None,
    *args, **kwargs,
) -> MobileNetV2BackBone:
    """Create a MobileNetV2 backbone.

    Args:
        weights (WeightsLike, optional): Pre-trained weights to load.
            Defaults to "default".
        out_indices (list[int], optional): List of layer indices to extract
            features from. Defaults to None.
    """
    return MobileNetV2BackBone(
        name="mobilenet_v2",
        weights=MobileNet_V2_Weights(weights),
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

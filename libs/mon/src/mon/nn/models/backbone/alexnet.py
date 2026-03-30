#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""AlexNet Backbone.

This module provides various AlexNet backbones using PyTorch.
"""

from __future__ import annotations

__all__ = [
    "AlexNet_Weights",
    "alexnet",
]

from torch import nn, Tensor
from torchvision.models._meta import _IMAGENET_CATEGORIES
from torchvision.models.alexnet import AlexNet

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

class AlexNetBackBone(ModelRegisterMixin, nn.Module):
    """AlexNet backbone."""

    arch: str = "alexnet"
    name: str = "alexnet"
    tasks: list[Task] = [Task.BACKBONE]
    model_dir: Path = current_dir

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        name: str,
        weights: WeightsLike | None = None,
        out_indices: list[int] | None = None,
        verbose: bool = True,
        *args, **kwargs
    ):
        """Initialize a new instance.

        Args:
            name (str): Name of the model variant.
            weights (WeightsLike, optional): Pre-trained weights to load.
                Defaults to None.
            out_indices (list[int], optional): List of layer indices to extract
                features from. If None, defaults to [2, 5, 8, 10, 12].
            verbose (bool, optional): Verbosity mode. Defaults to True.
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

        base_model = AlexNet(*args, **kwargs)

        # Load weights
        if is_weights_type(weights):
            base_model.load_state_dict(weights.state_dict())
            if self.verbose:
                log(f"Initialized '{name}' from weights: '{weights.path}'.")
        else:
            if self.verbose:
                log(f"Initialized '{name}' from scratch.")

        # In torchvision, AlexNet already has a 'features' block
        self.features = base_model.features
        self.out_indices = out_indices or [2, 5, 8, 10, 12]
        self.out_channels = [64, 192, 384, 256, 256]

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

@WEIGHTS.register(name="alexnet")
class AlexNet_Weights(WeightsEnum):

    IMAGENET1K_V1 = Weights(
        path=K.ZOO_ROOT / "backbone/alexnet/alexnet/imagenet1k_v1/alexnet_imagenet1k_v1.pt",
        url=Path("https://download.pytorch.org/models/alexnet-owt-7be5be79.pth"),
        num_classes=1000,
        transforms=None,
        meta={
            "num_params": 61100840,
            "min_size": (63, 63),
            "categories": _IMAGENET_CATEGORIES,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#alexnet-and-vgg",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 56.522,
                    "acc@5": 79.066,
                }
            },
            "_ops": 0.714,
            "_file_size": 233.087,
            "_docs": """These weights reproduce closely the results of the paper using a simplified training recipe.""",
        }
    )
    DEFAULT = IMAGENET1K_V1


# --- Model Variants ---

@BACKBONES.register(name="alexnet", metaclass=AlexNetBackBone)
def alexnet(
    weights: WeightsLike = "default",
    out_indices: list[int] | None = None,
    *args, **kwargs
) -> AlexNetBackBone:
    """Create an AlexNet backbone.

    Args:
        weights (WeightsLike, optional): Pre-trained weights to load.
            Defaults to "default".
        out_indices (list[int], optional): List of layer indices to extract
            features from. Defaults to None.
    """
    return AlexNetBackBone(
        name="alexnet",
        weights=AlexNet_Weights(weights),
        out_indices=out_indices,
        *args, **kwargs
    )

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion

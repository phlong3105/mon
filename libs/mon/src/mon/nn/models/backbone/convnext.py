#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""ConvNeXt Backbones.

This module provides various ConvNeXt backbones using PyTorch.
"""

from __future__ import annotations

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

from torch import nn, Tensor
from torchvision.models._meta import _IMAGENET_CATEGORIES
from torchvision.models.convnext import CNBlockConfig, ConvNeXt

from mon.core import (
    BACKBONES,
    is_weights_type,
    K,
    log,
    Path,
    Strategy,
    Task,
    WEIGHTS,
    Weights,
    WeightsEnum,
    WeightsLike,
)
from mon.nn.models.base import ModelRegisterMixin

current_file = Path(__file__).absolute()
current_dir = current_file.parents[0]


# ==============================================================================
# region BASE CLASSES
# ==============================================================================

class ConvNeXtBackBone(ModelRegisterMixin, nn.Module):
    """ConvNeXt backbone."""

    arch: str = "convnext"
    name: str = "convnext"
    tasks: list[Task] = [Task.BACKBONE]
    strategies: list[Strategy] = [Strategy.RESIZE]
    model_dir: Path = current_dir

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        name: str,
        block_setting: list[CNBlockConfig],
        stochastic_depth_prob: float,
        weights: Weights | None = None,
        out_indices: list[int] | None = None,
        verbose: bool = True,
        *args, **kwargs
    ):
        """Initialize a new instance.

        Args:
            name (str): Name of the model variant.
            block_setting (list[CNBlockConfig]): List of block settings for the
                ConvNeXt model.
            stochastic_depth_prob (float): Probability of applying stochastic
                depth to the model.
            weights (Weights | None, optional): Pre-trained weights to load.
                Defaults to None.
            out_indices (list[int] | None, optional): List of layer indices to
                extract features from. If None, defaults to [1, 3, 5, 7].
            verbose (bool, optional): Verbosity mode. Defaults to True.
        """
        super().__init__(name=name)

        # Assign attributes
        self.verbose = verbose

        # Define model
        if weights is not None and is_weights_type(weights):
            kwargs["num_classes"] = weights.num_classes or kwargs["num_classes"]

        base_model = ConvNeXt(
            block_setting=block_setting,
            stochastic_depth_prob=stochastic_depth_prob,
            *args, **kwargs
        )

        # Load weights
        if weights is not None and is_weights_type(weights):
            base_model.load_state_dict(weights.state_dict())
            if self.verbose:
                log(f"Initialized '{name}' from weights: '{weights.path.as_posix()}'.")
        else:
            if self.verbose:
                log(f"Initialized '{name}' from scratch.")

        # In torchvision, ConvNeXt already has a 'features' block
        self.features = base_model.features
        self.out_indices = out_indices or [1, 3, 5, 7]
        self.out_channels = self._get_out_channels(variant=name)

    def _get_out_channels(self, variant: str) -> list[int]:
        """Get the number of output channels for each layer."""
        mapping = {
            "convnext_tiny": [96, 192, 384, 768],
            "convnext_small": [96, 192, 384, 768],
            "convnext_base": [128, 256, 512, 1024],
            "convnext_large": [192, 384, 768, 1536],
        }
        return mapping.get(variant, [96, 192, 384, 768])

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

@WEIGHTS.register(name="convnext_tiny")
class ConvNeXt_Tiny_Weights(WeightsEnum):

    IMAGENET1K_V1 = Weights(
        path=K.ZOO_ROOT / "backbone/convnext/convnext_tiny/imagenet1k_v1/convnext_tiny_imagenet1k_v1.pt",
        url=Path("https://download.pytorch.org/models/convnext_tiny-983f1562.pth"),
        num_classes=1000,
        transforms=None,
        meta={
            "num_params": 28589128,
            "min_size": (32, 32),
            "categories": _IMAGENET_CATEGORIES,
            "recipe" : "https://github.com/pytorch/vision/tree/main/references/classification#convnext",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 82.520,
                    "acc@5": 96.146,
                }
            },
            "_ops": 4.456,
            "_file_size": 109.119,
            "_docs": """
                These weights improve upon the results of the original paper by using a modified version of TorchVision's
                `new training recipe
                <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
            """,
        }
    )
    DEFAULT = IMAGENET1K_V1


@WEIGHTS.register(name="convnext_small")
class ConvNeXt_Small_Weights(WeightsEnum):

    IMAGENET1K_V1 = Weights(
        path=K.ZOO_ROOT / "backbone/convnext/convnext_small/imagenet1k_v1/convnext_small_imagenet1k_v1.pt",
        url=Path("https://download.pytorch.org/models/convnext_small-0c510722.pth"),
        num_classes=1000,
        transforms=None,
        meta={
            "num_params": 50223688,
            "min_size": (32, 32),
            "categories": _IMAGENET_CATEGORIES,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#convnext",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 83.616,
                    "acc@5": 96.650,
                }
            },
            "_ops": 8.684,
            "_file_size": 191.703,
            "_docs": """
                These weights improve upon the results of the original paper by using a modified version of TorchVision's
                `new training recipe
                <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
            """,
        }
    )
    DEFAULT = IMAGENET1K_V1


@WEIGHTS.register(name="convnext_base")
class ConvNeXt_Base_Weights(WeightsEnum):

    IMAGENET1K_V1 = Weights(
        path=K.ZOO_ROOT / "backbone/convnext/convnext_base/imagenet1k_v1/convnext_base_imagenet1k_v1.pt",
        url=Path("https://download.pytorch.org/models/convnext_base-6075fbad.pth"),
        num_classes=1000,
        transforms=None,
        meta={
            "num_params": 88591464,
            "min_size": (32, 32),
            "categories": _IMAGENET_CATEGORIES,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#convnext",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 84.062,
                    "acc@5": 96.870,
                }
            },
            "_ops": 15.355,
            "_file_size": 338.064,
            "_docs": """
                These weights improve upon the results of the original paper by using a modified version of TorchVision's
                `new training recipe
                <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
            """,
        }
    )
    DEFAULT = IMAGENET1K_V1


@WEIGHTS.register(name="convnext_large")
class ConvNeXt_Large_Weights(WeightsEnum):

    IMAGENET1K_V1 = Weights(
        path=K.ZOO_ROOT / "backbone/convnext/convnext_large/imagenet1k_v1/convnext_large_imagenet1k_v1.pt",
        url=Path("https://download.pytorch.org/models/convnext_large-ea097f82.pth"),
        num_classes=1000,
        transforms=None,
        meta={
            "num_params": 197767336,
            "min_size": (32, 32),
            "categories": _IMAGENET_CATEGORIES,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#convnext",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 84.414,
                    "acc@5": 96.976,
                }
            },
            "_ops": 34.361,
            "_file_size": 754.537,
            "_docs": """
                These weights improve upon the results of the original paper by using a modified version of TorchVision's
                `new training recipe
                <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
            """,
        }
    )
    DEFAULT = IMAGENET1K_V1


# --- Model Variants ---

@BACKBONES.register(name="convnext_tiny", metaclass=ConvNeXtBackBone)
def convnext_tiny(
    weights: WeightsLike = "default",
    out_indices: list[int] | None = None,
    *args, **kwargs
) -> ConvNeXtBackBone:
    """Create a ConvNeXt-Tiny backbone.

    Args:
         weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
        out_indices (list[int] | None, optional): List of layer indices to
            extract features from. Defaults to None.
    """
    block_setting = [
        CNBlockConfig(96, 192, 3),
        CNBlockConfig(192, 384, 3),
        CNBlockConfig(384, 768, 9),
        CNBlockConfig(768, None, 3),
    ]
    stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.1)
    return ConvNeXtBackBone(
        name="convnext_tiny",
        block_setting=block_setting,
        stochastic_depth_prob=stochastic_depth_prob,
        weights=ConvNeXt_Tiny_Weights(weights),
        out_indices=out_indices,
        *args, **kwargs,
    )


@BACKBONES.register(name="convnext_small", metaclass=ConvNeXtBackBone)
def convnext_small(
    weights: WeightsLike = "default",
    out_indices: list[int] | None = None,
    *args, **kwargs
) -> ConvNeXtBackBone:
    """Create a ConvNeXt-Small backbone.

    Args:
         weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
        out_indices (list[int] | None, optional): List of layer indices to
            extract features from. Defaults to None.
    """
    block_setting = [
        CNBlockConfig(96, 192, 3),
        CNBlockConfig(192, 384, 3),
        CNBlockConfig(384, 768, 27),
        CNBlockConfig(768, None, 3),
    ]
    stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.4)
    return ConvNeXtBackBone(
        name="convnext_small",
        block_setting=block_setting,
        stochastic_depth_prob=stochastic_depth_prob,
        weights=ConvNeXt_Small_Weights(weights),
        out_indices=out_indices,
        *args, **kwargs
    )


@BACKBONES.register(name="convnext_base", metaclass=ConvNeXtBackBone)
def convnext_base(
    weights: WeightsLike = "default",
    out_indices: list[int] | None = None,
    *args, **kwargs,
) -> ConvNeXtBackBone:
    """Create a ConvNeXt-Base backbone.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
        out_indices (list[int] | None, optional): List of layer indices to
            extract features from. Defaults to None.
    """
    block_setting = [
        CNBlockConfig(128, 256, 3),
        CNBlockConfig(256, 512, 3),
        CNBlockConfig(512, 1024, 27),
        CNBlockConfig(1024, None, 3),
    ]
    stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.5)
    return ConvNeXtBackBone(
        name="convnext_base",
        block_setting=block_setting,
        stochastic_depth_prob=stochastic_depth_prob,
        weights=ConvNeXt_Base_Weights(weights),
        out_indices=out_indices,
        *args, **kwargs
    )


@BACKBONES.register(name="convnext_large", metaclass=ConvNeXtBackBone)
def convnext_large(
    weights: WeightsLike = "default",
    out_indices: list[int] | None = None,
    *args, **kwargs,
) -> ConvNeXtBackBone:
    """Create a ConvNeXt-Large backbone.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
        out_indices (list[int] | None, optional): List of layer indices to
            extract features from. Defaults to None.
    """
    block_setting = [
        CNBlockConfig(192, 384, 3),
        CNBlockConfig(384, 768, 3),
        CNBlockConfig(768, 1536, 27),
        CNBlockConfig(1536, None, 3),
    ]
    stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.5)
    return ConvNeXtBackBone(
        name="convnext_large",
        block_setting=block_setting,
        stochastic_depth_prob=stochastic_depth_prob,
        weights=ConvNeXt_Large_Weights(weights),
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

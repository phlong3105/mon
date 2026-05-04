#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""VGG backbones.

This module provides various VGG backbones using PyTorch.
"""

from __future__ import annotations

__all__ = [
    "VGG11_BN_Weights",
    "VGG11_Weights",
    "VGG13_BN_Weights",
    "VGG13_Weights",
    "VGG16_BN_Weights",
    "VGG16_Weights",
    "VGG19_BN_Weights",
    "VGG19_Weights",
    "vgg11",
    "vgg11_bn",
    "vgg13",
    "vgg13_bn",
    "vgg16",
    "vgg16_bn",
    "vgg19",
    "vgg19_bn",
]

from torch import nn, Tensor
from torchvision.models._meta import _IMAGENET_CATEGORIES
from torchvision.models.vgg import cfgs, make_layers, VGG

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

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region BASE CLASSES
# ==============================================================================

class VGGBackBone(ModelRegisterMixin, nn.Module):
    """VGG backbone."""

    arch: str = "vgg"
    name: str = "vgg"
    tasks: list[Task] = [Task.BACKBONE]
    strategies: list[Strategy] = [Strategy.RESIZE]
    model_dir: Path = current_dir

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        name: str,
        cfg: str,
        batch_norm: bool,
        weights: Weights | None = None,
        out_indices: list[int] | None = None,
        verbose: bool = True,
        *args, **kwargs,
    ):
        """Initialize a new instance.

        Args:
            name (str): Name of the model variant.
            cfg (str): Configuration string, e.g. 'A'.
            batch_norm (bool): Whether to use batch normalization.
            weights (Weights | None, optional): Pre-trained weights to load.
                Defaults to None.
            out_indices (list[int] | None, optional): List of layer indices to
                extract features from. If None, defaults to [6, 13, 23, 33, 43].
            verbose (bool, optional): Verbosity mode. Defaults to True.
        """
        super().__init__(name=name)

        # Assign attributes
        self.verbose = verbose

        # Define model
        if weights is not None and is_weights_type(weights):
            kwargs["init_weights"] = False
            kwargs["num_classes"] = weights.num_classes or kwargs["num_classes"]

        base_model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), *args, **kwargs)

        # Load weights
        if weights is not None and is_weights_type(weights):
            base_model.load_state_dict(weights.state_dict())
            if self.verbose:
                log(f"Initialized '{name}' from weights: '{weights.path}'.")
        else:
            if self.verbose:
                log(f"Initialized '{name}' from scratch.")

        # In torchvision, VGG already has a 'features' block
        self.features = base_model.features
        self.out_indices = out_indices or [6, 13, 23, 33, 43]
        self.out_channels = [64, 128, 256, 512, 512]

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

@WEIGHTS.register(name="vgg11")
class VGG11_Weights(WeightsEnum):

    IMAGENET1K_V1 = Weights(
        path=K.ZOO_ROOT / "backbone/vgg/vgg11/imagenet1k_v1/vgg11_imagenet1k_v1.pt",
        url=Path("https://download.pytorch.org/models/vgg11-8a719046.pth"),
        num_classes=1000,
        transforms=None,
        meta={
            "num_params": 132863336,
            "min_size": (32, 32),
            "categories": _IMAGENET_CATEGORIES,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#alexnet-and-vgg",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 69.020,
                    "acc@5": 88.628,
                },
            },
            "_ops": 7.609,
            "_file_size": 506.84,
            "_docs": """These weights were trained from scratch by using a simplified training recipe.""",
        },
    )
    DEFAULT = IMAGENET1K_V1


@WEIGHTS.register(name="vgg11_bn")
class VGG11_BN_Weights(WeightsEnum):

    IMAGENET1K_V1 = Weights(
        path=K.ZOO_ROOT / "backbone/vgg/vgg11_bn/imagenet1k_v1/vgg11_bn_imagenet1k_v1.pt",
        url=Path("https://download.pytorch.org/models/vgg11_bn-6002323d.pth"),
        num_classes=1000,
        transforms=None,
        meta={
            "num_params": 132868840,
            "min_size": (32, 32),
            "categories": _IMAGENET_CATEGORIES,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#alexnet-and-vgg",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 70.370,
                    "acc@5": 89.810,
                },
            },
            "_ops": 7.609,
            "_file_size": 506.881,
            "_docs": """These weights were trained from scratch by using a simplified training recipe.""",
        },
    )
    DEFAULT = IMAGENET1K_V1


@WEIGHTS.register(name="vgg13")
class VGG13_Weights(WeightsEnum):

    IMAGENET1K_V1 = Weights(
        path=K.ZOO_ROOT / "backbone/vgg/vgg13/imagenet1k_v1/vgg13_imagenet1k_v1.pt",
        url=Path("https://download.pytorch.org/models/vgg13-19584684.pth"),
        num_classes=1000,
        transforms=None,
        meta={
            "num_params": 133047848,
            "min_size": (32, 32),
            "categories": _IMAGENET_CATEGORIES,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#alexnet-and-vgg",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 69.928,
                    "acc@5": 89.246,
                },
            },
            "_ops": 11.308,
            "_file_size": 507.545,
            "_docs": """These weights were trained from scratch by using a simplified training recipe.""",
        },
    )
    DEFAULT = IMAGENET1K_V1


@WEIGHTS.register(name="vgg13_bn")
class VGG13_BN_Weights(WeightsEnum):

    IMAGENET1K_V1 = Weights(
        path=K.ZOO_ROOT / "backbone/vgg/vgg13_bn/imagenet1k_v1/vgg13_bn_imagenet1k_v1.pt",
        url=Path("https://download.pytorch.org/models/vgg13_bn-abd245e5.pth"),
        num_classes=1000,
        transforms=None,
        meta={
            "num_params": 133053736,
            "min_size": (32, 32),
            "categories": _IMAGENET_CATEGORIES,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#alexnet-and-vgg",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 71.586,
                    "acc@5": 90.374,
                },
            },
            "_ops": 11.308,
            "_file_size": 507.59,
            "_docs": """These weights were trained from scratch by using a simplified training recipe.""",
        },
    )
    DEFAULT = IMAGENET1K_V1


@WEIGHTS.register(name="vgg16")
class VGG16_Weights(WeightsEnum):

    IMAGENET1K_V1 = Weights(
        path=K.ZOO_ROOT / "backbone/vgg/vgg16/imagenet1k_v1/vgg16_imagenet1k_v1.pt",
        url=Path("https://download.pytorch.org/models/vgg16-397923af.pth"),
        num_classes=1000,
        transforms=None,
        meta={
            "num_params": 138357544,
            "min_size": (32, 32),
            "categories": _IMAGENET_CATEGORIES,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#alexnet-and-vgg",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 71.592,
                    "acc@5": 90.382,
                },
            },
            "_ops": 15.47,
            "_file_size": 527.796,
            "_docs": """These weights were trained from scratch by using a simplified training recipe.""",
        },
    )
    DEFAULT = IMAGENET1K_V1


@WEIGHTS.register(name="vgg16_bn")
class VGG16_BN_Weights(WeightsEnum):

    IMAGENET1K_V1 = Weights(
        path=K.ZOO_ROOT / "backbone/vgg/vgg16_bn/imagenet1k_v1/vgg16_bn_imagenet1k_v1.pt",
        url=Path("https://download.pytorch.org/models/vgg16_bn-6c64b313.pth"),
        num_classes=1000,
        transforms=None,
        meta={
            "num_params": 138365992,
            "min_size": (32, 32),
            "categories": _IMAGENET_CATEGORIES,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#alexnet-and-vgg",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 73.360,
                    "acc@5": 91.516,
                },
            },
            "_ops": 15.47,
            "_file_size": 527.866,
            "_docs": """These weights were trained from scratch by using a simplified training recipe.""",
        },
    )
    DEFAULT = IMAGENET1K_V1


@WEIGHTS.register(name="vgg19")
class VGG19_Weights(WeightsEnum):

    IMAGENET1K_V1 = Weights(
        path=K.ZOO_ROOT / "backbone/vgg/vgg19/imagenet1k_v1/vgg19_imagenet1k_v1.pt",
        url=Path("https://download.pytorch.org/models/vgg19-dcbb9e9d.pth"),
        num_classes=1000,
        transforms=None,
        meta={
            "num_params": 143667240,
            "min_size": (32, 32),
            "categories": _IMAGENET_CATEGORIES,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#alexnet-and-vgg",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 72.376,
                    "acc@5": 90.876,
                },
            },
            "_ops": 19.632,
            "_file_size": 548.051,
            "_docs": """These weights were trained from scratch by using a simplified training recipe.""",
        },
    )
    DEFAULT = IMAGENET1K_V1


@WEIGHTS.register(name="vgg19_bn")
class VGG19_BN_Weights(WeightsEnum):

    IMAGENET1K_V1 = Weights(
        path=K.ZOO_ROOT / "backbone/vgg/vgg19_bn/imagenet1k_v1/vgg19_bn_imagenet1k_v1.pt",
        url=Path("https://download.pytorch.org/models/vgg19_bn-c79401a0.pth"),
        num_classes=1000,
        transforms=None,
        meta={
            "num_params": 143678248,
            "min_size": (32, 32),
            "categories": _IMAGENET_CATEGORIES,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#alexnet-and-vgg",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 74.218,
                    "acc@5": 91.842,
                },
            },
            "_ops": 19.632,
            "_file_size": 548.143,
            "_docs": """These weights were trained from scratch by using a simplified training recipe.""",
        },
    )
    DEFAULT = IMAGENET1K_V1


# --- Model Variants ---

@BACKBONES.register(name="vgg11", metaclass=VGGBackBone)
def vgg11(
    weights: WeightsLike = "default",
    out_indices: list[int] | None = None,
    *args, **kwargs,
) -> VGGBackBone:
    """Create a VGG11 backbone.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
        out_indices (list[int] | None, optional): List of layer indices to
            extract features from. Defaults to None.
    """
    return VGGBackBone(
        name="vgg11",
        cfg="A",
        batch_norm=False,
        weights=VGG11_Weights(weights),
        out_indices=out_indices,
        *args, **kwargs,
    )


@BACKBONES.register(name="vgg11_bn", metaclass=VGGBackBone)
def vgg11_bn(
    weights: WeightsLike = "default",
    out_indices: list[int] | None = None,
    *args, **kwargs,
) -> VGGBackBone:
    """Create a VGG11-BN backbone.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
        out_indices (list[int] | None, optional): List of layer indices to
            extract features from. Defaults to None.
    """
    return VGGBackBone(
        name="vgg11_bn",
        cfg="A",
        batch_norm=True,
        weights=VGG11_BN_Weights(weights),
        out_indices=out_indices,
        *args, **kwargs,
    )


@BACKBONES.register(name="vgg13", metaclass=VGGBackBone)
def vgg13(
    weights: WeightsLike = "default",
    out_indices: list[int] | None = None,
    *args, **kwargs,
) -> VGGBackBone:
    """Create a VGG13 backbone.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
        out_indices (list[int] | None, optional): List of layer indices to
            extract features from. Defaults to None.
    """
    return VGGBackBone(
        name="vgg13",
        cfg="B",
        batch_norm=False,
        weights=VGG13_Weights(weights),
        out_indices=out_indices,
        *args, **kwargs,
    )


@BACKBONES.register(name="vgg13_bn", metaclass=VGGBackBone)
def vgg13_bn(
    weights: WeightsLike = "default",
    out_indices: list[int] | None = None,
    *args, **kwargs,
) -> VGGBackBone:
    """Create a VGG13-BN backbone.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
        out_indices (list[int] | None, optional): List of layer indices to
            extract features from. Defaults to None.
    """
    return VGGBackBone(
        name="vgg13_bn",
        cfg="B",
        batch_norm=True,
        weights=VGG13_BN_Weights(weights),
        out_indices=out_indices,
        *args, **kwargs,
    )


@BACKBONES.register(name="vgg16", metaclass=VGGBackBone)
def vgg16(
    weights: WeightsLike = "default",
    out_indices: list[int] | None = None,
    *args, **kwargs,
) -> VGGBackBone:
    """Create a VGG16 backbone.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
        out_indices (list[int] | None, optional): List of layer indices to
            extract features from. Defaults to None.
    """
    return VGGBackBone(
        name="vgg16",
        cfg="D",
        batch_norm=False,
        weights=VGG16_Weights(weights),
        out_indices=out_indices,
        *args, **kwargs,
    )


@BACKBONES.register(name="vgg16_bn", metaclass=VGGBackBone)
def vgg16_bn(
    weights: WeightsLike = "default",
    out_indices: list[int] | None = None,
    *args, **kwargs,
) -> VGGBackBone:
    """Create a VGG16-BN backbone.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
        out_indices (list[int] | None, optional): List of layer indices to
            extract features from. Defaults to None.
    """
    return VGGBackBone(
        name="vgg16_bn",
        cfg="D",
        batch_norm=True,
        weights=VGG16_BN_Weights(weights),
        out_indices=out_indices,
        *args, **kwargs,
    )


@BACKBONES.register(name="vgg19", metaclass=VGGBackBone)
def vgg19(
    weights: WeightsLike = "default",
    out_indices: list[int] | None = None,
    *args, **kwargs,
) -> VGGBackBone:
    """Create a VGG19 backbone.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
        out_indices (list[int] | None, optional): List of layer indices to
            extract features from. Defaults to None.
    """
    return VGGBackBone(
        name="vgg19",
        cfg="E",
        batch_norm=False,
        weights=VGG19_Weights(weights),
        out_indices=out_indices,
        *args, **kwargs,
    )


@BACKBONES.register(name="vgg19_bn", metaclass=VGGBackBone)
def vgg19_bn(
    weights: WeightsLike = "default",
    out_indices: list[int] | None = None,
    *args, **kwargs,
) -> VGGBackBone:
    """Create a VGG19-BN backbone.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
        out_indices (list[int] | None, optional): List of layer indices to
            extract features from. Defaults to None.
    """
    return VGGBackBone(
        name="vgg19_bn",
        cfg="E",
        batch_norm=True,
        weights=VGG19_BN_Weights(weights),
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

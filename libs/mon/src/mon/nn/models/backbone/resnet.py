#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""ResNet Backbones.

This module provides various ResNet backbones using PyTorch.
"""

from __future__ import annotations

__all__ = [
    "ResNeXt101_32X8D_Weights",
    "ResNeXt101_64X4D_Weights",
    "ResNeXt50_32X4D_Weights",
    "ResNet101_Weights",
    "ResNet152_Weights",
    "ResNet18_Weights",
    "ResNet34_Weights",
    "ResNet50_Weights",
    "Wide_ResNet101_2_Weights",
    "Wide_ResNet50_2_Weights",
    "resnet101",
    "resnet152",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnext101_32x8d",
    "resnext101_64x4d",
    "resnext50_32x4d",
    "wide_resnet101_2",
    "wide_resnet50_2",
]

from typing import Union

from torch import nn, Tensor
from torchvision.models._meta import _IMAGENET_CATEGORIES
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet

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

class ResNetBackBone(ModelRegisterMixin, nn.Module):
    """ResNet backbone.

    References:
        - Paper: https://arxiv.org/abs/1512.03385
    """

    arch: str = "resnet"
    name: str = "resnet"
    tasks: list[Task] = [Task.BACKBONE]
    strategies: list[Strategy] = [Strategy.RESIZE]
    model_dir: Path = current_dir

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        name: str,
        block: type[Union[BasicBlock, Bottleneck]],
        layers: list[int],
        weights: Weights | None = None,
        out_indices: list[int] | None = None,
        verbose: bool = True,
        *args, **kwargs,
    ):
        """Initialize a new instance.

        Args:
            name (str): Name of the model variant.
            block (BasicBlock | Bottleneck): Type of residual block to use.
            layers (list[int]): Number of residual blocks to include in each stage.
            weights (Weights | None, optional): Pre-trained weights to load.
                Defaults to None.
            out_indices (list[int] | None, optional): List of layer indices to
                extract features from. If None, defaults to [4, 5, 6, 7].
            verbose (bool, optional): Verbosity mode. Defaults to True.
        """
        super().__init__(name=name)

        # Assign attributes
        self.verbose = verbose

        # Define model
        if weights is not None and is_weights_type(weights):
            kwargs["num_classes"] = weights.num_classes or kwargs["num_classes"]

        base_model = ResNet(block=block, layers=layers, *args, **kwargs)

        # Load weights
        if weights is not None and is_weights_type(weights):
            base_model.load_state_dict(weights.state_dict())
            if self.verbose:
                log(f"Initialized '{name}' from weights: '{weights.path.as_posix()}'.")
        else:
            if self.verbose:
                log(f"Initialized '{name}' from scratch.")

        # Remove the global average pool and classifier head
        # For ResNet, we usually want the features before the final layers
        self.features = nn.Sequential(*list(base_model.children())[:-2])
        self.out_indices = out_indices or [4, 5, 6, 7]
        self.out_channels = self._get_out_channels(variant=name)

    def _get_out_channels(self, variant: str) -> list[int]:
        """Get the number of output channels for each stage."""
        # Bottleneck models (50+) have 4x more channels in the output of each stage
        if any(x in variant for x in ["50", "101", "152"]):
            return [256, 512, 1024, 2048]
        return [64, 128, 256, 512]

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

@WEIGHTS.register(name="resnet18")
class ResNet18_Weights(WeightsEnum):

    IMAGENET1K_V1 = Weights(
        path=K.ZOO_ROOT / "backbone/resnet/resnet18/imagenet1k_v1/resnet18_imagenet1k_v1.pt",
        url=Path("https://download.pytorch.org/models/resnet18-f37072fd.pth"),
        num_classes=1000,
        transforms=None,
        meta={
            "num_params": 11689512,
            "min_size": (1, 1),
            "categories": _IMAGENET_CATEGORIES,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#resnet",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 69.758,
                    "acc@5": 89.078,
                },
            },
            "_ops": 1.814,
            "_file_size": 44.661,
            "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
        },
    )
    DEFAULT = IMAGENET1K_V1


@WEIGHTS.register(name="resnet34")
class ResNet34_Weights(WeightsEnum):

    IMAGENET1K_V1 = Weights(
        path=K.ZOO_ROOT / "backbone/resnet/resnet34/imagenet1k_v1/resnet34_imagenet1k_v1.pt",
        url=Path("https://download.pytorch.org/models/resnet34-b627a593.pth"),
        num_classes=1000,
        transforms=None,
        meta={
            "num_params": 21797672,
            "min_size": (1, 1),
            "categories": _IMAGENET_CATEGORIES,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#resnet",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 73.314,
                    "acc@5": 91.420,
                },
            },
            "_ops": 3.664,
            "_file_size": 83.275,
            "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
        },
    )
    DEFAULT = IMAGENET1K_V1


@WEIGHTS.register(name="resnet50")
class ResNet50_Weights(WeightsEnum):

    IMAGENET1K_V1 = Weights(
        path=K.ZOO_ROOT / "backbone/resnet/resnet50/imagenet1k_v1/resnet50_imagenet1k_v1.pt",
        url=Path("https://download.pytorch.org/models/resnet50-0676ba61.pth"),
        num_classes=1000,
        transforms=None,
        meta={
            "num_params": 25557032,
            "min_size": (1, 1),
            "categories": _IMAGENET_CATEGORIES,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#resnet",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 76.130,
                    "acc@5": 92.862,
                },
            },
            "_ops": 4.089,
            "_file_size": 97.781,
            "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
        },
    )
    IMAGENET1K_V2 = Weights(
        path=K.ZOO_ROOT / "backbone/resnet/resnet50/imagenet1k_v2/resnet50_imagenet1k_v2.pt",
        url=Path("https://download.pytorch.org/models/resnet50-11ad3fa6.pth"),
        num_classes=1000,
        transforms=None,
        meta={
            "num_params": 25557032,
            "min_size": (1, 1),
            "categories": _IMAGENET_CATEGORIES,
            "recipe": "https://github.com/pytorch/vision/issues/3995#issuecomment-1013906621",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 80.858,
                    "acc@5": 95.434,
                },
            },
            "_ops": 4.089,
            "_file_size": 97.79,
            "_docs": """
                These weights improve upon the results of the original paper by using TorchVision's `new training recipe
                <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
            """,
        },
    )
    DEFAULT = IMAGENET1K_V2


@WEIGHTS.register(name="resnet101")
class ResNet101_Weights(WeightsEnum):

    IMAGENET1K_V1 = Weights(
        path=K.ZOO_ROOT / "backbone/resnet/resnet101/imagenet1k_v1/resnet101_imagenet1k_v1.pt",
        url=Path("https://download.pytorch.org/models/resnet101-63fe2227.pth"),
        num_classes=1000,
        transforms=None,
        meta={
            "num_params": 44549160,
            "min_size": (1, 1),
            "categories": _IMAGENET_CATEGORIES,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#resnet",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 77.374,
                    "acc@5": 93.546,
                },
            },
            "_ops": 7.801,
            "_file_size": 170.511,
            "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
        },
    )
    IMAGENET1K_V2 = Weights(
        path=K.ZOO_ROOT / "backbone/resnet/resnet101/imagenet1k_v2/resnet101_imagenet1k_v2.pt",
        url=Path("https://download.pytorch.org/models/resnet101-cd907fc2.pth"),
        num_classes=1000,
        transforms=None,
        meta={
            "num_params": 44549160,
            "min_size": (1, 1),
            "categories": _IMAGENET_CATEGORIES,
            "recipe": "https://github.com/pytorch/vision/issues/3995#new-recipe",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 81.886,
                    "acc@5": 95.780,
                },
            },
            "_ops": 7.801,
            "_file_size": 170.53,
            "_docs": """
                These weights improve upon the results of the original paper by using TorchVision's `new training recipe
                <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
            """,
        },
    )
    DEFAULT = IMAGENET1K_V2


@WEIGHTS.register(name="resnet152")
class ResNet152_Weights(WeightsEnum):

    IMAGENET1K_V1 = Weights(
        path=K.ZOO_ROOT / "backbone/resnet/resnet152/imagenet1k_v1/resnet152_imagenet1k_v1.pt",
        url=Path("https://download.pytorch.org/models/resnet152-394f9c45.pth"),
        num_classes=1000,
        transforms=None,
        meta={
            "num_params": 60192808,
            "min_size": (1, 1),
            "categories": _IMAGENET_CATEGORIES,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#resnet",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 78.312,
                    "acc@5": 94.046,
                },
            },
            "_ops": 11.514,
            "_file_size": 230.434,
            "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
        },
    )
    IMAGENET1K_V2 = Weights(
        path=K.ZOO_ROOT / "backbone/resnet/resnet152/imagenet1k_v2/resnet152_imagenet1k_v2.pt",
        url=Path("https://download.pytorch.org/models/resnet152-f82ba261.pth"),
        num_classes=1000,
        transforms=None,
        meta={
            "num_params": 60192808,
            "min_size": (1, 1),
            "categories": _IMAGENET_CATEGORIES,
            "recipe": "https://github.com/pytorch/vision/issues/3995#new-recipe",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 82.284,
                    "acc@5": 96.002,
                },
            },
            "_ops": 11.514,
            "_file_size": 230.474,
            "_docs": """
                These weights improve upon the results of the original paper by using TorchVision's `new training recipe
                <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
            """,
        },
    )
    DEFAULT = IMAGENET1K_V2


@WEIGHTS.register(name="resnext50_32x4d")
class ResNeXt50_32X4D_Weights(WeightsEnum):

    IMAGENET1K_V1 = Weights(
        path=K.ZOO_ROOT / "backbone/resnet/resnext50_32x4d/imagenet1k_v1/resnext50_32x4d_imagenet1k_v1.pt",
        url=Path("https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth"),
        num_classes=1000,
        transforms=None,
        meta={
            "num_params": 25028904,
            "min_size": (1, 1),
            "categories": _IMAGENET_CATEGORIES,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#resnext",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 77.618,
                    "acc@5": 93.698,
                },
            },
            "_ops": 4.23,
            "_file_size": 95.789,
            "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
        },
    )
    IMAGENET1K_V2 = Weights(
        path=K.ZOO_ROOT / "backbone/resnet/resnext50_32x4d/imagenet1k_v2/resnext50_32x4d_imagenet1k_v2.pt",
        url=Path("https://download.pytorch.org/models/resnext50_32x4d-1a0047aa.pth"),
        num_classes=1000,
        transforms=None,
        meta={
            "num_params": 25028904,
            "min_size": (1, 1),
            "categories": _IMAGENET_CATEGORIES,
            "recipe": "https://github.com/pytorch/vision/issues/3995#new-recipe",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 81.198,
                    "acc@5": 95.340,
                },
            },
            "_ops": 4.23,
            "_file_size": 95.833,
            "_docs": """
                These weights improve upon the results of the original paper by using TorchVision's `new training recipe
                <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
            """,
        },
    )
    DEFAULT = IMAGENET1K_V2


@WEIGHTS.register(name="resnext101_32x8d")
class ResNeXt101_32X8D_Weights(WeightsEnum):

    IMAGENET1K_V1 = Weights(
        path=K.ZOO_ROOT / "backbone/resnet/resnext101_32x8d/imagenet1k_v1/resnext101_32x8d_imagenet1k_v1.pt",
        url=Path("https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth"),
        num_classes=1000,
        transforms=None,
        meta={
            "num_params": 88791336,
            "min_size": (1, 1),
            "categories": _IMAGENET_CATEGORIES,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#resnext",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 79.312,
                    "acc@5": 94.526,
                },
            },
            "_ops": 16.414,
            "_file_size": 339.586,
            "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
        },
    )
    IMAGENET1K_V2 = Weights(
        path=K.ZOO_ROOT / "backbone/resnet/resnext101_32x8d/imagenet1k_v2/resnext101_32x8d_imagenet1k_v2.pt",
        url=Path("https://download.pytorch.org/models/resnext101_32x8d-110c445d.pth"),
        num_classes=1000,
        transforms=None,
        meta={
            "num_params": 88791336,
            "min_size": (1, 1),
            "categories": _IMAGENET_CATEGORIES,
            "recipe": "https://github.com/pytorch/vision/issues/3995#new-recipe-with-fixres",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 82.834,
                    "acc@5": 96.228,
                },
            },
            "_ops": 16.414,
            "_file_size": 339.673,
            "_docs": """
                These weights improve upon the results of the original paper by using TorchVision's `new training recipe
                <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
            """,
        },
    )
    DEFAULT = IMAGENET1K_V2


@WEIGHTS.register(name="resnext101_64x4d")
class ResNeXt101_64X4D_Weights(WeightsEnum):

    IMAGENET1K_V1 = Weights(
        path=K.ZOO_ROOT / "backbone/resnet/resnext101_64x4d/imagenet1k_v1/resnext101_64x4d_imagenet1k_v1.pt",
        url=Path("ttps://download.pytorch.org/models/resnext101_64x4d-173b62eb.pth"),
        num_classes=1000,
        transforms=None,
        meta={
            "num_params": 83455272,
            "min_size": (1, 1),
            "categories": _IMAGENET_CATEGORIES,
            "recipe": "https://github.com/pytorch/vision/pull/5935",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 83.246,
                    "acc@5": 96.454,
                },
            },
            "_ops": 15.46,
            "_file_size": 319.318,
            "_docs": """
                These weights were trained from scratch by using TorchVision's `new training recipe
                <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
            """,
        },
    )
    DEFAULT = IMAGENET1K_V1


@WEIGHTS.register(name="wide_resnet50_2")
class Wide_ResNet50_2_Weights(WeightsEnum):

    IMAGENET1K_V1 = Weights(
        path=K.ZOO_ROOT / "backbone/resnet/wide_resnet50_2/imagenet1k_v1/wide_resnet50_2_imagenet1k_v1.pt",
        url=Path("https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth"),
        num_classes=1000,
        transforms=None,
        meta={
            "num_params": 68883240,
            "min_size": (1, 1),
            "categories": _IMAGENET_CATEGORIES,
            "recipe": "https://github.com/pytorch/vision/pull/912#issue-445437439",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 78.468,
                    "acc@5": 94.086,
                },
            },
            "_ops": 11.398,
            "_file_size": 131.82,
            "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
        },
    )
    IMAGENET1K_V2 = Weights(
        path=K.ZOO_ROOT / "backbone/resnet/wide_resnet50_2/imagenet1k_v2/wide_resnet50_2_imagenet1k_v2.pt",
        url=Path("https://download.pytorch.org/models/wide_resnet50_2-9ba9bcbe.pth"),
        num_classes=1000,
        transforms=None,
        meta={
            "num_params": 68883240,
            "min_size": (1, 1),
            "categories": _IMAGENET_CATEGORIES,
            "recipe": "https://github.com/pytorch/vision/issues/3995#new-recipe-with-fixres",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 81.602,
                    "acc@5": 95.758,
                },
            },
            "_ops": 11.398,
            "_file_size": 263.124,
            "_docs": """
                These weights improve upon the results of the original paper by using TorchVision's `new training recipe
                <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
            """,
        },
    )
    DEFAULT = IMAGENET1K_V2


@WEIGHTS.register(name="wide_resnet101_2")
class Wide_ResNet101_2_Weights(WeightsEnum):

    IMAGENET1K_V1 = Weights(
        path=K.ZOO_ROOT / "backbone/resnet/wide_resnet101_2/imagenet1k_v1/wide_resnet101_2_imagenet1k_v1.pt",
        url=Path("https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth"),
        num_classes=1000,
        transforms=None,
        meta={
            "num_params": 126886696,
            "min_size": (1, 1),
            "categories": _IMAGENET_CATEGORIES,
            "recipe": "https://github.com/pytorch/vision/pull/912#issue-445437439",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 78.848,
                    "acc@5": 94.284,
                },
            },
            "_ops": 22.753,
            "_file_size": 242.896,
            "_docs": """These weights reproduce closely the results of the paper using a simple training recipe.""",
        },
    )
    IMAGENET1K_V2 = Weights(
        path=K.ZOO_ROOT / "backbone/resnet/wide_resnet50_2/imagenet1k_v2/wide_resnet50_2_imagenet1k_v2.pt",
        url=Path("https://download.pytorch.org/models/wide_resnet101_2-d733dc28.pth"),
        num_classes=1000,
        transforms=None,
        meta={
            "num_params": 126886696,
            "min_size": (1, 1),
            "categories": _IMAGENET_CATEGORIES,
            "recipe": "https://github.com/pytorch/vision/issues/3995#new-recipe",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 82.510,
                    "acc@5": 96.020,
                },
            },
            "_ops": 22.753,
            "_file_size": 484.747,
            "_docs": """
                These weights improve upon the results of the original paper by using TorchVision's `new training recipe
                <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
            """,
        },
    )
    DEFAULT = IMAGENET1K_V2


# --- Model Variants ---

@BACKBONES.register(name="resnet18", metaclass=ResNetBackBone)
def resnet18(
    weights: WeightsLike = "default",
    out_indices: list[int] | None = None,
    *args, **kwargs,
) -> ResNetBackBone:
    """Create a ResNet-18 backbone.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
        out_indices (list[int] | None, optional): List of layer indices to
            extract features from. Defaults to None.
    """
    return ResNetBackBone(
        name="resnet18",
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        weights=ResNet18_Weights(weights),
        out_indices=out_indices,
        *args, **kwargs,
    )


@BACKBONES.register(name="resnet34", metaclass=ResNetBackBone)
def resnet34(
    weights: WeightsLike = "default",
    out_indices: list[int] | None = None,
    *args, **kwargs,
) -> ResNetBackBone:
    """Create a ResNet-34 backbone.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
        out_indices (list[int] | None, optional): List of layer indices to
            extract features from. Defaults to None.
    """
    return ResNetBackBone(
        name="resnet34",
        block=BasicBlock,
        layers=[3, 4, 6, 3],
        weights=ResNet34_Weights(weights),
        out_indices=out_indices,
        *args, **kwargs,
    )


@BACKBONES.register(name="resnet50", metaclass=ResNetBackBone)
def resnet50(
    weights: WeightsLike = "default",
    out_indices: list[int] | None = None,
    *args, **kwargs,
) -> ResNetBackBone:
    """Create a ResNet-50 backbone.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
        out_indices (list[int] | None, optional): List of layer indices to
            extract features from. Defaults to None.
    """
    return ResNetBackBone(
        name="resnet50",
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        weights=ResNet50_Weights(weights),
        out_indices=out_indices,
        *args, **kwargs,
    )


@BACKBONES.register(name="resnet101", metaclass=ResNetBackBone)
def resnet101(
    weights: WeightsLike = "default",
    out_indices: list[int] | None = None,
    *args, **kwargs,
) -> ResNetBackBone:
    """Create a ResNet-101 backbone.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
        out_indices (list[int] | None, optional): List of layer indices to
            extract features from. Defaults to None.
    """
    return ResNetBackBone(
        name="resnet101",
        block=Bottleneck,
        layers=[3, 4, 23, 3],
        weights=ResNet101_Weights(weights),
        out_indices=out_indices,
        *args, **kwargs,
    )


@BACKBONES.register(name="resnet152", metaclass=ResNetBackBone)
def resnet152(
    weights: WeightsLike = "default",
    out_indices: list[int] | None = None,
    *args, **kwargs,
) -> ResNetBackBone:
    """Create a ResNet-152 backbone.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
        out_indices (list[int] | None, optional): List of layer indices to
            extract features from. Defaults to None.
    """
    return ResNetBackBone(
        name="resnet152",
        block=Bottleneck,
        layers=[3, 8, 36, 3],
        weights=ResNet152_Weights(weights),
        out_indices=out_indices,
        *args, **kwargs,
    )


@BACKBONES.register(name="resnext50_32x4d", metaclass=ResNetBackBone)
def resnext50_32x4d(
    weights: WeightsLike = "default",
    out_indices: list[int] | None = None,
    *args, **kwargs,
) -> ResNetBackBone:
    """Create a ResNeXt-50 32x4d backbone.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
        out_indices (list[int] | None, optional): List of layer indices to
            extract features from. Defaults to None.
    """
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 4
    return ResNetBackBone(
        name="resnext50_32x4d",
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        weights=ResNeXt50_32X4D_Weights(weights),
        out_indices=out_indices,
        *args, **kwargs,
    )


@BACKBONES.register(name="resnext101_32x8d", metaclass=ResNetBackBone)
def resnext101_32x8d(
    weights: WeightsLike = "default",
    out_indices: list[int] | None = None,
    *args, **kwargs,
) -> ResNetBackBone:
    """Create a ResNeXt-101 32x8d backbone.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
        out_indices (list[int] | None, optional): List of layer indices to
            extract features from. Defaults to None.
    """
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 4
    return ResNetBackBone(
        name="resnext101_32x8d",
        block=Bottleneck,
        layers=[3, 4, 23, 3],
        weights=ResNeXt101_32X8D_Weights(weights),
        out_indices=out_indices,
        *args, **kwargs,
    )


@BACKBONES.register(name="resnext101_64x4d", metaclass=ResNetBackBone)
def resnext101_64x4d(
    weights: WeightsLike = "default",
    out_indices: list[int] | None = None,
    *args, **kwargs,
) -> ResNetBackBone:
    """Create a ResNeXt-101 64x4d backbone.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
        out_indices (list[int] | None, optional): List of layer indices to
            extract features from. Defaults to None.
    """
    kwargs["groups"] = 64
    kwargs["width_per_group"] = 4
    return ResNetBackBone(
        name="resnext101_64x4d",
        block=Bottleneck,
        layers=[3, 4, 23, 3],
        weights=ResNeXt101_64X4D_Weights(weights),
        out_indices=out_indices,
        *args, **kwargs,
    )


@BACKBONES.register(name="wide_resnet50_2", metaclass=ResNetBackBone)
def wide_resnet50_2(
    weights: WeightsLike = "default",
    out_indices: list[int] | None = None,
    *args, **kwargs,
) -> ResNetBackBone:
    """Create a Wide-ResNet-50-2 backbone.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
        out_indices (list[int] | None, optional): List of layer indices to
            extract features from. Defaults to None.
    """
    kwargs["width_per_group"] = 64 * 2
    return ResNetBackBone(
        name="wide_resnet50_2",
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        weights=Wide_ResNet50_2_Weights(weights),
        out_indices=out_indices,
        *args, **kwargs,
    )


@BACKBONES.register(name="wide_resnet101_2", metaclass=ResNetBackBone)
def wide_resnet101_2(
    weights: WeightsLike = "default",
    out_indices: list[int] | None = None,
    *args, **kwargs,
) -> ResNetBackBone:
    """Create a Wide-ResNet-101-2 backbone.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
        out_indices (list[int] | None, optional): List of layer indices to
            extract features from. Defaults to None.
    """
    kwargs["width_per_group"] = 64 * 2
    return ResNetBackBone(
        name="wide_resnet101_2",
        block=Bottleneck,
        layers=[3, 4, 23, 3],
        weights=Wide_ResNet101_2_Weights(weights),
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

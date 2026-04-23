#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Swin Transformer backbones.

This module provides various Swin Transformer backbones using PyTorch.
"""

from __future__ import annotations

__all__ = [
    "Swin_B_Weights",
    "Swin_S_Weights",
    "Swin_T_Weights",
    "Swin_V2_B_Weights",
    "Swin_V2_S_Weights",
    "Swin_V2_T_Weights",
    "swin_b",
    "swin_s",
    "swin_t",
    "swin_v2_b",
    "swin_v2_s",
    "swin_v2_t",
]

from torch import nn, Tensor
from torchvision.models._meta import _IMAGENET_CATEGORIES
from torchvision.models.swin_transformer import SwinTransformer

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

class SwinBackBone(ModelRegisterMixin, nn.Module):
    """Swin backbone."""

    arch: str = "swin"
    name: str = "swin"
    tasks: list[Task] = [Task.BACKBONE]
    strategies: list[Strategy] = [Strategy.RESIZE]
    model_dir: Path = current_dir

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        name: str,
        patch_size: list[int],
        embed_dim: int,
        depths: list[int],
        num_heads: list[int],
        window_size: list[int],
        stochastic_depth_prob: float,
        weights: Weights | None = None,
        out_indices: list[int] | None = None,
        verbose: bool = True,
        *args, **kwargs,
    ):
        """Initialize a new instance.

        Args:
            name (str): Name of the model variant.
            patch_size: Patch size of the backbone.
            embed_dim (int): Embedding dimension of the backbone.
            depths (list[int]): Depth of each Swin Transformer stage.
            num_heads (list[int]): Number of attention heads in different layers.
            window_size (list[int]): Window size of the Swin Transformer.
            stochastic_depth_prob (float): Probability of dropping a depth unit
                in each stage.
            weights (Weights, optional): Pre-trained weights to load.
                Defaults to None.
            out_indices (list[int], optional): List of layer indices to extract
                features from. If None, defaults to [1, 3, 5, 7].
            verbose (bool): Verbosity mode. Defaults to True.
        """
        super().__init__(name=name)

        # Assign attributes
        self.verbose = verbose

        # Define model
        if weights is not None and is_weights_type(weights):
            kwargs["num_classes"] = weights.num_classes or kwargs["num_classes"]

        base_model = SwinTransformer(
            patch_size=patch_size,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            stochastic_depth_prob=stochastic_depth_prob,
            *args, **kwargs,
        )

        # Load weights
        if weights is not None and is_weights_type(weights):
            base_model.load_state_dict(weights.state_dict())
            if self.verbose:
                log(f"Initialized '{name}' from weights: '{weights.path}'.")
        else:
            if self.verbose:
                log(f"Initialized '{name}' from scratch.")

        # In torchvision, Swin features are organized into 4 hierarchical stages
        # stage 0-1: resolution 1/4
        # stage 2-3: resolution 1/8
        # stage 4-5: resolution 1/16
        # stage 6-7: resolution 1/32
        self.features = base_model.features
        self.out_indices = out_indices or [1, 3, 5, 7]
        self.out_channels = self._get_out_channels(variant=name)

    def _get_out_channels(self, variant: str) -> list[int]:
        """Get the output channels for the specified variant."""
        # Channels usually follow the C, 2C, 4C, 8C pattern
        mapping = {
            "swin_t": [96, 192, 384, 768],
            "swin_s": [96, 192, 384, 768],
            "swin_b": [128, 256, 512, 1024],
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
        outputs = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.out_indices:
                # Swin outputs are often (B, H, W, C) or (B, L, C)
                # We permute to (B, C, H, W) to match CNN-style Necks
                feat = x.permute(0, 3, 1, 2)
                outputs.append(feat)
        return outputs

# endregion


# ==============================================================================
# region CONCRETE IMPLEMENTATIONS
# ==============================================================================

# --- Pre-trained Weights ---

@WEIGHTS.register(name="swin_t")
class Swin_T_Weights(WeightsEnum):

    IMAGENET1K_V1 = Weights(
        path=K.ZOO_ROOT / "backbone/swin/swin_t/imagenet1k_v1/swin_t_imagenet1k_v1.pt",
        url=Path("https://download.pytorch.org/models/swin_t-704ceda3.pth"),
        num_classes=1000,
        transforms=None,
        meta={
            "num_params": 28288354,
            "min_size": (224, 224),
            "categories": _IMAGENET_CATEGORIES,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#swintransformer",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 81.474,
                    "acc@5": 95.776,
                },
            },
            "_ops": 4.491,
            "_file_size": 108.19,
            "_docs": """These weights reproduce closely the results of the paper using a similar training recipe.""",
        },
    )
    DEFAULT = IMAGENET1K_V1


@WEIGHTS.register(name="swin_s")
class Swin_S_Weights(WeightsEnum):

    IMAGENET1K_V1 = Weights(
        path=K.ZOO_ROOT / "backbone/swin/swin_s/imagenet1k_v1/swin_s_imagenet1k_v1.pt",
        url=Path("https://download.pytorch.org/models/swin_s-5e29d889.pth"),
        num_classes=1000,
        transforms=None,
        meta={
            "num_params": 49606258,
            "min_size": (224, 224),
            "categories": _IMAGENET_CATEGORIES,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#swintransformer",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 83.196,
                    "acc@5": 96.360,
                },
            },
            "_ops": 8.741,
            "_file_size": 189.786,
            "_docs": """These weights reproduce closely the results of the paper using a similar training recipe.""",
        },
    )
    DEFAULT = IMAGENET1K_V1


@WEIGHTS.register(name="swin_b")
class Swin_B_Weights(WeightsEnum):

    IMAGENET1K_V1 = Weights(
        path=K.ZOO_ROOT / "backbone/swin/swin_b/imagenet1k_v1/swin_b_imagenet1k_v1.pt",
        url=Path("https://download.pytorch.org/models/swin_b-68c6b09e.pth"),
        num_classes=1000,
        transforms=None,
        meta={
            "num_params": 87768224,
            "min_size": (224, 224),
            "categories": _IMAGENET_CATEGORIES,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#swintransformer",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 83.582,
                    "acc@5": 96.640,
                },
            },
            "_ops": 15.431,
            "_file_size": 335.364,
            "_docs": """These weights reproduce closely the results of the paper using a similar training recipe.""",
        },
    )
    DEFAULT = IMAGENET1K_V1


@WEIGHTS.register(name="swin_v2_t")
class Swin_V2_T_Weights(WeightsEnum):

    IMAGENET1K_V1 = Weights(
        path=K.ZOO_ROOT / "backbone/swin/swin_v2_t/imagenet1k_v1/swin_v2_t_imagenet1k_v1.pt",
        url=Path("https://download.pytorch.org/models/swin_v2_t-b137f0e2.pth"),
        num_classes=1000,
        transforms=None,
        meta={
            "num_params": 28351570,
            "min_size": (256, 256),
            "categories": _IMAGENET_CATEGORIES,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#swintransformer-v2",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 82.072,
                    "acc@5": 96.132,
                },
            },
            "_ops": 5.94,
            "_file_size": 108.626,
            "_docs": """These weights reproduce closely the results of the paper using a similar training recipe.""",
        },
    )
    DEFAULT = IMAGENET1K_V1


@WEIGHTS.register(name="swin_v2_s")
class Swin_V2_S_Weights(WeightsEnum):

    IMAGENET1K_V1 = Weights(
        path=K.ZOO_ROOT / "backbone/swin/swin_v2_s/imagenet1k_v1/swin_v2_s_imagenet1k_v1.pt",
        url=Path("https://download.pytorch.org/models/swin_v2_s-637d8ceb.pth"),
        num_classes=1000,
        transforms=None,
        meta={
            "num_params": 49737442,
            "min_size": (256, 256),
            "categories": _IMAGENET_CATEGORIES,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#swintransformer-v2",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 83.712,
                    "acc@5": 96.816,
                },
            },
            "_ops": 11.546,
            "_file_size": 190.675,
            "_docs": """These weights reproduce closely the results of the paper using a similar training recipe.""",
        },
    )
    DEFAULT = IMAGENET1K_V1


@WEIGHTS.register(name="swin_v2_b")
class Swin_V2_B_Weights(WeightsEnum):

    IMAGENET1K_V1 = Weights(
        path=K.ZOO_ROOT / "backbone/swin/swin_v2_b/imagenet1k_v1/swin_v2_b_imagenet1k_v1.pt",
        url=Path("https://download.pytorch.org/models/swin_v2_b-781e5279.pth"),
        num_classes=1000,
        transforms=None,
        meta={
            "num_params": 87930848,
            "min_size": (256, 256),
            "categories": _IMAGENET_CATEGORIES,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#swintransformer-v2",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 84.112,
                    "acc@5": 96.864,
                },
            },
            "_ops": 20.325,
            "_file_size": 336.372,
            "_docs": """These weights reproduce closely the results of the paper using a similar training recipe.""",
        },
    )
    DEFAULT = IMAGENET1K_V1


# --- Model Variants ---

@BACKBONES.register(name="swin_t", metaclass=SwinBackBone)
def swin_t(
    weights: WeightsLike = "default",
    out_indices: list[int] | None = None,
    *args, **kwargs,
) -> SwinBackBone:
    """Create a Swin-T backbone.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
        out_indices (list[int], optional): List of layer indices to extract
            features from. Defaults to None.
    """
    return SwinBackBone(
        name="swin_t",
        patch_size=[4, 4],
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=[7, 7],
        stochastic_depth_prob=0.2,
        weights=Swin_T_Weights(weights),
        out_indices=out_indices,
        *args, **kwargs,
    )


@BACKBONES.register(name="swin_s", metaclass=SwinBackBone)
def swin_s(
    weights: WeightsLike = "default",
    out_indices: list[int] | None = None,
    *args, **kwargs,
) -> SwinBackBone:
    """Create a Swin-S backbone.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
        out_indices (list[int], optional): List of layer indices to extract
            features from. Defaults to None.
    """
    return SwinBackBone(
        name="swin_s",
        patch_size=[4, 4],
        embed_dim=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=[7, 7],
        stochastic_depth_prob=0.3,
        weights=Swin_S_Weights(weights),
        out_indices=out_indices,
        *args, **kwargs,
    )


@BACKBONES.register(name="swin_b", metaclass=SwinBackBone)
def swin_b(
    weights: WeightsLike = "default",
    out_indices: list[int] | None = None,
    *args, **kwargs,
) -> SwinBackBone:
    """Create a Swin-B backbone.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
        out_indices (list[int], optional): List of layer indices to extract
            features from. Defaults to None.
    """
    return SwinBackBone(
        name="swin_b",
        patch_size=[4, 4],
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=[7, 7],
        stochastic_depth_prob=0.5,
        weights=Swin_B_Weights(weights),
        out_indices=out_indices,
        *args, **kwargs,
    )


@BACKBONES.register(name="swin_v2_t", metaclass=SwinBackBone)
def swin_v2_t(
    weights: WeightsLike = "default",
    out_indices: list[int] | None = None,
    *args, **kwargs,
) -> SwinBackBone:
    """Create a Swin-V2-T backbone.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
        out_indices (list[int], optional): List of layer indices to extract
            features from. Defaults to None.
    """
    return SwinBackBone(
        name="swin_v2_t",
        patch_size=[4, 4],
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=[8, 8],
        stochastic_depth_prob=0.2,
        weights=Swin_V2_T_Weights(weights),
        out_indices=out_indices,
        *args, **kwargs,
    )


@BACKBONES.register(name="swin_v2_s", metaclass=SwinBackBone)
def swin_v2_s(
    weights: WeightsLike = "default",
    out_indices: list[int] | None = None,
    *args, **kwargs,
) -> SwinBackBone:
    """Create a Swin-V2-S backbone.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
        out_indices (list[int], optional): List of layer indices to extract
            features from. Defaults to None.
    """
    return SwinBackBone(
        name="swin_v2_s",
        patch_size=[4, 4],
        embed_dim=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=[8, 8],
        stochastic_depth_prob=0.3,
        weights=Swin_V2_S_Weights(weights),
        out_indices=out_indices,
        *args, **kwargs,
    )


@BACKBONES.register(name="swin_v2_b", metaclass=SwinBackBone)
def swin_v2_b(
    weights: WeightsLike = "default",
    out_indices: list[int] | None = None,
    *args, **kwargs,
) -> SwinBackBone:
    """Create a Swin-V2-B backbone.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
        out_indices (list[int], optional): List of layer indices to extract
            features from. Defaults to None.
    """
    return SwinBackBone(
        name="swin_v2_b",
        patch_size=[4, 4],
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=[8, 8],
        stochastic_depth_prob=0.5,
        weights=Swin_V2_B_Weights(weights),
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

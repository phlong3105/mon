#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Swin Transformer backbones.

This module implements various Swin Transformer backbones using PyTorch.
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

import torch
import torch.nn as nn
from torchvision.models._meta import _IMAGENET_CATEGORIES
from torchvision.models.swin_transformer import SwinTransformer

from mon.core import BACKBONES, MLType, Path, ROOT_DIR, Task, WEIGHTS
from mon.core.dtypes import Weights, WeightsEnum
from ...base import RegistrableMixin

current_file = Path(__file__).normalize()
current_dir  = current_file.parents[0]


# ==============================================================================
# region BASE CLASSES & MIXINS
# ==============================================================================

# --- Base Classes ---

class SwinBackBone(nn.Module, RegistrableMixin):
    """Swin backbone.

    Attributes:
        features (torch.nn.Sequential): The feature extraction layers.
        out_indices (list): List of layer indices to extract features from.
        out_channels (list): List of output channels for each extracted layer.
    """

    _arch     : str          = "swin"
    _name     : str          = None
    _tasks    : list[Task]   = [Task.BACKBONE]
    _mltypes  : list[MLType] = []
    _model_dir: Path         = current_dir

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        name                 : str,
        patch_size           : list[int],
        embed_dim            : int,
        depths               : list[int],
        num_heads            : list[int],
        window_size          : list[int],
        stochastic_depth_prob: float,
        weights              : WeightsEnum | None = None,
        out_indices          : list | None        = None,
        *args, **kwargs
    ):
        """Initialize a new instance.

        Args:
            name: Variant of Swin Transformer to use.
            patch_size: Patch size of the backbone.
            embed_dim: Embedding dimension.
            depths: Depth of each Swin Transformer layer.
            num_heads: Number of attention heads.
            window_size: Window size for the Swin Transformer.
            stochastic_depth_prob: Stochastic depth probability.
            weights: Pre-trained weights to load.
            out_indices: List of layer indices to extract features from.
                If None, defaults to [1, 3, 5, 7].
            args: Additional positional arguments for the ResNet model.
            kwargs: Additional keyword arguments for the ResNet model
        """
        super().__init__(name=name, *args, **kwargs)

        # Load the base model
        if isinstance(weights, WeightsEnum):
            kwargs["num_classes"] = weights.num_classes

        base_model = SwinTransformer(
            patch_size            = patch_size,
            embed_dim             = embed_dim,
            depths                = depths,
            num_heads             = num_heads,
            window_size           = window_size,
            stochastic_depth_prob = stochastic_depth_prob,
            *args, **kwargs
        )

        if isinstance(weights, WeightsEnum):
            base_model.load_state_dict(weights.get_state_dict())

        # In torchvision, Swin features are organized into 4 hierarchical stages
        # stage 0-1: resolution 1/4
        # stage 2-3: resolution 1/8
        # stage 4-5: resolution 1/16
        # stage 6-7: resolution 1/32
        self.features     = base_model.features
        self.out_indices  = out_indices or [1, 3, 5, 7]
        self.out_channels = self._get_out_channels(variant=name)

    def _get_out_channels(self, variant: str) -> list[int]:
        # Channels usually follow the C, 2C, 4C, 8C pattern
        mapping = {
            "swin_t": [96,  192, 384, 768 ],
            "swin_s": [96,  192, 384, 768 ],
            "swin_b": [128, 256, 512, 1024],
        }
        return mapping.get(variant, [96, 192, 384, 768])

    # --- Callable & Context Manager ---
    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Forward the input through the network.

        Args:
            x: Input tensor of shape (B, C, H, W) and values ranging
                from 0.0 to 1.0.

        Returns:
            A list of feature maps from the specified layers.
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


# --- Mixins ---


# endregion


# ==============================================================================
# region CONCRETE IMPLEMENTATIONS
# ==============================================================================

# --- Pre-trained Weights ---


@WEIGHTS.register(arch="swin", name="swin_t")
class Swin_T_Weights(WeightsEnum):

    IMAGENET1K_V1 = Weights(
        url         = "https://download.pytorch.org/models/swin_t-704ceda3.pth",
        path        = ROOT_DIR / "zoo/nn/backbone/swin/swin_t/imagenet1k_v1/swin_t_imagenet1k_v1.pth",
        num_classes = 1000,
        transforms  = None,
        meta        = {
            "num_params": 28288354,
            "min_size"  : (224, 224),
            "categories": _IMAGENET_CATEGORIES,
            "recipe"    : "https://github.com/pytorch/vision/tree/main/references/classification#swintransformer",
            "_metrics"  : {
                "ImageNet-1K": {
                    "acc@1": 81.474,
                    "acc@5": 95.776,
                }
            },
            "_ops"      : 4.491,
            "_file_size": 108.19,
            "_docs"     : """These weights reproduce closely the results of the paper using a similar training recipe.""",
        }
    )
    DEFAULT = IMAGENET1K_V1


@WEIGHTS.register(arch="swin", name="swin_s")
class Swin_S_Weights(WeightsEnum):

    IMAGENET1K_V1 = Weights(
        url         = "https://download.pytorch.org/models/swin_s-5e29d889.pth",
        path        = ROOT_DIR / "zoo/nn/backbone/swin/swin_s/imagenet1k_v1/swin_s_imagenet1k_v1.pth",
        num_classes = 1000,
        transforms  = None,
        meta        = {
            "num_params": 49606258,
            "min_size"  : (224, 224),
            "categories": _IMAGENET_CATEGORIES,
            "recipe"    : "https://github.com/pytorch/vision/tree/main/references/classification#swintransformer",
            "_metrics"  : {
                "ImageNet-1K": {
                    "acc@1": 83.196,
                    "acc@5": 96.360,
                }
            },
            "_ops"      : 8.741,
            "_file_size": 189.786,
            "_docs"     : """These weights reproduce closely the results of the paper using a similar training recipe.""",
        }
    )
    DEFAULT = IMAGENET1K_V1


@WEIGHTS.register(arch="swin", name="swin_b")
class Swin_B_Weights(WeightsEnum):

    IMAGENET1K_V1 = Weights(
        url         = "https://download.pytorch.org/models/swin_b-68c6b09e.pth",
        path        = ROOT_DIR / "zoo/nn/backbone/swin/swin_b/imagenet1k_v1/swin_b_imagenet1k_v1.pth",
        num_classes = 1000,
        transforms  = None,
        meta        = {
            "num_params": 87768224,
            "min_size"  : (224, 224),
            "categories": _IMAGENET_CATEGORIES,
            "recipe"    : "https://github.com/pytorch/vision/tree/main/references/classification#swintransformer",
            "_metrics"  : {
                "ImageNet-1K": {
                    "acc@1": 83.582,
                    "acc@5": 96.640,
                }
            },
            "_ops"      : 15.431,
            "_file_size": 335.364,
            "_docs"     : """These weights reproduce closely the results of the paper using a similar training recipe.""",
        }
    )
    DEFAULT = IMAGENET1K_V1


@WEIGHTS.register(arch="swin", name="swin_v2_t")
class Swin_V2_T_Weights(WeightsEnum):

    IMAGENET1K_V1 = Weights(
        url         = "https://download.pytorch.org/models/swin_v2_t-b137f0e2.pth",
        path        = ROOT_DIR / "zoo/nn/backbone/swin/swin_v2_t/imagenet1k_v1/swin_v2_t_imagenet1k_v1.pth",
        num_classes = 1000,
        transforms  = None,
        meta        = {
            "num_params": 28351570,
            "min_size"  : (256, 256),
            "categories": _IMAGENET_CATEGORIES,
            "recipe"    : "https://github.com/pytorch/vision/tree/main/references/classification#swintransformer-v2",
            "_metrics"  : {
                "ImageNet-1K": {
                    "acc@1": 82.072,
                    "acc@5": 96.132,
                }
            },
            "_ops"      : 5.94,
            "_file_size": 108.626,
            "_docs"     : """These weights reproduce closely the results of the paper using a similar training recipe.""",
        }
    )
    DEFAULT = IMAGENET1K_V1


@WEIGHTS.register(arch="swin", name="swin_v2_s")
class Swin_V2_S_Weights(WeightsEnum):

    IMAGENET1K_V1 = Weights(
        url         = "https://download.pytorch.org/models/swin_v2_s-637d8ceb.pth",
        path        = ROOT_DIR / "zoo/nn/backbone/swin/swin_v2_s/imagenet1k_v1/swin_v2_s_imagenet1k_v1.pth",
        num_classes = 1000,
        transforms  = None,
        meta        = {
            "num_params": 49737442,
            "min_size"  : (256, 256),
            "categories": _IMAGENET_CATEGORIES,
            "recipe"    : "https://github.com/pytorch/vision/tree/main/references/classification#swintransformer-v2",
            "_metrics"  : {
                "ImageNet-1K": {
                    "acc@1": 83.712,
                    "acc@5": 96.816,
                }
            },
            "_ops"      : 11.546,
            "_file_size": 190.675,
            "_docs"     : """These weights reproduce closely the results of the paper using a similar training recipe.""",
        }
    )
    DEFAULT = IMAGENET1K_V1


@WEIGHTS.register(arch="swin", name="swin_v2_b")
class Swin_V2_B_Weights(WeightsEnum):

    IMAGENET1K_V1 = Weights(
        url         = "https://download.pytorch.org/models/swin_v2_b-781e5279.pth",
        path        = ROOT_DIR / "zoo/nn/backbone/swin/swin_v2_b/imagenet1k_v1/swin_v2_b_imagenet1k_v1.pth",
        num_classes = 1000,
        transforms  = None,
        meta        = {
            "num_params": 87930848,
            "min_size"  : (256, 256),
            "categories": _IMAGENET_CATEGORIES,
            "recipe"    : "https://github.com/pytorch/vision/tree/main/references/classification#swintransformer-v2",
            "_metrics"  : {
                "ImageNet-1K": {
                    "acc@1": 84.112,
                    "acc@5": 96.864,
                }
            },
            "_ops"      : 20.325,
            "_file_size": 336.372,
            "_docs"     : """These weights reproduce closely the results of the paper using a similar training recipe.""",
        }
    )
    DEFAULT = IMAGENET1K_V1


# --- Model Variants ---

@BACKBONES.register(name="swin_t")
def swin_t(
    weights    : WeightsEnum | str | None = Swin_T_Weights.DEFAULT,
    out_indices: list | None = None,
    *args, **kwargs
):
    """Create a Swin-T backbone.

    Args:
        weights: Pre-trained weights to load. Defaults to
            Swin_T_Weights.DEFAULT.
        out_indices: List of layer indices to extract features from. Defaults
            to None.
        args: Additional positional arguments for the ResNet model.
        kwargs: Additional keyword arguments for the ResNet model.

    Returns:
        A Swin-T backbone model.
    """
    return SwinBackBone(
        name                  = "swin_t",
        patch_size            = [4, 4],
        embed_dim             = 96,
        depths                = [2, 2, 6, 2],
        num_heads             = [3, 6, 12, 24],
        window_size           = [7, 7],
        stochastic_depth_prob = 0.2,
        weights               = Swin_T_Weights(weights),
        out_indices           = out_indices,
        *args, **kwargs
    )


@BACKBONES.register(name="swin_s")
def swin_s(
    weights    : WeightsEnum | str | None = Swin_S_Weights.DEFAULT,
    out_indices: list | None = None,
    *args, **kwargs
):
    """Create a Swin-S backbone.

    Args:
        weights: Pre-trained weights to load. Defaults to
            Swin_S_Weights.DEFAULT.
        out_indices: List of layer indices to extract features from. Defaults
            to None.
        args: Additional positional arguments for the ResNet model.
        kwargs: Additional keyword arguments for the ResNet model.

    Returns:
        A Swin-S backbone model.
    """
    return SwinBackBone(
        name                  = "swin_s",
        patch_size            = [4, 4],
        embed_dim             = 96,
        depths                = [2, 2, 18, 2],
        num_heads             = [3, 6, 12, 24],
        window_size           = [7, 7],
        stochastic_depth_prob = 0.3,
        weights               = Swin_S_Weights(weights),
        out_indices           = out_indices,
        *args, **kwargs
    )


@BACKBONES.register(name="swin_b")
def swin_b(
    weights    : WeightsEnum | str | None = Swin_B_Weights.DEFAULT,
    out_indices: list | None = None,
    *args, **kwargs
):
    """Create a Swin-B backbone.

    Args:
        weights: Pre-trained weights to load. Defaults to
            Swin_B_Weights.DEFAULT.
        out_indices: List of layer indices to extract features from. Defaults
            to None.
        args: Additional positional arguments for the ResNet model.
        kwargs: Additional keyword arguments for the ResNet model.

    Returns:
        A Swin-B backbone model.
    """
    return SwinBackBone(
        name                  = "swin_b",
        patch_size            = [4, 4],
        embed_dim             = 128,
        depths                = [2, 2, 18, 2],
        num_heads             = [4, 8, 16, 32],
        window_size           = [7, 7],
        stochastic_depth_prob = 0.5,
        weights               = Swin_B_Weights(weights),
        out_indices           = out_indices,
        *args, **kwargs
    )


@BACKBONES.register(name="swin_v2_t")
def swin_v2_t(
    weights    : WeightsEnum | str | None = Swin_V2_T_Weights.DEFAULT,
    out_indices: list | None = None,
    *args, **kwargs
):
    """Create a Swin-V2-T backbone.

    Args:
        weights: Pre-trained weights to load. Defaults to
            Swin_V2_T_Weights.DEFAULT.
        out_indices: List of layer indices to extract features from. Defaults
            to None.
        args: Additional positional arguments for the ResNet model.
        kwargs: Additional keyword arguments for the ResNet model.

    Returns:
        A Swin-V2-T backbone model.
    """
    return SwinBackBone(
        name                  = "swin_v2_t",
        patch_size            = [4, 4],
        embed_dim             = 96,
        depths                = [2, 2, 6, 2],
        num_heads             = [3, 6, 12, 24],
        window_size           = [8, 8],
        stochastic_depth_prob = 0.2,
        weights               = Swin_V2_T_Weights(weights),
        out_indices           = out_indices,
        *args, **kwargs
    )


@BACKBONES.register(name="swin_v2_s")
def swin_v2_s(
    weights    : WeightsEnum | str | None = Swin_V2_S_Weights.DEFAULT,
    out_indices: list | None = None,
    *args, **kwargs
):
    """Create a Swin-V2-S backbone.

    Args:
        weights: Pre-trained weights to load. Defaults to
            Swin_V2_S_Weights.DEFAULT.
        out_indices: List of layer indices to extract features from. Defaults
            to None.
        args: Additional positional arguments for the ResNet model.
        kwargs: Additional keyword arguments for the ResNet model.

    Returns:
        A Swin-V2-S backbone model.
    """
    return SwinBackBone(
        name                  = "swin_v2_s",
        patch_size            = [4, 4],
        embed_dim             = 96,
        depths                = [2, 2, 18, 2],
        num_heads             = [3, 6, 12, 24],
        window_size           = [8, 8],
        stochastic_depth_prob = 0.3,
        weights               = Swin_V2_S_Weights(weights),
        out_indices           = out_indices,
        *args, **kwargs
    )


@BACKBONES.register(name="swin_v2_b")
def swin_v2_b(
    weights    : WeightsEnum | str | None = Swin_V2_B_Weights.DEFAULT,
    out_indices: list | None = None,
    *args, **kwargs
):
    """Create a Swin-V2-B backbone.

    Args:
        weights: Pre-trained weights to load. Defaults to
            Swin_V2_B_Weights.DEFAULT.
        out_indices: List of layer indices to extract features from. Defaults
            to None.
        args: Additional positional arguments for the ResNet model.
        kwargs: Additional keyword arguments for the ResNet model.

    Returns:
        A Swin-V2-B backbone model.
    """
    return SwinBackBone(
        name                  = "swin_v2_b",
        patch_size            = [4, 4],
        embed_dim             = 128,
        depths                = [2, 2, 18, 2],
        num_heads             = [4, 8, 16, 32],
        window_size           = [8, 8],
        stochastic_depth_prob = 0.5,
        weights               = Swin_V2_B_Weights(weights),
        out_indices           = out_indices,
        *args, **kwargs
    )

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    model_ = swin_t(weights="default")
    x = torch.ones(1, 3, 224, 224)
    y = model_(x)
    print(model_.features)
    print(x)
    print(y)

# endregion

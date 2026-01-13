#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""EfficientNet backbones.

This module provides various EfficientNet V1 and V2 backbones using PyTorch.
"""

from __future__ import annotations

__all__ = [
    "EfficientNet_B0_Weights",
    "EfficientNet_B1_Weights",
    "EfficientNet_B2_Weights",
    "EfficientNet_B3_Weights",
    "EfficientNet_B4_Weights",
    "EfficientNet_B5_Weights",
    "EfficientNet_B6_Weights",
    "EfficientNet_B7_Weights",
    "EfficientNet_V2_L_Weights",
    "EfficientNet_V2_M_Weights",
    "EfficientNet_V2_S_Weights",
    "efficientnet_b0",
    "efficientnet_b1",
    "efficientnet_b2",
    "efficientnet_b3",
    "efficientnet_b4",
    "efficientnet_b5",
    "efficientnet_b6",
    "efficientnet_b7",
    "efficientnet_v2_l",
    "efficientnet_v2_m",
    "efficientnet_v2_s",
]

from functools import partial
from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
from torchvision.models._meta import _IMAGENET_CATEGORIES
from torchvision.models.efficientnet import (
    _efficientnet_conf,
    EfficientNet,
    FusedMBConvConfig,
    MBConvConfig,
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

class EfficientNetBackBone(nn.Module, RegistrableMixin):
    """EfficientNet V1 and V2 backbone.

    Attributes:
        features (torch.nn.Sequential): The feature extraction layers.
        out_indices (list): List of layer indices to extract features from.
        out_channels (list): List of output channels for each extracted layer.
    """

    _arch     : str          = "efficientnet"
    _name     : str          = None
    _tasks    : list[Task]   = [Task.BACKBONE]
    _mltypes  : list[MLType] = []
    _model_dir: Path         = current_dir

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        name                     : str,
        inverted_residual_setting: Sequence[Union[MBConvConfig, FusedMBConvConfig]],
        dropout                  : float,
        last_channel             : Optional[int],
        weights                  : WeightsEnum | None = None,
        out_indices              : list | None        = None,
        *args, **kwargs
    ):
        """Initialize a new instance.

        Args:
            name: Variant of EfficientNet to use.
            inverted_residual_setting: A list of InvertedResidualConfig to
                construct blocks.
            dropout: Dropout rate before final classifier.
            last_channel: Channel dimension of the last convolutional layer.
            weights: Pre-trained weights to load.
            out_indices: List of layer indices to extract features from.
                If None, defaults to [2, 3, 5, 8].
            args: Additional positional arguments for the ResNet model.
            kwargs: Additional keyword arguments for the ResNet model
        """
        super().__init__(name=name, *args, **kwargs)
        # Load the base model
        if isinstance(weights, WeightsEnum):
            kwargs["num_classes"] = weights.num_classes

        base_model = EfficientNet(
            inverted_residual_setting = inverted_residual_setting,
            dropout                   = dropout,
            last_channel              = last_channel,
            *args, **kwargs
        )

        if isinstance(weights, WeightsEnum):
            base_model.load_state_dict(weights.get_state_dict())

        # In torchvision, DenseNet already has a 'features' block
        self.features     = base_model.features
        self.out_indices  = out_indices or [2, 3, 5, 8]
        self.out_channels = self._get_out_channels(variant=name)

    def _get_out_channels(self, variant: str) -> list[int]:
        # Channels grow as the variant (B0-B7) increases
        mapping = {
            "efficientnet_b0": [24, 40, 112, 1280],
            "efficientnet_b1": [24, 40, 112, 1280],  # B1 uses same channels as B0 but more layers
            "efficientnet_b3": [32, 48, 136, 1536],
            "efficientnet_b7": [48, 80, 224, 2560],
        }
        return mapping.get(variant, [24, 40, 112, 1280])

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

@WEIGHTS.register(arch="efficientnet", name="efficientnet_b0")
class EfficientNet_B0_Weights(WeightsEnum):

    IMAGENET1K_V1 = Weights(
        url         = "https://download.pytorch.org/models/efficientnet_b0_rwightman-7f5810bc.pth",
        path        = ZOO_DIR / "nn/backbone/efficientnet/efficientnet_b0/imagenet1k_v1/efficientnet_b0_imagenet1k_v1.pth",
        num_classes = 1000,
        transforms  = None,
        meta        = {
            "num_params": 5288548,
            "min_size"  : (1, 1),
            "categories": _IMAGENET_CATEGORIES,
            "recipe"    : "https://github.com/pytorch/vision/tree/main/references/classification#efficientnet-v1",
            "_metrics"  : {
                "ImageNet-1K": {
                    "acc@1": 77.692,
                    "acc@5": 93.532,
                }
            },
            "_ops"      : 0.386,
            "_file_size": 20.451,
            "_docs"     : """These weights are ported from the original paper.""",
        }
    )
    DEFAULT = IMAGENET1K_V1


@WEIGHTS.register(arch="efficientnet", name="efficientnet_b1")
class EfficientNet_B1_Weights(WeightsEnum):

    IMAGENET1K_V1 = Weights(
        url         = "https://download.pytorch.org/models/efficientnet_b1_rwightman-bac287d4.pth",
        path        = ZOO_DIR / "nn/backbone/efficientnet/efficientnet_b1/imagenet1k_v1/efficientnet_b1_imagenet1k_v1.pth",
        num_classes = 1000,
        transforms  = None,
        meta        = {
            "num_params": 7794184,
            "min_size"  : (1, 1),
            "categories": _IMAGENET_CATEGORIES,
            "recipe"    : "https://github.com/pytorch/vision/tree/main/references/classification#efficientnet-v1",
            "_metrics"  : {
                "ImageNet-1K": {
                    "acc@1": 78.642,
                    "acc@5": 94.186,
                }
            },
            "_ops"      : 0.687,
            "_file_size": 30.134,
            "_docs"     : """These weights are ported from the original paper.""",
        }
    )
    IMAGENET1K_V2 = Weights(
        url         = "https://download.pytorch.org/models/efficientnet_b1-c27df63c.pth",
        path        = ZOO_DIR / "nn/backbone//efficientnet//efficientnet_b1/imagenet1k_v2//efficientnet_b1_imagenet1k_v2.pth",
        num_classes = 1000,
        transforms  = None,
        meta        = {
            "num_params": 7794184,
            "min_size"  : (1, 1),
            "categories": _IMAGENET_CATEGORIES,
            "recipe"    : "https://github.com/pytorch/vision/issues/3995#new-recipe-with-lr-wd-crop-tuning",
            "_metrics"  : {
                "ImageNet-1K": {
                    "acc@1": 79.838,
                    "acc@5": 94.934,
                }
            },
            "_ops"      : 0.687,
            "_file_size": 30.136,
            "_docs"     : """
                These weights improve upon the results of the original paper by using a modified version of TorchVision's
                `new training recipe
                <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
            """,
        }
    )
    DEFAULT = IMAGENET1K_V2


@WEIGHTS.register(arch="efficientnet", name="efficientnet_b2")
class EfficientNet_B2_Weights(WeightsEnum):

    IMAGENET1K_V1 = Weights(
        url         = "https://download.pytorch.org/models/efficientnet_b2_rwightman-c35c1473.pth",
        path        = ZOO_DIR / "nn/backbone/efficientnet/efficientnet_b2/imagenet1k_v1/efficientnet_b2_imagenet1k_v1.pth",
        num_classes = 1000,
        transforms  = None,
        meta        = {
            "num_params": 9109994,
            "min_size"  : (1, 1),
            "categories": _IMAGENET_CATEGORIES,
            "recipe"    : "https://github.com/pytorch/vision/tree/main/references/classification#efficientnet-v1",
            "_metrics"  : {
                "ImageNet-1K": {
                    "acc@1": 80.608,
                    "acc@5": 95.310,
                }
            },
            "_ops"      : 1.088,
            "_file_size": 35.174,
            "_docs"     : """These weights are ported from the original paper.""",
        }
    )
    DEFAULT = IMAGENET1K_V1


@WEIGHTS.register(arch="efficientnet", name="efficientnet_b3")
class EfficientNet_B3_Weights(WeightsEnum):

    IMAGENET1K_V1 = Weights(
        url         = "https://download.pytorch.org/models/efficientnet_b3_rwightman-b3899882.pth",
        path        = ZOO_DIR / "nn/backbone/efficientnet/efficientnet_b3/imagenet1k_v1/efficientnet_b3_imagenet1k_v1.pth",
        num_classes = 1000,
        transforms  = None,
        meta        = {
            "num_params": 12233232,
            "min_size"  : (1, 1),
            "categories": _IMAGENET_CATEGORIES,
            "recipe"    : "https://github.com/pytorch/vision/tree/main/references/classification#efficientnet-v1",
            "_metrics"  : {
                "ImageNet-1K": {
                    "acc@1": 82.008,
                    "acc@5": 96.054,
                }
            },
            "_ops"      : 1.827,
            "_file_size": 47.184,
            "_docs"     : """These weights are ported from the original paper.""",
        }
    )
    DEFAULT = IMAGENET1K_V1


@WEIGHTS.register(arch="efficientnet", name="efficientnet_b4")
class EfficientNet_B4_Weights(WeightsEnum):

    IMAGENET1K_V1 = Weights(
        url         = "https://download.pytorch.org/models/efficientnet_b4_rwightman-23ab8bcd.pth",
        path        = ZOO_DIR / "nn/backbone/efficientnet/efficientnet_b4/imagenet1k_v1/efficientnet_b4_imagenet1k_v1.pth",
        num_classes = 1000,
        transforms  = None,
        meta        = {
            "num_params": 19341616,
            "min_size"  : (1, 1),
            "categories": _IMAGENET_CATEGORIES,
            "recipe"    : "https://github.com/pytorch/vision/tree/main/references/classification#efficientnet-v1",
            "_metrics"  : {
                "ImageNet-1K": {
                    "acc@1": 83.384,
                    "acc@5": 96.594,
                }
            },
            "_ops"      : 4.394,
            "_file_size": 74.489,
            "_docs"     : """These weights are ported from the original paper.""",
        }
    )
    DEFAULT = IMAGENET1K_V1


@WEIGHTS.register(arch="efficientnet", name="efficientnet_b5")
class EfficientNet_B5_Weights(WeightsEnum):

    IMAGENET1K_V1 = Weights(
        url         = "https://download.pytorch.org/models/efficientnet_b5_lukemelas-1a07897c.pth",
        path        = ZOO_DIR / "nn/backbone/efficientnet/efficientnet_b5/imagenet1k_v1/efficientnet_b5_imagenet1k_v1.pth",
        num_classes = 1000,
        transforms  = None,
        meta        = {
            "num_params": 30389784,
            "min_size"  : (1, 1),
            "categories": _IMAGENET_CATEGORIES,
            "recipe"    : "https://github.com/pytorch/vision/tree/main/references/classification#efficientnet-v1",
            "_metrics"  : {
                "ImageNet-1K": {
                    "acc@1": 83.444,
                    "acc@5": 96.628,
                }
            },
            "_ops"      : 10.266,
            "_file_size": 116.864,
            "_docs"     : """These weights are ported from the original paper.""",
        }
    )
    DEFAULT = IMAGENET1K_V1


@WEIGHTS.register(arch="efficientnet", name="efficientnet_b6")
class EfficientNet_B6_Weights(WeightsEnum):

    IMAGENET1K_V1 = Weights(
        url         = "https://download.pytorch.org/models/efficientnet_b6_lukemelas-24a108a5.pth",
        path        = ZOO_DIR / "nn/backbone/efficientnet/efficientnet_b6/imagenet1k_v1/efficientnet_b6_imagenet1k_v1.pth",
        num_classes = 1000,
        transforms  = None,
        meta        = {
            "num_params": 43040704,
            "min_size"  : (1, 1),
            "categories": _IMAGENET_CATEGORIES,
            "recipe"    : "https://github.com/pytorch/vision/tree/main/references/classification#efficientnet-v1",
            "_metrics"  : {
                "ImageNet-1K": {
                    "acc@1": 84.008,
                    "acc@5": 96.916,
                }
            },
            "_ops"      : 19.068,
            "_file_size": 165.362,
            "_docs"     : """These weights are ported from the original paper.""",
        }
    )
    DEFAULT = IMAGENET1K_V1


@WEIGHTS.register(arch="efficientnet", name="efficientnet_b7")
class EfficientNet_B7_Weights(WeightsEnum):

    IMAGENET1K_V1 = Weights(
        url         = "https://download.pytorch.org/models/efficientnet_b7_lukemelas-c5b4e57e.pth",
        path        = ZOO_DIR / "nn/backbone/efficientnet/efficientnet_b7/imagenet1k_v1/efficientnet_b7_imagenet1k_v1.pth",
        num_classes = 1000,
        transforms  = None,
        meta        = {
            "num_params": 66347960,
            "min_size"  : (1, 1),
            "categories": _IMAGENET_CATEGORIES,
            "recipe"    : "https://github.com/pytorch/vision/tree/main/references/classification#efficientnet-v1",
            "_metrics"  : {
                "ImageNet-1K": {
                    "acc@1": 84.122,
                    "acc@5": 96.908,
                }
            },
            "_ops"      : 37.746,
            "_file_size": 254.675,
            "_docs"     : """These weights are ported from the original paper.""",
        }
    )
    DEFAULT = IMAGENET1K_V1


@WEIGHTS.register(arch="efficientnet_v2", name="efficientnet_v2_s")
class EfficientNet_V2_S_Weights(WeightsEnum):

    IMAGENET1K_V1 = Weights(
        url         = "https://download.pytorch.org/models/efficientnet_v2_s-dd5fe13b.pth",
        path        = ZOO_DIR / "nn/backbone//efficientnet/efficientnet_v2_s/imagenet1k_v1/efficientnet_v2_s_imagenet1k_v1.pth",
        num_classes = 1000,
        transforms  = None,
        meta        = {
            "num_params": 21458488,
            "min_size"  : (33, 33),
            "categories": _IMAGENET_CATEGORIES,
            "recipe"    : "https://github.com/pytorch/vision/tree/main/references/classification#efficientnet-v2",
            "_metrics"  : {
                "ImageNet-1K": {
                    "acc@1": 84.228,
                    "acc@5": 96.878,
                }
            },
            "_ops"      : 8.366,
            "_file_size": 82.704,
            "_docs"     : """
                These weights improve upon the results of the original paper by using a modified version of TorchVision's
                `new training recipe
                <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
            """,
        }
    )
    DEFAULT = IMAGENET1K_V1


@WEIGHTS.register(arch="efficientnet_v2", name="efficientnet_v2_m")
class EfficientNet_V2_M_Weights(WeightsEnum):

    IMAGENET1K_V1 = Weights(
        url         = "https://download.pytorch.org/models/efficientnet_v2_m-dc08266a.pth",
        path        = ZOO_DIR / "nn/backbone/efficientnet/efficientnet_v2_m/imagenet1k_v1/efficientnet_v2_m_imagenet1k_v1.pth",
        num_classes = 1000,
        transforms  = None,
        meta        = {
            "num_params": 54139356,
            "min_size"  : (33, 33),
            "categories": _IMAGENET_CATEGORIES,
            "recipe"    : "https://github.com/pytorch/vision/tree/main/references/classification#efficientnet-v2",
            "_metrics"  : {
                "ImageNet-1K": {
                    "acc@1": 85.112,
                    "acc@5": 97.156,
                }
            },
            "_ops"      : 24.582,
            "_file_size": 208.01,
            "_docs"     : """
                These weights improve upon the results of the original paper by using a modified version of TorchVision's
                `new training recipe
                <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
            """,
        }
    )
    DEFAULT = IMAGENET1K_V1


@WEIGHTS.register(arch="efficientnet_v2", name="efficientnet_v2_l")
class EfficientNet_V2_L_Weights(WeightsEnum):

    IMAGENET1K_V1 = Weights(
        url         = "https://download.pytorch.org/models/efficientnet_v2_l-59c71312.pth",
        path        = ZOO_DIR / "nn/backbone/efficientnet/efficientnet_v2_l/imagenet1k_v1/efficientnet_v2_l_imagenet1k_v1.pth",
        num_classes = 1000,
        transforms  = None,
        meta        = {
            "num_params": 118515272,
            "min_size"  : (33, 33),
            "categories": _IMAGENET_CATEGORIES,
            "recipe"    : "https://github.com/pytorch/vision/tree/main/references/classification#efficientnet-v2",
            "_metrics"  : {
                "ImageNet-1K": {
                    "acc@1": 85.808,
                    "acc@5": 97.788,
                }
            },
            "_ops"      : 56.08,
            "_file_size": 454.573,
            "_docs"     : """These weights are ported from the original paper.""",
        }
    )
    DEFAULT = IMAGENET1K_V1


# --- Model Variants ---

@BACKBONES.register(name="efficientnet_b0")
def efficientnet_b0(
    weights    : WeightsEnum | str | None = EfficientNet_B0_Weights.DEFAULT,
    out_indices: list | None = None,
    *args, **kwargs
):
    """Create an EfficientNet-B0 backbone.

    Args:
        weights: Pre-trained weights to load. Defaults to
            EfficientNet_B0_Weights.DEFAULT.
        out_indices: List of layer indices to extract features from. Defaults
            to None.
        args: Additional positional arguments for the ResNet model.
        kwargs: Additional keyword arguments for the ResNet model.

    Returns:
        An EfficientNet-B0 backbone model.
    """
    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_b0", width_mult=1.0, depth_mult=1.0)
    return EfficientNetBackBone(
        name         = "efficientnet_b0",
        inverted_residual_setting = inverted_residual_setting,
        dropout      = kwargs.pop("dropout", 0.2),
        last_channel = last_channel,
        weights      = EfficientNet_B0_Weights(weights),
        out_indices  = out_indices,
        *args, **kwargs
    )


@BACKBONES.register(name="efficientnet_b1")
def efficientnet_b1(
    weights    : WeightsEnum | str | None = EfficientNet_B1_Weights.DEFAULT,
    out_indices: list | None = None,
    *args, **kwargs
):
    """Create an EfficientNet-B1 backbone.

    Args:
        weights: Pre-trained weights to load. Defaults to
            EfficientNet_B1_Weights.DEFAULT.
        out_indices: List of layer indices to extract features from. Defaults
            to None.
        args: Additional positional arguments for the ResNet model.
        kwargs: Additional keyword arguments for the ResNet model.

    Returns:
        An EfficientNet-B1 backbone model.
    """
    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_b1", width_mult=1.0, depth_mult=1.1)
    return EfficientNetBackBone(
        name         = "efficientnet_b1",
        inverted_residual_setting = inverted_residual_setting,
        dropout      = kwargs.pop("dropout", 0.2),
        last_channel = last_channel,
        weights      = EfficientNet_B1_Weights(weights),
        out_indices  = out_indices,
        *args, **kwargs
    )


@BACKBONES.register(name="efficientnet_b2")
def efficientnet_b2(
    weights    : WeightsEnum | str = EfficientNet_B2_Weights.DEFAULT,
    out_indices: list | None = None,
    *args, **kwargs
):
    """Create an EfficientNet-B2 backbone.

    Args:
        weights: Pre-trained weights to load. Defaults to
            EfficientNet_B2_Weights.DEFAULT.
        out_indices: List of layer indices to extract features from. Defaults
            to None.
        args: Additional positional arguments for the ResNet model.
        kwargs: Additional keyword arguments for the ResNet model.

    Returns:
        An EfficientNet-B2 backbone model.
    """
    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_b2", width_mult=1.1, depth_mult=1.2)
    return EfficientNetBackBone(
        name         = "efficientnet_b2",
        inverted_residual_setting = inverted_residual_setting,
        dropout      = kwargs.pop("dropout", 0.3),
        last_channel = last_channel,
        weights      = EfficientNet_B2_Weights(weights),
        out_indices  = out_indices,
        *args, **kwargs
    )


@BACKBONES.register(name="efficientnet_b3")
def efficientnet_b3(
    weights    : WeightsEnum | str = EfficientNet_B3_Weights.DEFAULT,
    out_indices: list | None = None,
    *args, **kwargs
):
    """Create an EfficientNet-B3 backbone.

    Args:
        weights: Pre-trained weights to load. Defaults to
            EfficientNet_B3_Weights.DEFAULT.
        out_indices: List of layer indices to extract features from. Defaults
            to None.
        args: Additional positional arguments for the ResNet model.
        kwargs: Additional keyword arguments for the ResNet model.

    Returns:
        An EfficientNet-B3 backbone model.
    """
    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_b3", width_mult=1.2, depth_mult=1.4)
    return EfficientNetBackBone(
        name         = "efficientnet_b3",
        inverted_residual_setting = inverted_residual_setting,
        dropout      = kwargs.pop("dropout", 0.3),
        last_channel = last_channel,
        weights      = EfficientNet_B3_Weights(weights),
        out_indices  = out_indices,
        *args, **kwargs
    )


@BACKBONES.register(name="efficientnet_b4")
def efficientnet_b4(
    weights    : WeightsEnum | str | None = EfficientNet_B4_Weights.DEFAULT,
    out_indices: list | None = None,
    *args, **kwargs
):
    """Create an EfficientNet-B4 backbone.

    Args:
        weights: Pre-trained weights to load. Defaults to
            EfficientNet_B4_Weights.DEFAULT.
        out_indices: List of layer indices to extract features from. Defaults
            to None.
        args: Additional positional arguments for the ResNet model.
        kwargs: Additional keyword arguments for the ResNet model.

    Returns:
        An EfficientNet-B4 backbone model.
    """
    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_b4", width_mult=1.4, depth_mult=1.8)
    return EfficientNetBackBone(
        name         = "efficientnet_b4",
        inverted_residual_setting = inverted_residual_setting,
        dropout      = kwargs.pop("dropout", 0.4),
        last_channel = last_channel,
        weights      = EfficientNet_B4_Weights(weights),
        out_indices  = out_indices,
        *args, **kwargs
    )


@BACKBONES.register(name="efficientnet_b5")
def efficientnet_b5(
    weights    : WeightsEnum | str | None = EfficientNet_B5_Weights.DEFAULT,
    out_indices: list | None = None,
    *args, **kwargs
):
    """Create an EfficientNet-B5 backbone.

    Args:
        weights: Pre-trained weights to load. Defaults to
            EfficientNet_B5_Weights.DEFAULT.
        out_indices: List of layer indices to extract features from. Defaults
            to None.
        args: Additional positional arguments for the ResNet model.
        kwargs: Additional keyword arguments for the ResNet model.

    Returns:
        An EfficientNet-B5 backbone model.
    """
    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_b5", width_mult=1.6, depth_mult=2.2)
    return EfficientNetBackBone(
        name         = "efficientnet_b5",
        inverted_residual_setting = inverted_residual_setting,
        dropout      = kwargs.pop("dropout", 0.4),
        last_channel = last_channel,
        norm_layer   = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01),
        weights      = EfficientNet_B5_Weights(weights),
        out_indices  = out_indices,
        *args, **kwargs
    )


@BACKBONES.register(name="efficientnet_b6")
def efficientnet_b6(
    weights    : WeightsEnum | str | None = EfficientNet_B6_Weights.DEFAULT,
    out_indices: list | None = None,
    *args, **kwargs
):
    """Create an EfficientNet-B6 backbone.

    Args:
        weights: Pre-trained weights to load. Defaults to
            EfficientNet_B6_Weights.DEFAULT.
        out_indices: List of layer indices to extract features from. Defaults
            to None.
        args: Additional positional arguments for the ResNet model.
        kwargs: Additional keyword arguments for the ResNet model.

    Returns:
        An EfficientNet-B6 backbone model.
    """
    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_b6", width_mult=1.8, depth_mult=2.6)
    return EfficientNetBackBone(
        name         = "efficientnet_b6",
        inverted_residual_setting = inverted_residual_setting,
        dropout      = kwargs.pop("dropout", 0.5),
        last_channel = last_channel,
        norm_layer   = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01),
        weights      = EfficientNet_B6_Weights(weights),
        out_indices  = out_indices,
        *args, **kwargs
    )


@BACKBONES.register(name="efficientnet_b7")
def efficientnet_b7(
    weights    : WeightsEnum | str | None = EfficientNet_B7_Weights.DEFAULT,
    out_indices: list | None = None,
    *args, **kwargs
):
    """Create an EfficientNet-B7 backbone.

    Args:
        weights: Pre-trained weights to load. Defaults to
            EfficientNet_B7_Weights.DEFAULT.
        out_indices: List of layer indices to extract features from. Defaults
            to None.
        args: Additional positional arguments for the ResNet model.
        kwargs: Additional keyword arguments for the ResNet model.

    Returns:
        An EfficientNet-B7 backbone model.
    """
    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_b7", width_mult=2.0, depth_mult=3.1)
    return EfficientNetBackBone(
        name         = "efficientnet_b7",
        inverted_residual_setting = inverted_residual_setting,
        dropout      = kwargs.pop("dropout", 0.5),
        last_channel = last_channel,
        norm_layer   = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01),
        weights      = EfficientNet_B7_Weights(weights),
        out_indices  = out_indices,
        *args, **kwargs
    )


@BACKBONES.register(name="efficientnet_v2_s")
def efficientnet_v2_s(
    weights    : WeightsEnum | str | None = EfficientNet_V2_S_Weights.DEFAULT,
    out_indices: list | None = None,
    *args, **kwargs
):
    """Create an EfficientNet-V2-S backbone.

    Args:
        weights: Pre-trained weights to load. Defaults to
            EfficientNet_V2_S_Weights.DEFAULT.
        out_indices: List of layer indices to extract features from. Defaults
            to None.
        args: Additional positional arguments for the ResNet model.
        kwargs: Additional keyword arguments for the ResNet model.

    Returns:
        An EfficientNet-V2-S backbone model.
    """
    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_v2_s")
    return EfficientNetBackBone(
        name         = "efficientnet_v2_s",
        inverted_residual_setting = inverted_residual_setting,
        dropout      = kwargs.pop("dropout", 0.2),
        last_channel = last_channel,
        norm_layer   = partial(nn.BatchNorm2d, eps=1e-03),
        weights      = EfficientNet_V2_S_Weights(weights),
        out_indices  = out_indices,
        *args, **kwargs
    )


@BACKBONES.register(name="efficientnet_v2_m")
def efficientnet_v2_m(
    weights    : WeightsEnum | str | None = EfficientNet_V2_M_Weights.DEFAULT,
    out_indices: list | None = None,
    *args, **kwargs
):
    """Create an EfficientNet-V2-M backbone.

    Args:
        weights: Pre-trained weights to load. Defaults to
            EfficientNet_V2_M_Weights.DEFAULT.
        out_indices: List of layer indices to extract features from. Defaults
            to None.
        args: Additional positional arguments for the ResNet model.
        kwargs: Additional keyword arguments for the ResNet model.

    Returns:
        An EfficientNet-V2-M backbone model.
    """
    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_v2_m")
    return EfficientNetBackBone(
        name         = "efficientnet_v2_m",
        inverted_residual_setting = inverted_residual_setting,
        dropout      = kwargs.pop("dropout", 0.3),
        last_channel = last_channel,
        norm_layer   = partial(nn.BatchNorm2d, eps=1e-03),
        weights      = EfficientNet_V2_M_Weights(weights),
        out_indices  = out_indices,
        *args, **kwargs
    )


@BACKBONES.register(name="efficientnet_v2_l")
def efficientnet_v2_l(
    weights    : WeightsEnum | str | None = EfficientNet_V2_L_Weights.DEFAULT,
    out_indices: list | None = None,
    *args, **kwargs
):
    """Create an EfficientNet-V2-L backbone.

    Args:
        weights: Pre-trained weights to load. Defaults to
            EfficientNet_V2_L_Weights.DEFAULT.
        out_indices: List of layer indices to extract features from. Defaults
            to None.
        args: Additional positional arguments for the ResNet model.
        kwargs: Additional keyword arguments for the ResNet model.

    Returns:
        An EfficientNet-V2-L backbone model.
    """
    inverted_residual_setting, last_channel = _efficientnet_conf("efficientnet_v2_l")
    return EfficientNetBackBone(
        name         = "efficientnet_v2_l",
        inverted_residual_setting = inverted_residual_setting,
        dropout      = kwargs.pop("dropout", 0.4),
        last_channel = last_channel,
        norm_layer   = partial(nn.BatchNorm2d, eps=1e-03),
        weights      = EfficientNet_V2_L_Weights(weights),
        out_indices  = out_indices,
        *args, **kwargs
    )

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    model_ = efficientnet_b0(weights="default")
    x = torch.ones(1, 3, 224, 224)
    y = model_(x)
    print(model_.features)
    print(x)
    print(y)

# endregion

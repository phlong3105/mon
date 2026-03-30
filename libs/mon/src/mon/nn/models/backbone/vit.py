#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Vision Transformer (ViT) backbones.

This module provides various ViT backbones using PyTorch.
"""

from __future__ import annotations

__all__ = [
    "ViT_B_16_Weights",
    "ViT_B_32_Weights",
    "ViT_H_14_Weights",
    "ViT_L_16_Weights",
    "ViT_L_32_Weights",
    "vit_b_16",
    "vit_b_32",
    "vit_h_14",
    "vit_l_16",
    "vit_l_32",
]

import torch
from torch import nn, Tensor
from torchvision.models._meta import _IMAGENET_CATEGORIES
from torchvision.models.vision_transformer import VisionTransformer

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

class ViTBackBone(ModelRegisterMixin, nn.Module):
    """ViT backbone."""

    arch: str = "vit"
    name: str = "vit"
    tasks: list[Task] = [Task.BACKBONE]
    model_dir: Path = current_dir

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        name: str,
        patch_size: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        weights: WeightsLike | None = None,
        out_indices: list[int] | None = None,
        verbose: bool = True,
        *args, **kwargs,
    ):
        """Initialize a new instance.

        Args:
             name (str): Name of the model variant.
            patch_size (int): Patch size of the model.
            num_layers (int): Number of transformer layers.
            num_heads (int): Number of attention heads.
            hidden_dim (int): Dimension of the hidden layers.
            mlp_dim (int): Dimension of the MLP layers.
            weights (WeightsLike, optional): Pre-trained weights to load.
                Defaults to None.
            out_indices (list[int], optional): List of layer indices to extract
                features from. If None, defaults to [2, 5, 8, 11].
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
            kwargs["image_size"] = weights.meta["min_size"][0]

        base_model = VisionTransformer(
            patch_size=patch_size,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
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

        # In torchvision, ViT already has several components
        self.patch_embed = base_model.conv_proj
        self.class_token = base_model.class_token
        self.pos_embedding = base_model.encoder.pos_embedding
        # Transformer blocks
        self.blocks = base_model.encoder.layers
        self.out_indices = out_indices or [2, 5, 8, 11]
        self.embed_dim = base_model.hidden_dim

    # --- Callable & Context Manager ---
    def forward(self, x: Tensor) -> list[Tensor]:
        """Forward the input through the network.

        Args:
            x (Tensor): Input tensor of shape (B, C, H, W) and values ranging
                from 0.0 to 1.0.

        Returns:
            list[Tensor]: List of feature maps from the specified layers.
        """
        # Patchify and add position
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)

        # Add CLS token
        batch_class_token = self.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat((batch_class_token, x), dim=1)
        x = x + self.pos_embedding

        # If you need multiscale features for a Neck (FPN):
        outputs = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in self.out_indices:
                # Remove CLS token and reshape back to spatial grid
                # For 224x224 image and patch 16, grid is 14x14
                feat = x[:, 1:, :]
                b, n, c = feat.shape
                h = w = int(n ** 0.5)
                feat = feat.transpose(1, 2).reshape(b, c, h, w)
                outputs.append(feat)
        return outputs

# endregion


# ==============================================================================
# region CONCRETE IMPLEMENTATIONS
# ==============================================================================

# --- Pre-trained Weights ---

@WEIGHTS.register(name="vit_b_16")
class ViT_B_16_Weights(WeightsEnum):

    IMAGENET1K_V1 = Weights(
        path=K.ZOO_ROOT / "backbone/vit/vit_b_16/imagenet1k_v1/vit_b_16_imagenet1k_v1.pt",
        url=Path("https://download.pytorch.org/models/vit_b_16-c867db91.pth"),
        num_classes=1000,
        transforms=None,
        meta={
            "num_params": 86567656,
            "min_size": (224, 224),
            "categories": _IMAGENET_CATEGORIES,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#vit_b_16",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 81.072,
                    "acc@5": 95.318,
                },
            },
            "_ops": 17.564,
            "_file_size": 330.285,
            "_docs": """
                These weights were trained from scratch by using a modified version of `DeIT
                <https://arxiv.org/abs/2012.12877>`_'s training recipe.
            """,
        },
    )
    IMAGENET1K_SWAG_E2E_V1 = Weights(
        path=K.ZOO_ROOT / "backbone/vit/vit_b_16/imagenet1k_v1/vit_b_16_swag_imagenet1k_v1.pt",
        url=Path("https://download.pytorch.org/models/vit_b_16_swag-9ac1b537.pth"),
        num_classes=1000,
        transforms=None,
        meta={
            "num_params": 86859496,
            "min_size": (384, 384),
            "categories": _IMAGENET_CATEGORIES,
            "recipe": "https://github.com/facebookresearch/SWAG",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 85.304,
                    "acc@5": 97.650,
                },
            },
            "_ops": 55.484,
            "_file_size": 331.398,
            "_docs": """
                These weights are learnt via transfer learning by end-to-end fine-tuning the original
                `SWAG <https://arxiv.org/abs/2201.08371>`_ weights on ImageNet-1K data.
            """,
        },
    )
    IMAGENET1K_SWAG_LINEAR_V1 = Weights(
        path=K.ZOO_ROOT / "backbone/vit/vit_b_16/imagenet1k_v1/vit_b_16_lc_swag_imagenet1k_v1.pt",
        url=Path("https://download.pytorch.org/models/vit_b_16_lc_swag-4e70ced5.pth"),
        num_classes=1000,
        transforms=None,
        meta={
            "num_params": 86859496,
            "min_size": (224, 224),
            "categories": _IMAGENET_CATEGORIES,
            "recipe": "https://github.com/pytorch/vision/pull/5793",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 81.886,
                    "acc@5": 96.180,
                },
            },
            "_ops": 17.564,
            "_file_size": 330.285,
            "_docs": """
                These weights are composed of the original frozen `SWAG <https://arxiv.org/abs/2201.08371>`_ trunk
                weights and a linear classifier learnt on top of them trained on ImageNet-1K data.
            """,
        },
    )
    DEFAULT = IMAGENET1K_V1


@WEIGHTS.register(name="vit_b_32")
class ViT_B_32_Weights(WeightsEnum):

    IMAGENET1K_V1 = Weights(
        path=K.ZOO_ROOT / "backbone/vit/vit_b_32/imagenet1k_v1/vit_b_32_imagenet1k_v1.pt",
        url=Path("https://download.pytorch.org/models/vit_b_32-d86f8d99.pth"),
        num_classes=1000,
        transforms=None,
        meta={
            "num_params": 88224232,
            "min_size": (224, 224),
            "categories": _IMAGENET_CATEGORIES,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#vit_b_32",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 75.912,
                    "acc@5": 92.466,
                },
            },
            "_ops": 4.409,
            "_file_size": 336.604,
            "_docs": """
                These weights were trained from scratch by using a modified version of `DeIT
                <https://arxiv.org/abs/2012.12877>`_'s training recipe.
            """,
        },
    )
    DEFAULT = IMAGENET1K_V1


@WEIGHTS.register(name="vit_l_16")
class ViT_L_16_Weights(WeightsEnum):

    IMAGENET1K_V1 = Weights(
        path=K.ZOO_ROOT / "backbone/vit/vit_l_16/imagenet1k_v1/vit_l_16_imagenet1k_v1.pt",
        url=Path("https://download.pytorch.org/models/vit_l_16-852ce7e3.pth"),
        num_classes=1000,
        transforms=None,
        meta={
            "num_params": 304326632,
            "min_size": (224, 224),
            "categories": _IMAGENET_CATEGORIES,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#vit_l_16",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 79.662,
                    "acc@5": 94.638,
                },
            },
            "_ops": 61.555,
            "_file_size": 1161.023,
            "_docs": """
                These weights were trained from scratch by using a modified version of TorchVision's
                `new training recipe
                <https://pytorch.org/blog/how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/>`_.
            """,
        },
    )
    IMAGENET1K_SWAG_E2E_V1 = Weights(
        path=K.ZOO_ROOT / "backbone/vit/vit_l_16/imagenet1k_v1/vit_l_16_swag_imagenet1k_v1.pt",
        url=Path("https://download.pytorch.org/models/vit_l_16_swag-4f3808c9.pth"),
        num_classes=1000,
        transforms=None,
        meta={
            "num_params": 305174504,
            "min_size": (512, 512),
            "categories": _IMAGENET_CATEGORIES,
            "recipe": "https://github.com/facebookresearch/SWAG",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 88.064,
                    "acc@5": 98.512,
                },
            },
            "_ops": 361.986,
            "_file_size": 1164.258,
            "_docs": """
                These weights are learnt via transfer learning by end-to-end fine-tuning the original
                `SWAG <https://arxiv.org/abs/2201.08371>`_ weights on ImageNet-1K data.
            """,
        },
    )
    IMAGENET1K_SWAG_LINEAR_V1 = Weights(
        path=K.ZOO_ROOT / "backbone/vit/vit_l_16/imagenet1k_v1/vit_l_16_lc_swag_imagenet1k_v1.pt",
        url=Path("https://download.pytorch.org/models/vit_l_16_lc_swag-4d563306.pth"),
        num_classes=1000,
        transforms=None,
        meta={
            "num_params": 304326632,
            "min_size": (224, 224),
            "categories": _IMAGENET_CATEGORIES,
            "recipe": "https://github.com/pytorch/vision/pull/5793",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 85.146,
                    "acc@5": 97.422,
                },
            },
            "_ops": 61.555,
            "_file_size": 1161.023,
            "_docs": """
                These weights are composed of the original frozen `SWAG <https://arxiv.org/abs/2201.08371>`_ trunk
                weights and a linear classifier learnt on top of them trained on ImageNet-1K data.
            """,
        },
    )
    DEFAULT = IMAGENET1K_V1


@WEIGHTS.register(name="vit_l_32")
class ViT_L_32_Weights(WeightsEnum):

    IMAGENET1K_V1 = Weights(
        path=K.ZOO_ROOT / "backbone/vit/vit_l_32/imagenet1k_v1/vit_l_32_imagenet1k_v1.pt",
        url=Path("https://download.pytorch.org/models/vit_l_32-c7638314.pth"),
        num_classes=1000,
        transforms=None,
        meta={
            "num_params": 306535400,
            "min_size": (224, 224),
            "categories": _IMAGENET_CATEGORIES,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#vit_l_32",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 76.972,
                    "acc@5": 93.07,
                },
            },
            "_ops": 15.378,
            "_file_size": 1169.449,
            "_docs": """
                These weights were trained from scratch by using a modified version of `DeIT
                <https://arxiv.org/abs/2012.12877>`_'s training recipe.
            """,
        },
    )
    DEFAULT = IMAGENET1K_V1


@WEIGHTS.register(name="vit_h_14")
class ViT_H_14_Weights(WeightsEnum):

    IMAGENET1K_V1 = Weights(
        path=K.ZOO_ROOT / "backbone/vit/vit_h_14/imagenet1k_v1/vit_h_14_swag_imagenet1k_v1.pt",
        url=Path("https://download.pytorch.org/models/vit_h_14_swag-80465313.pth"),
        num_classes=1000,
        transforms=None,
        meta={
            "num_params": 633470440,
            "min_size": (518, 518),
            "categories": _IMAGENET_CATEGORIES,
            "recipe": "https://github.com/pytorch/vision/tree/main/references/classification#vit_l_16",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 88.552,
                    "acc@5": 98.694,
                },
            },
            "_ops": 1016.717,
            "_file_size": 2416.643,
            "_docs": """
                These weights are learnt via transfer learning by end-to-end fine-tuning the original
                `SWAG <https://arxiv.org/abs/2201.08371>`_ weights on ImageNet-1K data.
            """,
        },
    )
    IMAGENET1K_SWAG_LINEAR_V1 = Weights(
        path=K.ZOO_ROOT / "backbone/vit/vit_h_14/imagenet1k_v1/vit_h_14_lc_swag_imagenet1k_v1.pt",
        url=Path("https://download.pytorch.org/models/vit_h_14_lc_swag-c1eb923e.pth"),
        num_classes=1000,
        transforms=None,
        meta={
            "num_params": 632045800,
            "min_size": (224, 224),
            "categories": _IMAGENET_CATEGORIES,
            "recipe": "https://github.com/pytorch/vision/pull/5793",
            "_metrics": {
                "ImageNet-1K": {
                    "acc@1": 85.708,
                    "acc@5": 97.730,
                },
            },
            "_ops": 167.295,
            "_file_size": 2411.209,
            "_docs": """
                These weights are composed of the original frozen `SWAG <https://arxiv.org/abs/2201.08371>`_ trunk
                weights and a linear classifier learnt on top of them trained on ImageNet-1K data.
            """,
        },
    )
    DEFAULT = IMAGENET1K_V1


# --- Model Variants ---

@BACKBONES.register(name="vit_b_16", metaclass=ViTBackBone)
def vit_b_16(
    weights: WeightsLike = "default",
    out_indices: list[int] | None = None,
    *args, **kwargs,
) -> ViTBackBone:
    """Create a ViT-B/16 backbone.

    Args:
        weights (WeightsLike, optional): Pre-trained weights to load.
            Defaults to "default".
        out_indices (list[int], optional): List of layer indices to extract
            features from. Defaults to None.
    """
    return ViTBackBone(
        name="vit_b_16",
        patch_size=16,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
        weights=ViT_B_16_Weights(weights),
        out_indices=out_indices,
        *args, **kwargs,
    )


@BACKBONES.register(name="vit_b_32", metaclass=ViTBackBone)
def vit_b_32(
    weights: WeightsLike = "default",
    out_indices: list[int] | None = None,
    *args, **kwargs,
) -> ViTBackBone:
    """Create a ViT-B/32 backbone.

    Args:
        weights (WeightsLike, optional): Pre-trained weights to load.
            Defaults to "default".
        out_indices (list[int], optional): List of layer indices to extract
            features from. Defaults to None.
    """
    return ViTBackBone(
        name="vit_b_32",
        patch_size=32,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
        weights=ViT_B_32_Weights(weights),
        out_indices=out_indices,
        *args, **kwargs,
    )


@BACKBONES.register(name="vit_l_16", metaclass=ViTBackBone)
def vit_l_16(
    weights: WeightsLike = "default",
    out_indices: list[int] | None = None,
    *args, **kwargs,
) -> ViTBackBone:
    """Create a ViT-L/16 backbone.

    Args:
        weights (WeightsLike, optional): Pre-trained weights to load.
            Defaults to "default".
        out_indices (list[int], optional): List of layer indices to extract
            features from. Defaults to None.
    """
    return ViTBackBone(
        name="vit_l_16",
        patch_size=16,
        num_layers=24,
        num_heads=16,
        hidden_dim=1024,
        mlp_dim=4096,
        weights=ViT_L_16_Weights(weights),
        out_indices=out_indices,
        *args, **kwargs,
    )


@BACKBONES.register(name="vit_l_32", metaclass=ViTBackBone)
def vit_l_32(
    weights: WeightsLike = "default",
    out_indices: list[int] | None = None,
    *args, **kwargs,
) -> ViTBackBone:
    """Create a ViT-L/32 backbone.

    Args:
        weights (WeightsLike, optional): Pre-trained weights to load.
            Defaults to "default".
        out_indices (list[int], optional): List of layer indices to extract
            features from. Defaults to None.
    """
    return ViTBackBone(
        name="vit_l_32",
        patch_size=32,
        num_layers=24,
        num_heads=16,
        hidden_dim=1024,
        mlp_dim=4096,
        weights=ViT_L_32_Weights(weights),
        out_indices=out_indices,
        *args, **kwargs,
    )


@BACKBONES.register(name="vit_h_14", metaclass=ViTBackBone)
def vit_h_14(
    weights: WeightsLike = "default",
    out_indices: list[int] | None = None,
    *args, **kwargs,
) -> ViTBackBone:
    """Create a ViT-H/14 backbone.

    Args:
        weights (WeightsLike, optional): Pre-trained weights to load.
            Defaults to "default".
        out_indices (list[int], optional): List of layer indices to extract
            features from. Defaults to None.
    """
    return ViTBackBone(
        name="vit_h_14",
        patch_size=14,
        num_layers=32,
        num_heads=16,
        hidden_dim=1280,
        mlp_dim=5120,
        weights=ViT_H_14_Weights(weights),
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

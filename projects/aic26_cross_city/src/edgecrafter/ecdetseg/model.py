#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""EdgeCrafter Models.

This module provides the EdgeCrafter definition and pre-trained weights.

References:
    - Code: https://github.com/Intellindust-AI-Lab/EdgeCrafter
"""

from __future__ import annotations

__all__ = [
    "ECDet",
    "ECDet_L_Weights",
    "ECDet_M_Weights",
    "ECDet_S_Weights",
    "ECDet_X_Weights",
    "ECSeg",
    "ECSeg_L_Weights",
    "ECSeg_M_Weights",
    "ECSeg_S_Weights",
    "ECSeg_X_Weights",
    "ecdet_l",
    "ecdet_m",
    "ecdet_s",
    "ecdet_x",
    "ecseg_l",
    "ecseg_m",
    "ecseg_s",
    "ecseg_x",
]

from typing import override

import torch
from tensordict import TensorDict
from torch import Tensor

from mon.core import (
    is_weights_type,
    K,
    log,
    MODELS,
    Path,
    Size,
    Strategy,
    Task,
    WEIGHTS,
    Weights,
    WeightsEnum,
    WeightsLike,
)
from mon.metrics import benchmark, create_dummy_image
from mon.nn import Model, ModelRegisterMixin
from .engine.edgecrafter.decoder import ECTransformer
from .engine.edgecrafter.ecvit import ViTAdapter
from .engine.edgecrafter.hybrid_encoder import HybridEncoder
from .engine.edgecrafter.postprocessor import PostProcessor

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region BASE CLASSES
# ==============================================================================

class ECDet(ModelRegisterMixin, Model):
    """EdgeCrafter model for object detection.

    This class is meant to be used in inference mode.

    References:
        - Paper: "EdgeCrafter: Compact ViTs for Edge Dense Prediction via
          Task-Specialized Distillation," arXiv 2026.
        - Code: https://github.com/Intellindust-AI-Lab/EdgeCrafter/tree/main/ecdetseg
    """

    arch: str = "ecdet"
    name: str = "ecdet"
    tasks: list[Task] = [Task.DETECT]
    strategies: list[Strategy] = [Strategy.RESIZE]
    model_dir: Path = current_dir

    in_keys: set = {"image"}
    out_keys: set = {"labels", "boxes", "scores"}
    debug_keys: set = {}

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        name: str,
        backbone: dict,
        encoder: dict,
        decoder: dict,
        postprocessor: dict,
        weights: Weights | None = None,
        verbose: bool = True,
        *args, **kwargs
    ):
        """Initialize a new instance.

        Args:
            name (str): Name of the model variant.
            backbone (dict): Backbone configuration dictionary.
            encoder (dict): Encoder configuration dictionary.
            decoder (dict): Decoder configuration dictionary.
            postprocessor (dict): Post-processor configuration dictionary.
            weights (Weights | None, optional): Pre-trained weights to load.
                Defaults to None.
            verbose (bool, optional): Verbosity mode. Defaults to True.
        """
        super().__init__(name=name)

        # Assign attributes
        self.verbose = verbose

        # Define network
        if weights is not None and is_weights_type(weights):
            num_classes = weights.num_classes
        else:
            num_classes = kwargs.pop("num_classes", None)
        if num_classes is not None:
            decoder["num_classes"] = num_classes
            postprocessor["num_classes"] = num_classes

        self.backbone = ViTAdapter(**backbone)
        self.encoder = HybridEncoder(**encoder)
        self.decoder = ECTransformer(**decoder)
        self.postprocessor = PostProcessor(**postprocessor).deploy()

        # Load weights
        if weights is not None and is_weights_type(weights):
            state_dict = weights.state_dict()
            state_dict = state_dict["ema"]["module"] if "ema" in state_dict else state_dict["model"]
            self.load_state_dict(state_dict)
            if self.verbose:
                log(f"initialized {name} from weights {weights.path.as_posix()}.")
        else:
            if self.verbose:
                log(f"Initialized {name} from scratch.")

    # --- Callable & Context Manager ---
    @override
    def forward(
        self,
        image: Tensor,
        orig_imgsz: Size | None = None,
        *args, **kwargs
    ) -> tuple[Tensor, ...]:
        """Route the inputs through the model's different forward methods based
        on the context.

        Args:
            image (Tensor): Input image tensor of shape (B, C, H, W) and values
                ranging from 0.0 to 1.0.
            orig_imgsz (Size | None, optional): Original image size. Defaults to None.

        Returns:
            tuple[Tensor, ...]: A tuple containing:

                - labels (Tensor): Class labels tensor of shape (B, N) and
                  values ranging from 0 to num_classes-1.
                - boxes (Tensor): Bounding boxes tensor of shape (B, N, 4) in
                  XYXY format.
                - scores (Tensor): Class scores tensor of shape (B, N) and
                  values ranging from 0.0 to 1.0.
        """
        return self.forward_step(image=image, orig_imgsz=orig_imgsz, *args, **kwargs)

    @override
    def forward_step(
        self,
        image: Tensor,
        orig_imgsz: Size | None = None,
        *args, **kwargs
    ) -> tuple[Tensor, ...]:
        """Perform a single forward step of the model.

        Args:
            image (Tensor): Input image tensor of shape (B, C, H, W) and values
                ranging from 0.0 to 1.0.
            orig_imgsz (Size | None, optional): Original image size. Defaults to None.

        Returns:
            tuple[Tensor, ...]: A tuple containing:

                - labels (Tensor): Class labels tensor of shape (B, N) and
                  values ranging from 0 to num_classes-1.
                - boxes (Tensor): Bounding boxes tensor of shape (B, N, 4) in
                  XYXY format.
                - scores (Tensor): Class scores tensor of shape (B, N) and
                  values ranging from 0.0 to 1.0.
        """
        device = image.device
        h, w = orig_imgsz.hw if orig_imgsz is not None else image.shape[-2:]
        orig_target_sizes = torch.tensor([[h, w]], device=device)

        # 1. Network forward
        x = image
        x = self.backbone(x)
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.postprocessor(x, orig_target_sizes)

        # 2. Return final and intermediate results for debugging
        labels, boxes, scores = x
        return labels, boxes, scores

    # --- Benchmark ---
    @override
    def benchmark(self, imgsz: Size, *args, **kwargs) -> dict[str, float]:
        """Perform a single forward step of the model to benchmark its performance.

        Args:
            imgsz (Size): Input image size.
            **kwargs: Additional arguments for benchmarking, such as number
                of runs, device, etc.

        Returns:
            dict[str, float]: A dictionary containing the benchmark results,
                such as latency, FLOPs, and parameter count.
        """
        device = next(self.parameters()).device

        # Create dummy inputs
        dummy_input = create_dummy_image(imgsz=imgsz, device=device)
        data = TensorDict({"image": dummy_input}, batch_size=dummy_input.shape)
        inputs = {
            "data": data,
            "orig_imgsz": imgsz,
        }

        # Benchmark the model
        return benchmark(model=self, inputs=inputs, *args, **kwargs)


class ECSeg(ModelRegisterMixin, Model):
    """EdgeCrafter model for instance segmentation.

    This class is meant to be used in inference mode.

    References:
        - Paper: "EdgeCrafter: Compact ViTs for Edge Dense Prediction via
          Task-Specialized Distillation," arXiv 2026.
        - Code: https://github.com/Intellindust-AI-Lab/EdgeCrafter/tree/main/ecdetseg
    """

    arch: str = "ecseg"
    name: str = "ecseg"
    tasks: list[Task] = [Task.SEGMENT]
    strategies: list[Strategy] = [Strategy.RESIZE]
    model_dir: Path = current_dir

    in_keys: set = {"image"}
    out_keys: set = {"labels", "boxes", "scores", "masks"}
    debug_keys: set = {}

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        name: str,
        backbone: dict,
        encoder: dict,
        decoder: dict,
        postprocessor: dict,
        weights: Weights | None = None,
        verbose: bool = True,
        *args, **kwargs
    ):
        """Initialize a new instance.

        Args:
            name (str): Name of the model variant.
            backbone (dict): Backbone configuration dictionary.
            encoder (dict): Encoder configuration dictionary.
            decoder (dict): Decoder configuration dictionary.
            postprocessor (dict): Post-processor configuration dictionary.
            weights (Weights | None, optional): Pre-trained weights to load.
                Defaults to None.
            verbose (bool, optional): Verbosity mode. Defaults to True.
        """
        super().__init__(name=name)

        # Assign attributes
        self.verbose = verbose

        # Define network
        if weights is not None and is_weights_type(weights):
            num_classes = weights.num_classes
        else:
            num_classes = kwargs.pop("num_classes", None)
        if num_classes is not None:
            decoder["num_classes"] = num_classes
            postprocessor["num_classes"] = num_classes

        self.backbone = ViTAdapter(**backbone)
        self.encoder = HybridEncoder(**encoder)
        self.decoder = ECTransformer(**decoder)
        self.postprocessor = PostProcessor(**postprocessor).deploy()

        # Load weights
        if weights is not None and is_weights_type(weights):
            state_dict = weights.state_dict()
            state_dict = state_dict["ema"]["module"] if "ema" in state_dict else state_dict["model"]
            self.load_state_dict(state_dict)
            if self.verbose:
                log(f"initialized {name} from weights {weights.path.as_posix()}.")
        else:
            if self.verbose:
                log(f"Initialized {name} from scratch.")

    # --- Callable & Context Manager ---
    @override
    def forward(
        self,
        image: Tensor,
        orig_imgsz: Size | None = None,
        *args, **kwargs
    ) -> tuple[Tensor, ...]:
        """Route the inputs through the model's different forward methods based
        on the context.

        Args:
            image (Tensor): Input image tensor of shape (B, C, H, W) and values
                ranging from 0.0 to 1.0.
            orig_imgsz (Size | None, optional): Original image size. Defaults to None.

        Returns:
            tuple[Tensor, ...]: A tuple containing:

                - labels (Tensor): Class labels tensor of shape (B, N) and
                  values ranging from 0 to num_classes-1.
                - boxes (Tensor): Bounding boxes tensor of shape (B, N, 4) in
                  XYXY format.
                - scores (Tensor): Class scores tensor of shape (B, N) and
                  values ranging from 0.0 to 1.0.
                - masks (Tensor): Masks tensor of shape (B, N, H, W) in binary
                  format.
        """
        return self.forward_step(image=image, orig_imgsz=orig_imgsz, *args, **kwargs)

    @override
    def forward_step(
        self,
        image: Tensor,
        orig_imgsz: Size | None = None,
        *args, **kwargs
    ) -> tuple[Tensor, ...]:
        """Perform a single forward step of the model.

        Args:
            image (Tensor): Input image tensor of shape (B, C, H, W) and values
                ranging from 0.0 to 1.0.
            orig_imgsz (Size | None, optional): Original image size. Defaults to None.

        Returns:
            tuple[Tensor, ...]: A tuple containing:

                - labels (Tensor): Class labels tensor of shape (B, N) and
                  values ranging from 0 to num_classes-1.
                - boxes (Tensor): Bounding boxes tensor of shape (B, N, 4) in
                  XYXY format.
                - scores (Tensor): Class scores tensor of shape (B, N) and
                  values ranging from 0.0 to 1.0.
                - masks (Tensor): Masks tensor of shape (B, N, H, W) in binary
                  format.
        """
        device = image.device
        h, w = orig_imgsz.hw if orig_imgsz is not None else image.shape[-2:]
        orig_target_sizes = torch.tensor([[h, w]], device=device)

        # 1. Network forward
        x = image
        x = self.backbone(x)
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.postprocessor(x, orig_target_sizes)

        # 2. Return final and intermediate results for debugging
        labels, boxes, scores, masks = x
        return labels, boxes, scores, masks

    # --- Benchmark ---
    @override
    def benchmark(self, imgsz: Size, *args, **kwargs) -> dict[str, float]:
        """Perform a single forward step of the model to benchmark its performance.

        Args:
            imgsz (Size): Input image size.
            **kwargs: Additional arguments for benchmarking, such as number
                of runs, device, etc.

        Returns:
            dict[str, float]: A dictionary containing the benchmark results,
                such as latency, FLOPs, and parameter count.
        """
        device = next(self.parameters()).device

        # Create dummy inputs
        dummy_input = create_dummy_image(imgsz=imgsz, device=device)
        data = TensorDict({"image": dummy_input}, batch_size=dummy_input.shape)
        inputs = {
            "data": data,
            "orig_imgsz": imgsz,
        }

        # Benchmark the model
        return benchmark(model=self, inputs=inputs, *args, **kwargs)

# endregion


# ==============================================================================
# region CONCRETE IMPLEMENTATIONS
# ==============================================================================

# --- Pre-trained Weights ---

@WEIGHTS.register(name="ecdet_s")
class ECDet_S_Weights(WeightsEnum):

    COCO = Weights(
        path=K.ZOO_ROOT / "framework/edgecrafter/ecdet/ecdet_s/coco/ecdet_s_coco.pt",
        url="https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecdet_s.pth",
        num_classes=80,
        transforms=None,
        meta={},
    )
    DEFAULT = COCO


@WEIGHTS.register(name="ecdet_m")
class ECDet_M_Weights(WeightsEnum):

    COCO = Weights(
        path=K.ZOO_ROOT / "framework/edgecrafter/ecdet/ecdet_m/coco/ecdet_m_coco.pt",
        url="https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecdet_m.pth",
        num_classes=80,
        transforms=None,
        meta={},
    )
    DEFAULT = COCO


@WEIGHTS.register(name="ecdet_l")
class ECDet_L_Weights(WeightsEnum):

    COCO = Weights(
        path=K.ZOO_ROOT / "framework/edgecrafter/ecdet/ecdet_l/coco/ecdet_l_coco.pt",
        url="https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecdet_l.pth",
        num_classes=80,
        transforms=None,
        meta={},
    )
    DEFAULT = COCO


@WEIGHTS.register(name="ecdet_x")
class ECDet_X_Weights(WeightsEnum):

    COCO = Weights(
        path=K.ZOO_ROOT / "framework/edgecrafter/ecdet/ecdet_x/coco/ecdet_x_coco.pt",
        url="https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecdet_x.pth",
        num_classes=80,
        transforms=None,
        meta={},
    )
    DEFAULT = COCO


@WEIGHTS.register(name="ecseg_s")
class ECSeg_S_Weights(WeightsEnum):

    COCO = Weights(
        path=K.ZOO_ROOT / "framework/edgecrafter/ecseg/ecseg_s/coco/ecseg_s_coco.pt",
        url="https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecseg_s.pth",
        num_classes=80,
        transforms=None,
        meta={},
    )
    DEFAULT = COCO


@WEIGHTS.register(name="ecseg_m")
class ECSeg_M_Weights(WeightsEnum):

    COCO = Weights(
        path=K.ZOO_ROOT / "framework/edgecrafter/ecseg/ecseg_m/coco/ecseg_m_coco.pt",
        url="https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecseg_m.pth",
        num_classes=80,
        transforms=None,
        meta={},
    )
    DEFAULT = COCO


@WEIGHTS.register(name="ecseg_l")
class ECSeg_L_Weights(WeightsEnum):

    COCO = Weights(
        path=K.ZOO_ROOT / "framework/edgecrafter/ecseg/ecseg_l/coco/ecseg_l_coco.pt",
        url="https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecseg_l.pth",
        num_classes=80,
        transforms=None,
        meta={},
    )
    DEFAULT = COCO


@WEIGHTS.register(name="ecseg_x")
class ECSeg_X_Weights(WeightsEnum):

    COCO = Weights(
        path=K.ZOO_ROOT / "framework/edgecrafter/ecseg/ecseg_x/coco/ecseg_x_coco.pt",
        url="https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecseg_x.pth",
        num_classes=80,
        transforms=None,
        meta={},
    )
    DEFAULT = COCO


# --- Model Variants ---

ecdet_s_config = {
    "backbone": {
        "name": "ecvitt",
        "interaction_indexes": [10, 11],
        "embed_dim": 192,
        "num_heads": 3,
        "patch_size": 16,
        "proj_dim": None,
        "num_levels": 3,
        "embed_layer": "ConvPyramidPatchEmbed",
        "ffn_layer": "mlp",
        "ffn_ratio": 4,
        "skip_load_backbone": True,
    },
    "encoder": {
        "in_channels": [192, 192, 192],
        "feat_strides": [8, 16, 32],
        "hidden_dim": 192,
        "nhead": 8,
        "dim_feedforward": 512,
        "dropout": 0.0,
        "use_encoder_idx": [2],
        "num_encoder_layers": 1,
        "pe_temperature": 10000,
        "expansion": 0.34,
        "depth_mult": 0.67,
        "act": "silu",
        "eval_spatial_size": None,
        "csp_type": "csp2",
        "fuse_op": "sum",
    },
    "decoder": {
        "num_classes": 80,
        "hidden_dim": 192,
        "num_queries": 300,
        "feat_channels": [192, 192, 192],
        "feat_strides": [8, 16, 32],
        "num_levels": 3,
        "num_points": [3, 6, 3],
        "nhead": 8,
        "num_layers": 4,
        "dim_feedforward": 512,
        "dropout": 0.0,
        "activation": "silu",
        "num_denoising": 100,
        "label_noise_ratio": 0.5,
        "box_noise_scale": 1.0,
        "learn_query_content": False,
        "eval_spatial_size": [640, 640],
        "eval_idx": -1,
        "eps": 1e-2,
        "aux_loss": True,
        "cross_attn_method": "default",
        "query_select_method": "default",
        "reg_max": 32,
        "reg_scale": 4,
        "layer_scale": 1,
        "share_bbox_head": False,
        "share_score_head": False,
        "mask_downsample_ratio": None,
    },
    "postprocessor": {
        "num_classes": 80,
        "use_focal_loss": True,
        "num_top_queries": 300,
        "remap_mscoco_category": False,
    },
}
ecdet_m_config = {
    "backbone": {
        "name": "ecvittplus",
        "interaction_indexes": [10, 11],
        "embed_dim": 256,
        "num_heads": 4,
        "patch_size": 16,
        "proj_dim": None,
        "num_levels": 3,
        "embed_layer": "ConvPyramidPatchEmbed",
        "ffn_layer": "mlp",
        "ffn_ratio": 4,
        "skip_load_backbone": True,
    },
    "encoder": {
        "in_channels": [256, 256, 256],
        "feat_strides": [8, 16, 32],
        "hidden_dim": 256,
        "nhead": 8,
        "dim_feedforward": 512,
        "dropout": 0.0,
        "use_encoder_idx": [2],
        "num_encoder_layers": 1,
        "pe_temperature": 10000,
        "expansion": 0.75,
        "depth_mult": 0.67,
        "act": "silu",
        "eval_spatial_size": None,
        "csp_type": "csp2",
        "fuse_op": "sum",
    },
    "decoder": {
        "num_classes": 80,
        "hidden_dim": 256,
        "num_queries": 300,
        "feat_channels": [256, 256, 256],
        "feat_strides": [8, 16, 32],
        "num_levels": 3,
        "num_points": [3, 6, 3],
        "nhead": 8,
        "num_layers": 4,
        "dim_feedforward": 1024,
        "dropout": 0.0,
        "activation": "silu",
        "num_denoising": 100,
        "label_noise_ratio": 0.5,
        "box_noise_scale": 1.0,
        "learn_query_content": False,
        "eval_spatial_size": [640, 640],
        "eval_idx": -1,
        "eps": 1e-2,
        "aux_loss": True,
        "cross_attn_method": "default",
        "query_select_method": "default",
        "reg_max": 32,
        "reg_scale": 4,
        "layer_scale": 1,
        "share_bbox_head": False,
        "share_score_head": False,
        "mask_downsample_ratio": None,
    },
    "postprocessor": {
        "num_classes": 80,
        "use_focal_loss": True,
        "num_top_queries": 300,
        "remap_mscoco_category": False,
    },
}
ecdet_l_config = {
    "backbone": {
        "name": "ecvits",
        "interaction_indexes": [10, 11],
        "embed_dim": 384,
        "num_heads": 6,
        "patch_size": 16,
        "proj_dim": 256,
        "num_levels": 3,
        "embed_layer": "ConvPyramidPatchEmbed",
        "ffn_layer": "mlp",
        "ffn_ratio": 4,
        "skip_load_backbone": True,
    },
    "encoder": {
        "in_channels": [256, 256, 256],
        "feat_strides": [8, 16, 32],
        "hidden_dim": 256,
        "nhead": 8,
        "dim_feedforward": 1024,
        "dropout": 0.0,
        "use_encoder_idx": [2],
        "num_encoder_layers": 1,
        "pe_temperature": 10000,
        "expansion": 0.75,
        "depth_mult": 1,
        "act": "silu",
        "eval_spatial_size": None,
        "csp_type": "csp2",
        "fuse_op": "sum",
    },
    "decoder": {
        "num_classes": 80,
        "hidden_dim": 256,
        "num_queries": 300,
        "feat_channels": [256, 256, 256],
        "feat_strides": [8, 16, 32],
        "num_levels": 3,
        "num_points": [3, 6, 3],
        "nhead": 8,
        "num_layers": 4,
        "dim_feedforward": 1024,
        "dropout": 0.0,
        "activation": "silu",
        "num_denoising": 100,
        "label_noise_ratio": 0.5,
        "box_noise_scale": 1.0,
        "learn_query_content": False,
        "eval_spatial_size": [640, 640],
        "eval_idx": -1,
        "eps": 1e-2,
        "aux_loss": True,
        "cross_attn_method": "default",
        "query_select_method": "default",
        "reg_max": 32,
        "reg_scale": 4,
        "layer_scale": 1,
        "share_bbox_head": False,
        "share_score_head": False,
        "mask_downsample_ratio": None,
    },
    "postprocessor": {
        "num_classes": 80,
        "use_focal_loss": True,
        "num_top_queries": 300,
        "remap_mscoco_category": False,
    },
}
ecdet_x_config = {
    "backbone": {
        "name": "ecvitsplus",
        "interaction_indexes": [10, 11],
        "embed_dim": 384,
        "num_heads": 6,
        "patch_size": 16,
        "proj_dim": 256,
        "num_levels": 3,
        "embed_layer": "ConvPyramidPatchEmbed",
        "ffn_layer": "mlp",
        "ffn_ratio": 6,
        "skip_load_backbone": True,
    },
    "encoder": {
        "in_channels": [256, 256, 256],
        "feat_strides": [8, 16, 32],
        "hidden_dim": 256,
        "nhead": 8,
        "dim_feedforward": 2048,
        "dropout": 0.0,
        "use_encoder_idx": [2],
        "num_encoder_layers": 1,
        "pe_temperature": 10000,
        "expansion": 1.5,
        "depth_mult": 1,
        "act": "silu",
        "eval_spatial_size": None,
        "csp_type": "csp2",
        "fuse_op": "sum",
    },
    "decoder": {
        "num_classes": 80,
        "hidden_dim": 256,
        "num_queries": 300,
        "feat_channels": [256, 256, 256],
        "feat_strides": [8, 16, 32],
        "num_levels": 3,
        "num_points": [3, 6, 3],
        "nhead": 8,
        "num_layers": 4,
        "dim_feedforward": 2048,
        "dropout": 0.0,
        "activation": "silu",
        "num_denoising": 100,
        "label_noise_ratio": 0.5,
        "box_noise_scale": 1.0,
        "learn_query_content": False,
        "eval_spatial_size": [640, 640],
        "eval_idx": -1,
        "eps": 1e-2,
        "aux_loss": True,
        "cross_attn_method": "default",
        "query_select_method": "default",
        "reg_max": 32,
        "reg_scale": 4,
        "layer_scale": 1,
        "share_bbox_head": False,
        "share_score_head": False,
        "mask_downsample_ratio": None,
    },
    "postprocessor": {
        "num_classes": 80,
        "use_focal_loss": True,
        "num_top_queries": 300,
        "remap_mscoco_category": False,
    },
}

ecseg_s_config = {
    "backbone": {
        "name": "ecseg_vitt",
        "interaction_indexes": [10, 11],
        "embed_dim": 192,
        "num_heads": 3,
        "patch_size": 16,
        "proj_dim": None,
        "num_levels": 3,
        "embed_layer": "ConvPyramidPatchEmbed",
        "ffn_layer": "mlp",
        "ffn_ratio": 4,
        "skip_load_backbone": True,
    },
    "encoder": {
        "in_channels": [192, 192, 192],
        "feat_strides": [8, 16, 32],
        "hidden_dim": 192,
        "nhead": 8,
        "dim_feedforward": 512,
        "dropout": 0.0,
        "use_encoder_idx": [2],
        "num_encoder_layers": 1,
        "pe_temperature": 10000,
        "expansion": 0.34,
        "depth_mult": 0.67,
        "act": "silu",
        "eval_spatial_size": None,
        "csp_type": "csp2",
        "fuse_op": "sum",
    },
    "decoder": {
        "num_classes": 80,
        "hidden_dim": 192,
        "num_queries": 300,
        "feat_channels": [192, 192, 192],
        "feat_strides": [8, 16, 32],
        "num_levels": 3,
        "num_points": [3, 6, 3],
        "nhead": 8,
        "num_layers": 4,
        "dim_feedforward": 512,
        "dropout": 0.0,
        "activation": "silu",
        "num_denoising": 100,
        "label_noise_ratio": 0.5,
        "box_noise_scale": 1.0,
        "learn_query_content": False,
        "eval_spatial_size": [640, 640],
        "eval_idx": -1,
        "eps": 1e-2,
        "aux_loss": True,
        "cross_attn_method": "default",
        "query_select_method": "default",
        "reg_max": 32,
        "reg_scale": 4,
        "layer_scale": 1,
        "share_bbox_head": False,
        "share_score_head": False,
        "mask_downsample_ratio": 4,
    },
    "postprocessor": {
        "num_classes": 80,
        "use_focal_loss": True,
        "num_top_queries": 300,
        "remap_mscoco_category": False,
    },
}
ecseg_m_config = {
    "backbone": {
        "name": "ecseg_vittplus",
        "interaction_indexes": [10, 11],
        "embed_dim": 256,
        "num_heads": 4,
        "patch_size": 16,
        "proj_dim": None,
        "num_levels": 3,
        "embed_layer": "ConvPyramidPatchEmbed",
        "ffn_layer": "mlp",
        "ffn_ratio": 4,
        "skip_load_backbone": True,
    },
    "encoder": {
        "in_channels": [256, 256, 256],
        "feat_strides": [8, 16, 32],
        "hidden_dim": 256,
        "nhead": 8,
        "dim_feedforward": 512,
        "dropout": 0.0,
        "use_encoder_idx": [2],
        "num_encoder_layers": 1,
        "pe_temperature": 10000,
        "expansion": 0.75,
        "depth_mult": 0.67,
        "act": "silu",
        "eval_spatial_size": None,
        "csp_type": "csp2",
        "fuse_op": "sum",
    },
    "decoder": {
        "num_classes": 80,
        "hidden_dim": 256,
        "num_queries": 300,
        "feat_channels": [256, 256, 256],
        "feat_strides": [8, 16, 32],
        "num_levels": 3,
        "num_points": [3, 6, 3],
        "nhead": 8,
        "num_layers": 4,
        "dim_feedforward": 1024,
        "dropout": 0.0,
        "activation": "silu",
        "num_denoising": 100,
        "label_noise_ratio": 0.5,
        "box_noise_scale": 1.0,
        "learn_query_content": False,
        "eval_spatial_size": [640, 640],
        "eval_idx": -1,
        "eps": 1e-2,
        "aux_loss": True,
        "cross_attn_method": "default",
        "query_select_method": "default",
        "reg_max": 32,
        "reg_scale": 4,
        "layer_scale": 1,
        "share_bbox_head": False,
        "share_score_head": False,
        "mask_downsample_ratio": 4,
    },
    "postprocessor": {
        "num_classes": 80,
        "use_focal_loss": True,
        "num_top_queries": 300,
        "remap_mscoco_category": False,
    },
}
ecseg_l_config = {
    "backbone": {
        "name": "ecseg_vits",
        "interaction_indexes": [10, 11],
        "embed_dim": 384,
        "num_heads": 6,
        "patch_size": 16,
        "proj_dim": 256,
        "num_levels": 3,
        "embed_layer": "ConvPyramidPatchEmbed",
        "ffn_layer": "mlp",
        "ffn_ratio": 4,
        "skip_load_backbone": True,
    },
    "encoder": {
        "in_channels": [256, 256, 256],
        "feat_strides": [8, 16, 32],
        "hidden_dim": 256,
        "nhead": 8,
        "dim_feedforward": 1024,
        "dropout": 0.0,
        "use_encoder_idx": [2],
        "num_encoder_layers": 1,
        "pe_temperature": 10000,
        "expansion": 0.75,
        "depth_mult": 1,
        "act": "silu",
        "eval_spatial_size": None,
        "csp_type": "csp2",
        "fuse_op": "sum",
    },
    "decoder": {
        "num_classes": 80,
        "hidden_dim": 256,
        "num_queries": 300,
        "feat_channels": [256, 256, 256],
        "feat_strides": [8, 16, 32],
        "num_levels": 3,
        "num_points": [3, 6, 3],
        "nhead": 8,
        "num_layers": 4,
        "dim_feedforward": 1024,
        "dropout": 0.0,
        "activation": "silu",
        "num_denoising": 100,
        "label_noise_ratio": 0.5,
        "box_noise_scale": 1.0,
        "learn_query_content": False,
        "eval_spatial_size": [640, 640],
        "eval_idx": -1,
        "eps": 1e-2,
        "aux_loss": True,
        "cross_attn_method": "default",
        "query_select_method": "default",
        "reg_max": 32,
        "reg_scale": 4,
        "layer_scale": 1,
        "share_bbox_head": False,
        "share_score_head": False,
        "mask_downsample_ratio": 4,
    },
    "postprocessor": {
        "num_classes": 80,
        "use_focal_loss": True,
        "num_top_queries": 300,
        "remap_mscoco_category": False,
    },
}
ecseg_x_config = {
    "backbone": {
        "name": "ecseg_vitsplus",
        "interaction_indexes": [10, 11],
        "embed_dim": 384,
        "num_heads": 6,
        "patch_size": 16,
        "proj_dim": 256,
        "num_levels": 3,
        "embed_layer": "ConvPyramidPatchEmbed",
        "ffn_layer": "mlp",
        "ffn_ratio": 6,
        "skip_load_backbone": True,
    },
    "encoder": {
        "in_channels": [256, 256, 256],
        "feat_strides": [8, 16, 32],
        "hidden_dim": 256,
        "nhead": 8,
        "dim_feedforward": 2048,
        "dropout": 0.0,
        "use_encoder_idx": [2],
        "num_encoder_layers": 1,
        "pe_temperature": 10000,
        "expansion": 1.5,
        "depth_mult": 1,
        "act": "silu",
        "eval_spatial_size": None,
        "csp_type": "csp2",
        "fuse_op": "sum",
    },
    "decoder": {
        "num_classes": 80,
        "hidden_dim": 256,
        "num_queries": 300,
        "feat_channels": [256, 256, 256],
        "feat_strides": [8, 16, 32],
        "num_levels": 3,
        "num_points": [3, 6, 3],
        "nhead": 8,
        "num_layers": 4,
        "dim_feedforward": 2048,
        "dropout": 0.0,
        "activation": "silu",
        "num_denoising": 100,
        "label_noise_ratio": 0.5,
        "box_noise_scale": 1.0,
        "learn_query_content": False,
        "eval_spatial_size": [640, 640],
        "eval_idx": -1,
        "eps": 1e-2,
        "aux_loss": True,
        "cross_attn_method": "default",
        "query_select_method": "default",
        "reg_max": 32,
        "reg_scale": 4,
        "layer_scale": 1,
        "share_bbox_head": False,
        "share_score_head": False,
        "mask_downsample_ratio": 4,
    },
    "postprocessor": {
        "num_classes": 80,
        "use_focal_loss": True,
        "num_top_queries": 300,
        "remap_mscoco_category": False,
    },
}


@MODELS.register(name="ecdet_s", metaclass=ECDet)
def ecdet_s(weights: WeightsLike = "default", *args, **kwargs):
    """Create an EdgeCrafter model.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
    """
    _ = kwargs.pop("name", "ecdet_s")
    backbone = kwargs.pop("backbone", ecdet_s_config["backbone"])
    encoder = kwargs.pop("encoder", ecdet_s_config["encoder"])
    decoder = kwargs.pop("decoder", ecdet_s_config["decoder"])
    postprocessor = kwargs.pop("postprocessor", ecdet_s_config["postprocessor"])
    return ECDet(
        name="ecdet_s",
        backbone=backbone,
        encoder=encoder,
        decoder=decoder,
        postprocessor=postprocessor,
        weights=ECDet_S_Weights(weights),
        *args, **kwargs,
    )


@MODELS.register(name="ecdet_m", metaclass=ECDet)
def ecdet_m(weights: WeightsLike = "default", *args, **kwargs):
    """Create an EdgeCrafter model.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
    """
    _ = kwargs.pop("name", "ecdet_m")
    backbone = kwargs.pop("backbone", ecdet_m_config["backbone"])
    encoder = kwargs.pop("encoder", ecdet_m_config["encoder"])
    decoder = kwargs.pop("decoder", ecdet_m_config["decoder"])
    postprocessor = kwargs.pop("postprocessor", ecdet_m_config["postprocessor"])
    return ECDet(
        name="ecdet_m",
        backbone=backbone,
        encoder=encoder,
        decoder=decoder,
        postprocessor=postprocessor,
        weights=ECDet_M_Weights(weights),
        *args, **kwargs,
    )


@MODELS.register(name="ecdet_l", metaclass=ECDet)
def ecdet_l(weights: WeightsLike = "default", *args, **kwargs):
    """Create an EdgeCrafter model.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
    """
    _ = kwargs.pop("name", "ecdet_l")
    backbone = kwargs.pop("backbone", ecdet_l_config["backbone"])
    encoder = kwargs.pop("encoder", ecdet_l_config["encoder"])
    decoder = kwargs.pop("decoder", ecdet_l_config["decoder"])
    postprocessor = kwargs.pop("postprocessor", ecdet_l_config["postprocessor"])
    return ECDet(
        name="ecdet_l",
        backbone=backbone,
        encoder=encoder,
        decoder=decoder,
        postprocessor=postprocessor,
        weights=ECDet_L_Weights(weights),
        *args, **kwargs,
    )


@MODELS.register(name="ecdet_x", metaclass=ECDet)
def ecdet_x(weights: WeightsLike = "default", *args, **kwargs):
    """Create an EdgeCrafter model.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
    """
    _ = kwargs.pop("name", "ecdet_x")
    backbone = kwargs.pop("backbone", ecdet_x_config["backbone"])
    encoder = kwargs.pop("encoder", ecdet_x_config["encoder"])
    decoder = kwargs.pop("decoder", ecdet_x_config["decoder"])
    postprocessor = kwargs.pop("postprocessor", ecdet_x_config["postprocessor"])
    return ECDet(
        name="ecdet_x",
        backbone=backbone,
        encoder=encoder,
        decoder=decoder,
        postprocessor=postprocessor,
        weights=ECDet_X_Weights(weights),
        *args, **kwargs,
    )


@MODELS.register(name="ecseg_s", metaclass=ECSeg)
def ecseg_s(weights: WeightsLike = "default", *args, **kwargs):
    """Create an EdgeCrafter model.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
    """
    _ = kwargs.pop("name", "ecseg_s")
    backbone = kwargs.pop("backbone", ecseg_s_config["backbone"])
    encoder = kwargs.pop("encoder", ecseg_s_config["encoder"])
    decoder = kwargs.pop("decoder", ecseg_s_config["decoder"])
    postprocessor = kwargs.pop("postprocessor", ecseg_s_config["postprocessor"])
    return ECSeg(
        name="ecseg_s",
        backbone=backbone,
        encoder=encoder,
        decoder=decoder,
        postprocessor=postprocessor,
        weights=ECSeg_S_Weights(weights),
        *args, **kwargs,
    )


@MODELS.register(name="ecseg_m", metaclass=ECSeg)
def ecseg_m(weights: WeightsLike = "default", *args, **kwargs):
    """Create an EdgeCrafter model.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
    """
    _ = kwargs.pop("name", "ecseg_m")
    backbone = kwargs.pop("backbone", ecseg_m_config["backbone"])
    encoder = kwargs.pop("encoder", ecseg_m_config["encoder"])
    decoder = kwargs.pop("decoder", ecseg_m_config["decoder"])
    postprocessor = kwargs.pop("postprocessor", ecseg_m_config["postprocessor"])
    return ECSeg(
        name="ecseg_m",
        backbone=backbone,
        encoder=encoder,
        decoder=decoder,
        postprocessor=postprocessor,
        weights=ECSeg_M_Weights(weights),
        *args, **kwargs,
    )


@MODELS.register(name="ecseg_l", metaclass=ECSeg)
def ecseg_l(weights: WeightsLike = "default", *args, **kwargs):
    """Create an EdgeCrafter model.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
    """
    _ = kwargs.pop("name", "ecseg_l")
    backbone = kwargs.pop("backbone", ecseg_l_config["backbone"])
    encoder = kwargs.pop("encoder", ecseg_l_config["encoder"])
    decoder = kwargs.pop("decoder", ecseg_l_config["decoder"])
    postprocessor = kwargs.pop("postprocessor", ecseg_l_config["postprocessor"])
    return ECSeg(
        name="ecseg_l",
        backbone=backbone,
        encoder=encoder,
        decoder=decoder,
        postprocessor=postprocessor,
        weights=ECSeg_L_Weights(weights),
        *args, **kwargs,
    )


@MODELS.register(name="ecseg_x", metaclass=ECSeg)
def ecseg_x(weights: WeightsLike = "default", *args, **kwargs):
    """Create an EdgeCrafter model.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
    """
    _ = kwargs.pop("name", "ecseg_x")
    backbone = kwargs.pop("backbone", ecseg_x_config["backbone"])
    encoder = kwargs.pop("encoder", ecseg_x_config["encoder"])
    decoder = kwargs.pop("decoder", ecseg_x_config["decoder"])
    postprocessor = kwargs.pop("postprocessor", ecseg_x_config["postprocessor"])
    return ECSeg(
        name="ecseg_x",
        backbone=backbone,
        encoder=encoder,
        decoder=decoder,
        postprocessor=postprocessor,
        weights=ECSeg_X_Weights(weights),
        *args, **kwargs,
    )

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion

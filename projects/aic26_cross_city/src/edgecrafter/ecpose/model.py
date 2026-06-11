#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""EdgeCrafter Models.

This module provides the EdgeCrafter definition and pre-trained weights.

References:
    - Code: https://github.com/Intellindust-AI-Lab/EdgeCrafter
"""

from __future__ import annotations

__all__ = [
    "ECPose",
    "ECPose_L_Weights",
    "ECPose_M_Weights",
    "ECPose_S_Weights",
    "ECPose_X_Weights",
    "ecpose_l",
    "ecpose_m",
    "ecpose_s",
    "ecpose_x",
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
from .engine.edgecrafter.detrpose_postprocesses import DETRPosePostProcessor
from .engine.edgecrafter.detrpose_transformer import DETRTransformer
from .engine.edgecrafter.ecvit import ViTAdapter
from .engine.edgecrafter.hybrid_encoder import HybridEncoder

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region BASE CLASSES
# ==============================================================================

class ECPose(ModelRegisterMixin, Model):
    """EdgeCrafter model for pose estimation.

    This class is meant to be used in inference mode.

    References:
        - Paper: "EdgeCrafter: Compact ViTs for Edge Dense Prediction via
          Task-Specialized Distillation," arXiv 2026.
        - Code: https://github.com/Intellindust-AI-Lab/EdgeCrafter/tree/main/ecpose
    """

    arch: str = "ecpose"
    name: str = "ecpose"
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
        self.decoder = DETRTransformer(**decoder)
        self.postprocessor = DETRPosePostProcessor(**postprocessor).deploy()

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

# endregion


# ==============================================================================
# region CONCRETE IMPLEMENTATIONS
# ==============================================================================

# --- Pre-trained Weights ---

@WEIGHTS.register(name="ecpose_s")
class ECPose_S_Weights(WeightsEnum):

    COCO = Weights(
        path=K.ZOO_ROOT / "framework/edgecrafter/ecpose/ecpose_s/coco/ecpose_s_coco.pt",
        url="https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecpose_s.pth",
        num_classes=1,
        transforms=None,
        meta={},
    )
    DEFAULT = COCO


@WEIGHTS.register(name="ecpose_m")
class ECPose_M_Weights(WeightsEnum):

    COCO = Weights(
        path=K.ZOO_ROOT / "framework/edgecrafter/ecpose/ecpose_m/coco/ecpose_m_coco.pt",
        url="https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecpose_m.pth",
        num_classes=1,
        transforms=None,
        meta={},
    )
    DEFAULT = COCO


@WEIGHTS.register(name="ecpose_l")
class ECPose_L_Weights(WeightsEnum):

    COCO = Weights(
        path=K.ZOO_ROOT / "framework/edgecrafter/ecpose/ecpose_l/coco/ecpose_l_coco.pt",
        url="https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecpose_l.pth",
        num_classes=1,
        transforms=None,
        meta={},
    )
    DEFAULT = COCO


@WEIGHTS.register(name="ecpose_x")
class ECPose_X_Weights(WeightsEnum):

    COCO = Weights(
        path=K.ZOO_ROOT / "framework/edgecrafter/ecpose/ecpose_x/coco/ecpose_x_coco.pt",
        url="https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecpose_x.pth",
        num_classes=1,
        transforms=None,
        meta={},
    )
    DEFAULT = COCO


# --- Model Variants ---

ecpose_s_config = {
    "backbone": {
        "name": "ecpose_vitt",
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
        "use_encoder_idx": [2],
        "num_encoder_layers": 1,
        "nhead": 8,
        "dim_feedforward": 512,
        "dropout": 0.0,
        "expansion": 0.34,
        "depth_mult": 0.67,
        "act": "silu",
        "csp_type": "csp2",
        "fuse_op": "sum",
    },
    "decoder": {
        "hidden_dim": 192,
        "dropout": 0.0,
        "nhead": 8,
        "num_queries": 60,
        "dim_feedforward": 512,
        "num_decoder_layers": 3,
        "normalize_before": False,
        "return_intermediate_dec": True,
        "activation": "relu",
        "num_feature_levels": 3,
        "dec_n_points": 4,
        "learnable_tgt_init": True,
        "two_stage_type": "standard",
        "num_body_points": 17,
        "aux_loss": True,
        "dec_pred_class_embed_share": False,
        "dec_pred_pose_embed_share": False,
        "two_stage_class_embed_share": False,
        "two_stage_bbox_embed_share": False,
        "cls_no_bias": False,
        "feat_strides": [8, 16, 32],
        "reg_max": 32,
        "reg_scale": 4,
        "eval_spatial_size": [640, 640],
    },
    "postprocessor": {
        "num_select": 60,
        "num_body_points": 17,
    },
}
ecpose_m_config = {
    "backbone": {
        "name": "ecpose_vittplus",
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
        "use_encoder_idx": [2],
        "num_encoder_layers": 1,
        "nhead": 8,
        "dim_feedforward": 512,
        "dropout": 0.0,
        "expansion": 0.75,
        "depth_mult": 0.67,
        "act": "silu",
        "csp_type": "csp2",
        "fuse_op": "sum",
    },
    "decoder": {
        "hidden_dim": 256,
        "dropout": 0.0,
        "nhead": 8,
        "num_queries": 60,
        "dim_feedforward": 512,
        "num_decoder_layers": 4,
        "normalize_before": False,
        "return_intermediate_dec": True,
        "activation": "relu",
        "num_feature_levels": 3,
        "dec_n_points": 4,
        "learnable_tgt_init": True,
        "two_stage_type": "standard",
        "num_body_points": 17,
        "aux_loss": True,
        "dec_pred_class_embed_share": False,
        "dec_pred_pose_embed_share": False,
        "two_stage_class_embed_share": False,
        "two_stage_bbox_embed_share": False,
        "cls_no_bias": False,
        "feat_strides": [8, 16, 32],
        "reg_max": 32,
        "reg_scale": 4,
        "eval_spatial_size": [640, 640],
    },
    "postprocessor": {
        "num_select": 60,
        "num_body_points": 17,
    },
}
ecpose_l_config = {
    "backbone": {
        "name": "ecpose_vits",
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
        "use_encoder_idx": [2],
        "num_encoder_layers": 1,
        "nhead": 8,
        "dim_feedforward": 1024,
        "dropout": 0.0,
        "expansion": 0.75,
        "depth_mult": 1,
        "act": "silu",
        "csp_type": "csp2",
        "fuse_op": "sum",
    },
    "decoder": {
        "hidden_dim": 256,
        "dropout": 0.0,
        "nhead": 8,
        "num_queries": 60,
        "dim_feedforward": 1024,
        "num_decoder_layers": 4,
        "normalize_before": False,
        "return_intermediate_dec": True,
        "activation": "relu",
        "num_feature_levels": 3,
        "dec_n_points": 4,
        "learnable_tgt_init": True,
        "two_stage_type": "standard",
        "num_body_points": 17,
        "aux_loss": True,
        "dec_pred_class_embed_share": False,
        "dec_pred_pose_embed_share": False,
        "two_stage_class_embed_share": False,
        "two_stage_bbox_embed_share": False,
        "cls_no_bias": False,
        "feat_strides": [8, 16, 32],
        "reg_max": 32,
        "reg_scale": 4,
        "eval_spatial_size": [640, 640],
    },
    "postprocessor": {
        "num_select": 60,
        "num_body_points": 17,
    },
}
ecpose_x_config = {
    "backbone": {
        "name": "ecpose_vitsplus",
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
        "use_encoder_idx": [2],
        "num_encoder_layers": 1,
        "nhead": 8,
        "dim_feedforward": 2048,
        "dropout": 0.0,
        "expansion": 1.5,
        "depth_mult": 1,
        "act": "silu",
        "csp_type": "csp2",
        "fuse_op": "sum",
    },
    "decoder": {
        "hidden_dim": 256,
        "dropout": 0.0,
        "nhead": 8,
        "num_queries": 60,
        "dim_feedforward": 2048,
        "num_decoder_layers": 4,
        "normalize_before": False,
        "return_intermediate_dec": True,
        "activation": "relu",
        "num_feature_levels": 3,
        "dec_n_points": 4,
        "learnable_tgt_init": True,
        "two_stage_type": "standard",
        "num_body_points": 17,
        "aux_loss": True,
        "dec_pred_class_embed_share": False,
        "dec_pred_pose_embed_share": False,
        "two_stage_class_embed_share": False,
        "two_stage_bbox_embed_share": False,
        "cls_no_bias": False,
        "feat_strides": [8, 16, 32],
        "reg_max": 32,
        "reg_scale": 4,
        "eval_spatial_size": [640, 640],
    },
    "postprocessor": {
        "num_select": 60,
        "num_body_points": 17,
    },
}


@MODELS.register(name="ecpose_s", metaclass=ECPose)
def ecpose_s(weights: WeightsLike = "default", *args, **kwargs):
    """Create an EdgeCrafter model.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
    """
    _ = kwargs.pop("name", "ecpose_s")
    backbone = kwargs.pop("backbone", ecpose_s_config["backbone"])
    encoder = kwargs.pop("encoder", ecpose_s_config["encoder"])
    decoder = kwargs.pop("decoder", ecpose_s_config["decoder"])
    postprocessor = kwargs.pop("postprocessor", ecpose_s_config["postprocessor"])
    return ECPose(
        name="ecpose_s",
        backbone=backbone,
        encoder=encoder,
        decoder=decoder,
        postprocessor=postprocessor,
        weights=ECPose_S_Weights(weights),
        *args, **kwargs,
    )


@MODELS.register(name="ecpose_m", metaclass=ECPose)
def ecpose_m(weights: WeightsLike = "default", *args, **kwargs):
    """Create an EdgeCrafter model.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
    """
    _ = kwargs.pop("name", "ecpose_m")
    backbone = kwargs.pop("backbone", ecpose_m_config["backbone"])
    encoder = kwargs.pop("encoder", ecpose_m_config["encoder"])
    decoder = kwargs.pop("decoder", ecpose_m_config["decoder"])
    postprocessor = kwargs.pop("postprocessor", ecpose_m_config["postprocessor"])
    return ECPose(
        name="ecpose_m",
        backbone=backbone,
        encoder=encoder,
        decoder=decoder,
        postprocessor=postprocessor,
        weights=ECPose_M_Weights(weights),
        *args, **kwargs,
    )


@MODELS.register(name="ecpose_l", metaclass=ECPose)
def ecpose_l(weights: WeightsLike = "default", *args, **kwargs):
    """Create an EdgeCrafter model.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
    """
    _ = kwargs.pop("name", "ecpose_l")
    backbone = kwargs.pop("backbone", ecpose_l_config["backbone"])
    encoder = kwargs.pop("encoder", ecpose_l_config["encoder"])
    decoder = kwargs.pop("decoder", ecpose_l_config["decoder"])
    postprocessor = kwargs.pop("postprocessor", ecpose_l_config["postprocessor"])
    return ECPose(
        name="ecpose_l",
        backbone=backbone,
        encoder=encoder,
        decoder=decoder,
        postprocessor=postprocessor,
        weights=ECPose_L_Weights(weights),
        *args, **kwargs,
    )


@MODELS.register(name="ecpose_x", metaclass=ECPose)
def ecpose_x(weights: WeightsLike = "default", *args, **kwargs):
    """Create an EdgeCrafter model.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
    """
    _ = kwargs.pop("name", "ecpose_x")
    backbone = kwargs.pop("backbone", ecpose_x_config["backbone"])
    encoder = kwargs.pop("encoder", ecpose_x_config["encoder"])
    decoder = kwargs.pop("decoder", ecpose_x_config["decoder"])
    postprocessor = kwargs.pop("postprocessor", ecpose_x_config["postprocessor"])
    return ECPose(
        name="ecpose_x",
        backbone=backbone,
        encoder=encoder,
        decoder=decoder,
        postprocessor=postprocessor,
        weights=ECPose_X_Weights(weights),
        *args, **kwargs,
    )

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion

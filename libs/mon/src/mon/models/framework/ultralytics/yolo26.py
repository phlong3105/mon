#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Ultralytics YOLO26 Models.

This module provides the Ultralytics YOLO26 definition and pre-trained weights.

References:
    - Code: https://docs.ultralytics.com/models/yolo26/
"""

from __future__ import annotations

__all__ = [
    "YOLO26l_CLS_Weights",
    "YOLO26l_OBB_Weights",
    "YOLO26l_POSE_Weights",
    "YOLO26l_SEG_Weights",
    "YOLO26l_Weights",
    "YOLO26m_CLS_Weights",
    "YOLO26m_OBB_Weights",
    "YOLO26m_POSE_Weights",
    "YOLO26m_SEG_Weights",
    "YOLO26m_Weights",
    "YOLO26n_CLS_Weights",
    "YOLO26n_OBB_Weights",
    "YOLO26n_POSE_Weights",
    "YOLO26n_SEG_Weights",
    "YOLO26n_Weights",
    "YOLO26s_CLS_Weights",
    "YOLO26s_OBB_Weights",
    "YOLO26s_POSE_Weights",
    "YOLO26s_SEG_Weights",
    "YOLO26s_Weights",
    "YOLO26x_CLS_Weights",
    "YOLO26x_OBB_Weights",
    "YOLO26x_POSE_Weights",
    "YOLO26x_SEG_Weights",
    "YOLO26x_Weights",
    "yolo26l",
    "yolo26l_cls",
    "yolo26l_obb",
    "yolo26l_pose",
    "yolo26l_seg",
    "yolo26m",
    "yolo26m_cls",
    "yolo26m_obb",
    "yolo26m_pose",
    "yolo26m_seg",
    "yolo26n",
    "yolo26n_cls",
    "yolo26n_obb",
    "yolo26n_pose",
    "yolo26n_seg",
    "yolo26s",
    "yolo26s_cls",
    "yolo26s_obb",
    "yolo26s_pose",
    "yolo26s_seg",
    "yolo26x",
    "yolo26x_cls",
    "yolo26x_obb",
    "yolo26x_pose",
    "yolo26x_seg",
]

from mon.core import (
    K,
    MODELS,
    Path,
    WEIGHTS,
    Weights,
    WeightsEnum,
    WeightsLike,
)
from .model import YOLO

try:
    import ultralytics
except ImportError:
    raise ImportError("Please install 'ultralytics' first.")

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region DETECTION
# ==============================================================================

# --- Pre-trained Weights ---

@WEIGHTS.register(name="yolo26n")
class YOLO26n_Weights(WeightsEnum):

    COCO = Weights(
        path=K.ZOO_ROOT / "framework/ultralytics/yolo26/yolo26n/coco/yolo26n_coco.pt",
        url=Path("https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt"),
        num_classes=80,
        transforms=None,
        meta={}
    )
    DEFAULT = COCO


@WEIGHTS.register(name="yolo26s")
class YOLO26s_Weights(WeightsEnum):

    COCO = Weights(
        path=K.ZOO_ROOT / "framework/ultralytics/yolo26/yolo26s/coco/yolo26s_coco.pt",
        url=Path("https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s.pt"),
        num_classes=80,
        transforms=None,
        meta={}
    )
    DEFAULT = COCO


@WEIGHTS.register(name="yolo26m")
class YOLO26m_Weights(WeightsEnum):

    COCO = Weights(
        path=K.ZOO_ROOT / "framework/ultralytics/yolo26/yolo26m/coco/yolo26m_coco.pt",
        url=Path("https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt"),
        num_classes=80,
        transforms=None,
        meta={}
    )
    DEFAULT = COCO


@WEIGHTS.register(name="yolo26l")
class YOLO26l_Weights(WeightsEnum):

    COCO = Weights(
        path=K.ZOO_ROOT / "framework/ultralytics/yolo26/yolo26l/coco/yolo26l_coco.pt",
        url=Path("https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26l.pt"),
        num_classes=80,
        transforms=None,
        meta={}
    )
    DEFAULT = COCO


@WEIGHTS.register(name="yolo26x")
class YOLO26x_Weights(WeightsEnum):

    COCO = Weights(
        path=K.ZOO_ROOT / "framework/ultralytics/yolo26/yolo26x/coco/yolo26x_coco.pt",
        url=Path("https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x.pt"),
        num_classes=80,
        transforms=None,
        meta={}
    )
    DEFAULT = COCO


# --- Model Variants ---

@MODELS.register(arch="yolo26", name="yolo26n", metaclass=YOLO)
def yolo26n(weights: WeightsLike = "default", *args, **kwargs):
    """Create an Ultralytics YOLO model.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
    """
    _ = kwargs.pop("name", "yolo26n")
    return YOLO(name="yolo26n", weights=YOLO26n_Weights(weights), *args, **kwargs)


@MODELS.register(arch="yolo26", name="yolo26s", metaclass=YOLO)
def yolo26s(weights: WeightsLike = "default", *args, **kwargs):
    """Create an Ultralytics YOLO model.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
    """
    _ = kwargs.pop("name", "yolo26s")
    return YOLO(name="yolo26s", weights=YOLO26s_Weights(weights), *args, **kwargs)


@MODELS.register(arch="yolo26", name="yolo26m", metaclass=YOLO)
def yolo26m(weights: WeightsLike = "default", *args, **kwargs):
    """Create an Ultralytics YOLO model.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
    """
    _ = kwargs.pop("name", "yolo26m")
    return YOLO(name="yolo26m", weights=YOLO26m_Weights(weights), *args, **kwargs)


@MODELS.register(arch="yolo26", name="yolo26l", metaclass=YOLO)
def yolo26l(weights: WeightsLike = "default", *args, **kwargs):
    """Create an Ultralytics YOLO model.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
    """
    _ = kwargs.pop("name", "yolo26l")
    return YOLO(name="yolo26l", weights=YOLO26l_Weights(weights), *args, **kwargs)


@MODELS.register(arch="yolo26", name="yolo26x", metaclass=YOLO)
def yolo26x(weights: WeightsLike = "default", *args, **kwargs):
    """Create an Ultralytics YOLO model.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
    """
    _ = kwargs.pop("name", "yolo26x")
    return YOLO(name="yolo26x", weights=YOLO26x_Weights(weights), *args, **kwargs)

# endregion


# ==============================================================================
# region SEGMENTATION
# ==============================================================================

# --- Pre-trained Weights ---

@WEIGHTS.register(name="yolo26n_seg")
class YOLO26n_SEG_Weights(WeightsEnum):

    COCO = Weights(
        path=K.ZOO_ROOT / "framework/ultralytics/yolo26/yolo26n_seg/coco/yolo26n_seg_coco.pt",
        url=Path("https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-seg.pt"),
        num_classes=80,
        transforms=None,
        meta={}
    )
    DEFAULT = COCO


@WEIGHTS.register(name="yolo26s_seg")
class YOLO26s_SEG_Weights(WeightsEnum):

    COCO = Weights(
        path=K.ZOO_ROOT / "framework/ultralytics/yolo26/yolo26s_seg/coco/yolo26s_seg_coco.pt",
        url=Path("https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s-seg.pt"),
        num_classes=80,
        transforms=None,
        meta={}
    )
    DEFAULT = COCO


@WEIGHTS.register(name="yolo26m_seg")
class YOLO26m_SEG_Weights(WeightsEnum):

    COCO = Weights(
        path=K.ZOO_ROOT / "framework/ultralytics/yolo26/yolo26m_seg/coco/yolo26m_seg_coco.pt",
        url=Path("https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m-seg.pt"),
        num_classes=80,
        transforms=None,
        meta={}
    )
    DEFAULT = COCO


@WEIGHTS.register(name="yolo26l_seg")
class YOLO26l_SEG_Weights(WeightsEnum):

    COCO = Weights(
        path=K.ZOO_ROOT / "framework/ultralytics/yolo26/yolo26l_seg/coco/yolo26l_seg_coco.pt",
        url=Path("https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26l-seg.pt"),
        num_classes=80,
        transforms=None,
        meta={}
    )
    DEFAULT = COCO


@WEIGHTS.register(name="yolo26x_seg")
class YOLO26x_SEG_Weights(WeightsEnum):

    COCO = Weights(
        path=K.ZOO_ROOT / "framework/ultralytics/yolo26/yolo26x_seg/coco/yolo26x_seg_coco.pt",
        url=Path("https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x-seg.pt"),
        num_classes=80,
        transforms=None,
        meta={}
    )
    DEFAULT = COCO


# --- Model Variants ---

@MODELS.register(arch="yolo26", name="yolo26n_seg", metaclass=YOLO)
def yolo26n_seg(weights: WeightsLike = "default", *args, **kwargs):
    """Create an Ultralytics YOLO model.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
    """
    _ = kwargs.pop("name", "yolo26n_seg")
    return YOLO(name="yolo26n", weights=YOLO26n_SEG_Weights(weights), *args, **kwargs)


@MODELS.register(arch="yolo26", name="yolo26s_seg", metaclass=YOLO)
def yolo26s_seg(weights: WeightsLike = "default", *args, **kwargs):
    """Create an Ultralytics YOLO model.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
    """
    _ = kwargs.pop("name", "yolo26s_seg")
    return YOLO(name="yolo26s_seg", weights=YOLO26s_SEG_Weights(weights), *args, **kwargs)


@MODELS.register(arch="yolo26", name="yolo26m_seg", metaclass=YOLO)
def yolo26m_seg(weights: WeightsLike = "default", *args, **kwargs):
    """Create an Ultralytics YOLO model.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
    """
    _ = kwargs.pop("name", "yolo26m_seg")
    return YOLO(name="yolo26m_seg", weights=YOLO26m_SEG_Weights(weights), *args, **kwargs)


@MODELS.register(arch="yolo26", name="yolo26l_seg", metaclass=YOLO)
def yolo26l_seg(weights: WeightsLike = "default", *args, **kwargs):
    """Create an Ultralytics YOLO model.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
    """
    _ = kwargs.pop("name", "yolo26l_seg")
    return YOLO(name="yolo26l_seg", weights=YOLO26l_SEG_Weights(weights), *args, **kwargs)


@MODELS.register(arch="yolo26", name="yolo26x_seg", metaclass=YOLO)
def yolo26x_seg(weights: WeightsLike = "default", *args, **kwargs):
    """Create an Ultralytics YOLO model.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
    """
    _ = kwargs.pop("name", "yolo26x_seg")
    return YOLO(name="yolo26x_seg", weights=YOLO26x_SEG_Weights(weights), *args, **kwargs)

# endregion


# ==============================================================================
# region CLASSIFICATION
# ==============================================================================

# --- Pre-trained Weights ---

@WEIGHTS.register(name="yolo26n_cls")
class YOLO26n_CLS_Weights(WeightsEnum):

    COCO = Weights(
        path=K.ZOO_ROOT / "framework/ultralytics/yolo26/yolo26n_cls/imagenet/yolo26n_cls_imagenet.pt",
        url=Path("https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-cls.pt"),
        num_classes=1000,
        transforms=None,
        meta={}
    )
    DEFAULT = COCO


@WEIGHTS.register(name="yolo26s_cls")
class YOLO26s_CLS_Weights(WeightsEnum):

    COCO = Weights(
        path=K.ZOO_ROOT / "framework/ultralytics/yolo26/yolo26s_cls/imagenet/yolo26s_cls_imagenet.pt",
        url=Path("https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s-cls.pt"),
        num_classes=1000,
        transforms=None,
        meta={}
    )
    DEFAULT = COCO


@WEIGHTS.register(name="yolo26m_cls")
class YOLO26m_CLS_Weights(WeightsEnum):

    COCO = Weights(
        path=K.ZOO_ROOT / "framework/ultralytics/yolo26/yolo26m_cls/imagenet/yolo26m_cls_imagenet.pt",
        url=Path("https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m-cls.pt"),
        num_classes=1000,
        transforms=None,
        meta={}
    )
    DEFAULT = COCO


@WEIGHTS.register(name="yolo26l_cls")
class YOLO26l_CLS_Weights(WeightsEnum):

    COCO = Weights(
        path=K.ZOO_ROOT / "framework/ultralytics/yolo26/yolo26l_cls/imagenet/yolo26l_cls_imagenet.pt",
        url=Path("https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26l-cls.pt"),
        num_classes=1000,
        transforms=None,
        meta={}
    )
    DEFAULT = COCO


@WEIGHTS.register(name="yolo26x_cls")
class YOLO26x_CLS_Weights(WeightsEnum):

    COCO = Weights(
        path=K.ZOO_ROOT / "framework/ultralytics/yolo26/yolo26x_cls/imagenet/yolo26x_cls_imagenet.pt",
        url=Path("https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x-cls.pt"),
        num_classes=1000,
        transforms=None,
        meta={}
    )
    DEFAULT = COCO


# --- Model Variants ---

@MODELS.register(arch="yolo26", name="yolo26n_cls", metaclass=YOLO)
def yolo26n_cls(weights: WeightsLike = "default", *args, **kwargs):
    """Create an Ultralytics YOLO model.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
    """
    _ = kwargs.pop("name", "yolo26n_cls")
    return YOLO(name="yolo26n", weights=YOLO26n_CLS_Weights(weights), *args, **kwargs)


@MODELS.register(arch="yolo26", name="yolo26s_cls", metaclass=YOLO)
def yolo26s_cls(weights: WeightsLike = "default", *args, **kwargs):
    """Create an Ultralytics YOLO model.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
    """
    _ = kwargs.pop("name", "yolo26s_cls")
    return YOLO(name="yolo26s_cls", weights=YOLO26s_CLS_Weights(weights), *args, **kwargs)


@MODELS.register(arch="yolo26", name="yolo26m_cls", metaclass=YOLO)
def yolo26m_cls(weights: WeightsLike = "default", *args, **kwargs):
    """Create an Ultralytics YOLO model.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
    """
    _ = kwargs.pop("name", "yolo26m_cls")
    return YOLO(name="yolo26m_cls", weights=YOLO26m_CLS_Weights(weights), *args, **kwargs)


@MODELS.register(arch="yolo26", name="yolo26l_cls", metaclass=YOLO)
def yolo26l_cls(weights: WeightsLike = "default", *args, **kwargs):
    """Create an Ultralytics YOLO model.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
    """
    _ = kwargs.pop("name", "yolo26l_cls")
    return YOLO(name="yolo26l_cls", weights=YOLO26l_CLS_Weights(weights), *args, **kwargs)


@MODELS.register(arch="yolo26", name="yolo26x_cls", metaclass=YOLO)
def yolo26x_cls(weights: WeightsLike = "default", *args, **kwargs):
    """Create an Ultralytics YOLO model.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
    """
    _ = kwargs.pop("name", "yolo26x_cls")
    return YOLO(name="yolo26x_cls", weights=YOLO26x_CLS_Weights(weights), *args, **kwargs)

# endregion


# ==============================================================================
# region HUMAN POSE ESTIMATION
# ==============================================================================

# --- Pre-trained Weights ---

@WEIGHTS.register(name="yolo26n_pose")
class YOLO26n_POSE_Weights(WeightsEnum):

    COCO = Weights(
        path=K.ZOO_ROOT / "framework/ultralytics/yolo26/yolo26n_pose/coco/yolo26n_pose_coco.pt",
        url=Path("https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-pose.pt"),
        num_classes=1,
        transforms=None,
        meta={}
    )
    DEFAULT = COCO


@WEIGHTS.register(name="yolo26s_pose")
class YOLO26s_POSE_Weights(WeightsEnum):

    COCO = Weights(
        path=K.ZOO_ROOT / "framework/ultralytics/yolo26/yolo26s_pose/coco/yolo26s_pose_coco.pt",
        url=Path("https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s-pose.pt"),
        num_classes=1,
        transforms=None,
        meta={}
    )
    DEFAULT = COCO


@WEIGHTS.register(name="yolo26m_pose")
class YOLO26m_POSE_Weights(WeightsEnum):

    COCO = Weights(
        path=K.ZOO_ROOT / "framework/ultralytics/yolo26/yolo26m_pose/coco/yolo26m_pose_coco.pt",
        url=Path("https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m-pose.pt"),
        num_classes=1,
        transforms=None,
        meta={}
    )
    DEFAULT = COCO


@WEIGHTS.register(name="yolo26l_pose")
class YOLO26l_POSE_Weights(WeightsEnum):

    COCO = Weights(
        path=K.ZOO_ROOT / "framework/ultralytics/yolo26/yolo26l_pose/coco/yolo26l_pose_coco.pt",
        url=Path("https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26l-pose.pt"),
        num_classes=1,
        transforms=None,
        meta={}
    )
    DEFAULT = COCO


@WEIGHTS.register(name="yolo26x_pose")
class YOLO26x_POSE_Weights(WeightsEnum):

    COCO = Weights(
        path=K.ZOO_ROOT / "framework/ultralytics/yolo26/yolo26x_pose/coco/yolo26x_pose_coco.pt",
        url=Path("https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x-pose.pt"),
        num_classes=1,
        transforms=None,
        meta={}
    )
    DEFAULT = COCO


# --- Model Variants ---

@MODELS.register(arch="yolo26", name="yolo26n_pose", metaclass=YOLO)
def yolo26n_pose(weights: WeightsLike = "default", *args, **kwargs):
    """Create an Ultralytics YOLO model.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
    """
    _ = kwargs.pop("name", "yolo26n_pose")
    return YOLO(name="yolo26n", weights=YOLO26n_POSE_Weights(weights), *args, **kwargs)


@MODELS.register(arch="yolo26", name="yolo26s_pose", metaclass=YOLO)
def yolo26s_pose(weights: WeightsLike = "default", *args, **kwargs):
    """Create an Ultralytics YOLO model.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
    """
    _ = kwargs.pop("name", "yolo26s_pose")
    return YOLO(name="yolo26s_pose", weights=YOLO26s_POSE_Weights(weights), *args, **kwargs)


@MODELS.register(arch="yolo26", name="yolo26m_pose", metaclass=YOLO)
def yolo26m_pose(weights: WeightsLike = "default", *args, **kwargs):
    """Create an Ultralytics YOLO model.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
    """
    _ = kwargs.pop("name", "yolo26m_pose")
    return YOLO(name="yolo26m_pose", weights=YOLO26m_POSE_Weights(weights), *args, **kwargs)


@MODELS.register(arch="yolo26", name="yolo26l_pose", metaclass=YOLO)
def yolo26l_pose(weights: WeightsLike = "default", *args, **kwargs):
    """Create an Ultralytics YOLO model.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
    """
    _ = kwargs.pop("name", "yolo26l_pose")
    return YOLO(name="yolo26l_pose", weights=YOLO26l_POSE_Weights(weights), *args, **kwargs)


@MODELS.register(arch="yolo26", name="yolo26x_pose", metaclass=YOLO)
def yolo26x_pose(weights: WeightsLike = "default", *args, **kwargs):
    """Create an Ultralytics YOLO model.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
    """
    _ = kwargs.pop("name", "yolo26x_pose")
    return YOLO(name="yolo26x_pose", weights=YOLO26x_POSE_Weights(weights), *args, **kwargs)

# endregion


# ==============================================================================
# region OBB
# ==============================================================================

# --- Pre-trained Weights ---

@WEIGHTS.register(name="yolo26n_obb")
class YOLO26n_OBB_Weights(WeightsEnum):

    COCO = Weights(
        path=K.ZOO_ROOT / "framework/ultralytics/yolo26/yolo26n_obb/dota_v1/yolo26n_obb_dota_v1.pt",
        url=Path("https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-obb.pt"),
        num_classes=15,
        transforms=None,
        meta={}
    )
    DEFAULT = COCO


@WEIGHTS.register(name="yolo26s_obb")
class YOLO26s_OBB_Weights(WeightsEnum):

    COCO = Weights(
        path=K.ZOO_ROOT / "framework/ultralytics/yolo26/yolo26s_obb/dota_v1/yolo26s_obb_dota_v1.pt",
        url=Path("https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s-obb.pt"),
        num_classes=15,
        transforms=None,
        meta={}
    )
    DEFAULT = COCO


@WEIGHTS.register(name="yolo26m_obb")
class YOLO26m_OBB_Weights(WeightsEnum):

    COCO = Weights(
        path=K.ZOO_ROOT / "framework/ultralytics/yolo26/yolo26m_obb/dota_v1/yolo26m_obb_dota_v1.pt",
        url=Path("https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m-obb.pt"),
        num_classes=15,
        transforms=None,
        meta={}
    )
    DEFAULT = COCO


@WEIGHTS.register(name="yolo26l_obb")
class YOLO26l_OBB_Weights(WeightsEnum):

    COCO = Weights(
        path=K.ZOO_ROOT / "framework/ultralytics/yolo26/yolo26l_obb/dota_v1/yolo26l_obb_dota_v1.pt",
        url=Path("https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26l-obb.pt"),
        num_classes=15,
        transforms=None,
        meta={}
    )
    DEFAULT = COCO


@WEIGHTS.register(name="yolo26x_obb")
class YOLO26x_OBB_Weights(WeightsEnum):

    COCO = Weights(
        path=K.ZOO_ROOT / "framework/ultralytics/yolo26/yolo26x_obb/dota_v1/yolo26x_obb_dota_v1.pt",
        url=Path("https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x-obb.pt"),
        num_classes=15,
        transforms=None,
        meta={}
    )
    DEFAULT = COCO


# --- Model Variants ---

@MODELS.register(arch="yolo26", name="yolo26n_obb", metaclass=YOLO)
def yolo26n_obb(weights: WeightsLike = "default", *args, **kwargs):
    """Create an Ultralytics YOLO model.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
    """
    _ = kwargs.pop("name", "yolo26n_obb")
    return YOLO(name="yolo26n", weights=YOLO26n_OBB_Weights(weights), *args, **kwargs)


@MODELS.register(arch="yolo26", name="yolo26s_obb", metaclass=YOLO)
def yolo26s_obb(weights: WeightsLike = "default", *args, **kwargs):
    """Create an Ultralytics YOLO model.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
    """
    _ = kwargs.pop("name", "yolo26s_obb")
    return YOLO(name="yolo26s_obb", weights=YOLO26s_OBB_Weights(weights), *args, **kwargs)


@MODELS.register(arch="yolo26", name="yolo26m_obb", metaclass=YOLO)
def yolo26m_obb(weights: WeightsLike = "default", *args, **kwargs):
    """Create an Ultralytics YOLO model.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
    """
    _ = kwargs.pop("name", "yolo26m_obb")
    return YOLO(name="yolo26m_obb", weights=YOLO26m_OBB_Weights(weights), *args, **kwargs)


@MODELS.register(arch="yolo26", name="yolo26l_obb", metaclass=YOLO)
def yolo26l_obb(weights: WeightsLike = "default", *args, **kwargs):
    """Create an Ultralytics YOLO model.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
    """
    _ = kwargs.pop("name", "yolo26l_obb")
    return YOLO(name="yolo26l_obb", weights=YOLO26l_OBB_Weights(weights), *args, **kwargs)


@MODELS.register(arch="yolo26", name="yolo26x_obb", metaclass=YOLO)
def yolo26x_obb(weights: WeightsLike = "default", *args, **kwargs):
    """Create an Ultralytics YOLO model.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
    """
    _ = kwargs.pop("name", "yolo26x_obb")
    return YOLO(name="yolo26x_obb", weights=YOLO26x_OBB_Weights(weights), *args, **kwargs)

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion

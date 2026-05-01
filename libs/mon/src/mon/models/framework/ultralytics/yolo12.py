#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Ultralytics YOLO12 Models.

This module provides the Ultralytics YOLO12 definition and pre-trained weights.

References:
    - Code: https://docs.ultralytics.com/models/yolo12/
"""

from __future__ import annotations

__all__ = [
    "YOLO12l_CLS_Weights",
    "YOLO12l_OBB_Weights",
    "YOLO12l_POSE_Weights",
    "YOLO12l_SEG_Weights",
    "YOLO12l_Weights",
    "YOLO12m_CLS_Weights",
    "YOLO12m_OBB_Weights",
    "YOLO12m_POSE_Weights",
    "YOLO12m_SEG_Weights",
    "YOLO12m_Weights",
    "YOLO12n_CLS_Weights",
    "YOLO12n_OBB_Weights",
    "YOLO12n_POSE_Weights",
    "YOLO12n_SEG_Weights",
    "YOLO12n_Weights",
    "YOLO12s_CLS_Weights",
    "YOLO12s_OBB_Weights",
    "YOLO12s_POSE_Weights",
    "YOLO12s_SEG_Weights",
    "YOLO12s_Weights",
    "YOLO12x_CLS_Weights",
    "YOLO12x_OBB_Weights",
    "YOLO12x_POSE_Weights",
    "YOLO12x_SEG_Weights",
    "YOLO12x_Weights",
    "yolo12l",
    "yolo12l_cls",
    "yolo12l_obb",
    "yolo12l_pose",
    "yolo12l_seg",
    "yolo12m",
    "yolo12m_cls",
    "yolo12m_obb",
    "yolo12m_pose",
    "yolo12m_seg",
    "yolo12n",
    "yolo12n_cls",
    "yolo12n_obb",
    "yolo12n_pose",
    "yolo12n_seg",
    "yolo12s",
    "yolo12s_cls",
    "yolo12s_obb",
    "yolo12s_pose",
    "yolo12s_seg",
    "yolo12x",
    "yolo12x_cls",
    "yolo12x_obb",
    "yolo12x_pose",
    "yolo12x_seg",
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

@WEIGHTS.register(name="yolo12n")
class YOLO12n_Weights(WeightsEnum):

    COCO = Weights(
        path=K.ZOO_ROOT / "framework/ultralytics/yolo12/yolo12n/coco/yolo12n_coco.pt",
        url=Path("https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo12n.pt"),
        num_classes=80,
        transforms=None,
        meta={}
    )
    DEFAULT = COCO


@WEIGHTS.register(name="yolo12s")
class YOLO12s_Weights(WeightsEnum):

    COCO = Weights(
        path=K.ZOO_ROOT / "framework/ultralytics/yolo12/yolo12s/coco/yolo12s_coco.pt",
        url=Path("https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo12s.pt"),
        num_classes=80,
        transforms=None,
        meta={}
    )
    DEFAULT = COCO


@WEIGHTS.register(name="yolo12m")
class YOLO12m_Weights(WeightsEnum):

    COCO = Weights(
        path=K.ZOO_ROOT / "framework/ultralytics/yolo12/yolo12m/coco/yolo12m_coco.pt",
        url=Path("https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo12m.pt"),
        num_classes=80,
        transforms=None,
        meta={}
    )
    DEFAULT = COCO


@WEIGHTS.register(name="yolo12l")
class YOLO12l_Weights(WeightsEnum):

    COCO = Weights(
        path=K.ZOO_ROOT / "framework/ultralytics/yolo12/yolo12l/coco/yolo12l_coco.pt",
        url=Path("https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo12l.pt"),
        num_classes=80,
        transforms=None,
        meta={}
    )
    DEFAULT = COCO


@WEIGHTS.register(name="yolo12x")
class YOLO12x_Weights(WeightsEnum):

    COCO = Weights(
        path=K.ZOO_ROOT / "framework/ultralytics/yolo12/yolo12x/coco/yolo12x_coco.pt",
        url=Path("https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo12x.pt"),
        num_classes=80,
        transforms=None,
        meta={}
    )
    DEFAULT = COCO


# --- Model Variants ---

@MODELS.register(arch="yolo12", name="yolo12n", metaclass=YOLO)
def yolo12n(weights: WeightsLike = "default", *args, **kwargs):
    """Create an Ultralytics YOLO model.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
    """
    _ = kwargs.pop("name", "yolo12n")
    return YOLO(name="yolo12n", weights=YOLO12n_Weights(weights), *args, **kwargs)


@MODELS.register(arch="yolo12", name="yolo12s", metaclass=YOLO)
def yolo12s(weights: WeightsLike = "default", *args, **kwargs):
    """Create an Ultralytics YOLO model.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
    """
    _ = kwargs.pop("name", "yolo12s")
    return YOLO(name="yolo12s", weights=YOLO12s_Weights(weights), *args, **kwargs)


@MODELS.register(arch="yolo12", name="yolo12m", metaclass=YOLO)
def yolo12m(weights: WeightsLike = "default", *args, **kwargs):
    """Create an Ultralytics YOLO model.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
    """
    _ = kwargs.pop("name", "yolo12m")
    return YOLO(name="yolo12m", weights=YOLO12m_Weights(weights), *args, **kwargs)


@MODELS.register(arch="yolo12", name="yolo12l", metaclass=YOLO)
def yolo12l(weights: WeightsLike = "default", *args, **kwargs):
    """Create an Ultralytics YOLO model.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
    """
    _ = kwargs.pop("name", "yolo12l")
    return YOLO(name="yolo12l", weights=YOLO12l_Weights(weights), *args, **kwargs)


@MODELS.register(arch="yolo12", name="yolo12x", metaclass=YOLO)
def yolo12x(weights: WeightsLike = "default", *args, **kwargs):
    """Create an Ultralytics YOLO model.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
    """
    _ = kwargs.pop("name", "yolo12x")
    return YOLO(name="yolo12x", weights=YOLO12x_Weights(weights), *args, **kwargs)

# endregion


# ==============================================================================
# region SEGMENTATION
# ==============================================================================

# --- Pre-trained Weights ---

@WEIGHTS.register(name="yolo12n_seg")
class YOLO12n_SEG_Weights(WeightsEnum):

    COCO = Weights(
        path=K.ZOO_ROOT / "framework/ultralytics/yolo12/yolo12n_seg/coco/yolo12n_seg_coco.pt",
        url=Path("https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo12n-seg.pt"),
        num_classes=80,
        transforms=None,
        meta={}
    )
    DEFAULT = COCO


@WEIGHTS.register(name="yolo12s_seg")
class YOLO12s_SEG_Weights(WeightsEnum):

    COCO = Weights(
        path=K.ZOO_ROOT / "framework/ultralytics/yolo12/yolo12s_seg/coco/yolo12s_seg_coco.pt",
        url=Path("https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo12s-seg.pt"),
        num_classes=80,
        transforms=None,
        meta={}
    )
    DEFAULT = COCO


@WEIGHTS.register(name="yolo12m_seg")
class YOLO12m_SEG_Weights(WeightsEnum):

    COCO = Weights(
        path=K.ZOO_ROOT / "framework/ultralytics/yolo12/yolo12m_seg/coco/yolo12m_seg_coco.pt",
        url=Path("https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo12m-seg.pt"),
        num_classes=80,
        transforms=None,
        meta={}
    )
    DEFAULT = COCO


@WEIGHTS.register(name="yolo12l_seg")
class YOLO12l_SEG_Weights(WeightsEnum):

    COCO = Weights(
        path=K.ZOO_ROOT / "framework/ultralytics/yolo12/yolo12l_seg/coco/yolo12l_seg_coco.pt",
        url=Path("https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo12l-seg.pt"),
        num_classes=80,
        transforms=None,
        meta={}
    )
    DEFAULT = COCO


@WEIGHTS.register(name="yolo12x_seg")
class YOLO12x_SEG_Weights(WeightsEnum):

    COCO = Weights(
        path=K.ZOO_ROOT / "framework/ultralytics/yolo12/yolo12x_seg/coco/yolo12x_seg_coco.pt",
        url=Path("https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo12x-seg.pt"),
        num_classes=80,
        transforms=None,
        meta={}
    )
    DEFAULT = COCO


# --- Model Variants ---

@MODELS.register(arch="yolo12", name="yolo12n_seg", metaclass=YOLO)
def yolo12n_seg(weights: WeightsLike = "default", *args, **kwargs):
    """Create an Ultralytics YOLO model.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
    """
    _ = kwargs.pop("name", "yolo12n_seg")
    return YOLO(name="yolo12n", weights=YOLO12n_SEG_Weights(weights), *args, **kwargs)


@MODELS.register(arch="yolo12", name="yolo12s_seg", metaclass=YOLO)
def yolo12s_seg(weights: WeightsLike = "default", *args, **kwargs):
    """Create an Ultralytics YOLO model.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
    """
    _ = kwargs.pop("name", "yolo12s_seg")
    return YOLO(name="yolo12s_seg", weights=YOLO12s_SEG_Weights(weights), *args, **kwargs)


@MODELS.register(arch="yolo12", name="yolo12m_seg", metaclass=YOLO)
def yolo12m_seg(weights: WeightsLike = "default", *args, **kwargs):
    """Create an Ultralytics YOLO model.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
    """
    _ = kwargs.pop("name", "yolo12m_seg")
    return YOLO(name="yolo12m_seg", weights=YOLO12m_SEG_Weights(weights), *args, **kwargs)


@MODELS.register(arch="yolo12", name="yolo12l_seg", metaclass=YOLO)
def yolo12l_seg(weights: WeightsLike = "default", *args, **kwargs):
    """Create an Ultralytics YOLO model.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
    """
    _ = kwargs.pop("name", "yolo12l_seg")
    return YOLO(name="yolo12l_seg", weights=YOLO12l_SEG_Weights(weights), *args, **kwargs)


@MODELS.register(arch="yolo12", name="yolo12x_seg", metaclass=YOLO)
def yolo12x_seg(weights: WeightsLike = "default", *args, **kwargs):
    """Create an Ultralytics YOLO model.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
    """
    _ = kwargs.pop("name", "yolo12x_seg")
    return YOLO(name="yolo12x_seg", weights=YOLO12x_SEG_Weights(weights), *args, **kwargs)

# endregion


# ==============================================================================
# region CLASSIFICATION
# ==============================================================================

# --- Pre-trained Weights ---

@WEIGHTS.register(name="yolo12n_cls")
class YOLO12n_CLS_Weights(WeightsEnum):

    COCO = Weights(
        path=K.ZOO_ROOT / "framework/ultralytics/yolo12/yolo12n_cls/imagenet/yolo12n_cls_imagenet.pt",
        url=Path("https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo12n-cls.pt"),
        num_classes=1000,
        transforms=None,
        meta={}
    )
    DEFAULT = COCO


@WEIGHTS.register(name="yolo12s_cls")
class YOLO12s_CLS_Weights(WeightsEnum):

    COCO = Weights(
        path=K.ZOO_ROOT / "framework/ultralytics/yolo12/yolo12s_cls/imagenet/yolo12s_cls_imagenet.pt",
        url=Path("https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo12s-cls.pt"),
        num_classes=1000,
        transforms=None,
        meta={}
    )
    DEFAULT = COCO


@WEIGHTS.register(name="yolo12m_cls")
class YOLO12m_CLS_Weights(WeightsEnum):

    COCO = Weights(
        path=K.ZOO_ROOT / "framework/ultralytics/yolo12/yolo12m_cls/imagenet/yolo12m_cls_imagenet.pt",
        url=Path("https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo12m-cls.pt"),
        num_classes=1000,
        transforms=None,
        meta={}
    )
    DEFAULT = COCO


@WEIGHTS.register(name="yolo12l_cls")
class YOLO12l_CLS_Weights(WeightsEnum):

    COCO = Weights(
        path=K.ZOO_ROOT / "framework/ultralytics/yolo12/yolo12l_cls/imagenet/yolo12l_cls_imagenet.pt",
        url=Path("https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo12l-cls.pt"),
        num_classes=1000,
        transforms=None,
        meta={}
    )
    DEFAULT = COCO


@WEIGHTS.register(name="yolo12x_cls")
class YOLO12x_CLS_Weights(WeightsEnum):

    COCO = Weights(
        path=K.ZOO_ROOT / "framework/ultralytics/yolo12/yolo12x_cls/imagenet/yolo12x_cls_imagenet.pt",
        url=Path("https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo12x-cls.pt"),
        num_classes=1000,
        transforms=None,
        meta={}
    )
    DEFAULT = COCO


# --- Model Variants ---

@MODELS.register(arch="yolo12", name="yolo12n_cls", metaclass=YOLO)
def yolo12n_cls(weights: WeightsLike = "default", *args, **kwargs):
    """Create an Ultralytics YOLO model.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
    """
    _ = kwargs.pop("name", "yolo12n_cls")
    return YOLO(name="yolo12n", weights=YOLO12n_CLS_Weights(weights), *args, **kwargs)


@MODELS.register(arch="yolo12", name="yolo12s_cls", metaclass=YOLO)
def yolo12s_cls(weights: WeightsLike = "default", *args, **kwargs):
    """Create an Ultralytics YOLO model.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
    """
    _ = kwargs.pop("name", "yolo12s_cls")
    return YOLO(name="yolo12s_cls", weights=YOLO12s_CLS_Weights(weights), *args, **kwargs)


@MODELS.register(arch="yolo12", name="yolo12m_cls", metaclass=YOLO)
def yolo12m_cls(weights: WeightsLike = "default", *args, **kwargs):
    """Create an Ultralytics YOLO model.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
    """
    _ = kwargs.pop("name", "yolo12m_cls")
    return YOLO(name="yolo12m_cls", weights=YOLO12m_CLS_Weights(weights), *args, **kwargs)


@MODELS.register(arch="yolo12", name="yolo12l_cls", metaclass=YOLO)
def yolo12l_cls(weights: WeightsLike = "default", *args, **kwargs):
    """Create an Ultralytics YOLO model.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
    """
    _ = kwargs.pop("name", "yolo12l_cls")
    return YOLO(name="yolo12l_cls", weights=YOLO12l_CLS_Weights(weights), *args, **kwargs)


@MODELS.register(arch="yolo12", name="yolo12x_cls", metaclass=YOLO)
def yolo12x_cls(weights: WeightsLike = "default", *args, **kwargs):
    """Create an Ultralytics YOLO model.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
    """
    _ = kwargs.pop("name", "yolo12x_cls")
    return YOLO(name="yolo12x_cls", weights=YOLO12x_CLS_Weights(weights), *args, **kwargs)

# endregion


# ==============================================================================
# region HUMAN POSE ESTIMATION
# ==============================================================================

# --- Pre-trained Weights ---

@WEIGHTS.register(name="yolo12n_pose")
class YOLO12n_POSE_Weights(WeightsEnum):

    COCO = Weights(
        path=K.ZOO_ROOT / "framework/ultralytics/yolo12/yolo12n_pose/coco/yolo12n_pose_coco.pt",
        url=Path("https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo12n-pose.pt"),
        num_classes=1,
        transforms=None,
        meta={}
    )
    DEFAULT = COCO


@WEIGHTS.register(name="yolo12s_pose")
class YOLO12s_POSE_Weights(WeightsEnum):

    COCO = Weights(
        path=K.ZOO_ROOT / "framework/ultralytics/yolo12/yolo12s_pose/coco/yolo12s_pose_coco.pt",
        url=Path("https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo12s-pose.pt"),
        num_classes=1,
        transforms=None,
        meta={}
    )
    DEFAULT = COCO


@WEIGHTS.register(name="yolo12m_pose")
class YOLO12m_POSE_Weights(WeightsEnum):

    COCO = Weights(
        path=K.ZOO_ROOT / "framework/ultralytics/yolo12/yolo12m_pose/coco/yolo12m_pose_coco.pt",
        url=Path("https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo12m-pose.pt"),
        num_classes=1,
        transforms=None,
        meta={}
    )
    DEFAULT = COCO


@WEIGHTS.register(name="yolo12l_pose")
class YOLO12l_POSE_Weights(WeightsEnum):

    COCO = Weights(
        path=K.ZOO_ROOT / "framework/ultralytics/yolo12/yolo12l_pose/coco/yolo12l_pose_coco.pt",
        url=Path("https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo12l-pose.pt"),
        num_classes=1,
        transforms=None,
        meta={}
    )
    DEFAULT = COCO


@WEIGHTS.register(name="yolo12x_pose")
class YOLO12x_POSE_Weights(WeightsEnum):

    COCO = Weights(
        path=K.ZOO_ROOT / "framework/ultralytics/yolo12/yolo12x_pose/coco/yolo12x_pose_coco.pt",
        url=Path("https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo12x-pose.pt"),
        num_classes=1,
        transforms=None,
        meta={}
    )
    DEFAULT = COCO


# --- Model Variants ---

@MODELS.register(arch="yolo12", name="yolo12n_pose", metaclass=YOLO)
def yolo12n_pose(weights: WeightsLike = "default", *args, **kwargs):
    """Create an Ultralytics YOLO model.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
    """
    _ = kwargs.pop("name", "yolo12n_pose")
    return YOLO(name="yolo12n", weights=YOLO12n_POSE_Weights(weights), *args, **kwargs)


@MODELS.register(arch="yolo12", name="yolo12s_pose", metaclass=YOLO)
def yolo12s_pose(weights: WeightsLike = "default", *args, **kwargs):
    """Create an Ultralytics YOLO model.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
    """
    _ = kwargs.pop("name", "yolo12s_pose")
    return YOLO(name="yolo12s_pose", weights=YOLO12s_POSE_Weights(weights), *args, **kwargs)


@MODELS.register(arch="yolo12", name="yolo12m_pose", metaclass=YOLO)
def yolo12m_pose(weights: WeightsLike = "default", *args, **kwargs):
    """Create an Ultralytics YOLO model.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
    """
    _ = kwargs.pop("name", "yolo12m_pose")
    return YOLO(name="yolo12m_pose", weights=YOLO12m_POSE_Weights(weights), *args, **kwargs)


@MODELS.register(arch="yolo12", name="yolo12l_pose", metaclass=YOLO)
def yolo12l_pose(weights: WeightsLike = "default", *args, **kwargs):
    """Create an Ultralytics YOLO model.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
    """
    _ = kwargs.pop("name", "yolo12l_pose")
    return YOLO(name="yolo12l_pose", weights=YOLO12l_POSE_Weights(weights), *args, **kwargs)


@MODELS.register(arch="yolo12", name="yolo12x_pose", metaclass=YOLO)
def yolo12x_pose(weights: WeightsLike = "default", *args, **kwargs):
    """Create an Ultralytics YOLO model.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
    """
    _ = kwargs.pop("name", "yolo12x_pose")
    return YOLO(name="yolo12x_pose", weights=YOLO12x_POSE_Weights(weights), *args, **kwargs)

# endregion


# ==============================================================================
# region OBB
# ==============================================================================

# --- Pre-trained Weights ---

@WEIGHTS.register(name="yolo12n_obb")
class YOLO12n_OBB_Weights(WeightsEnum):

    COCO = Weights(
        path=K.ZOO_ROOT / "framework/ultralytics/yolo12/yolo12n_obb/dota_v1/yolo12n_obb_dota_v1.pt",
        url=Path("https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo12n-obb.pt"),
        num_classes=15,
        transforms=None,
        meta={}
    )
    DEFAULT = COCO


@WEIGHTS.register(name="yolo12s_obb")
class YOLO12s_OBB_Weights(WeightsEnum):

    COCO = Weights(
        path=K.ZOO_ROOT / "framework/ultralytics/yolo12/yolo12s_obb/dota_v1/yolo12s_obb_dota_v1.pt",
        url=Path("https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo12s-obb.pt"),
        num_classes=15,
        transforms=None,
        meta={}
    )
    DEFAULT = COCO


@WEIGHTS.register(name="yolo12m_obb")
class YOLO12m_OBB_Weights(WeightsEnum):

    COCO = Weights(
        path=K.ZOO_ROOT / "framework/ultralytics/yolo12/yolo12m_obb/dota_v1/yolo12m_obb_dota_v1.pt",
        url=Path("https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo12m-obb.pt"),
        num_classes=15,
        transforms=None,
        meta={}
    )
    DEFAULT = COCO


@WEIGHTS.register(name="yolo12l_obb")
class YOLO12l_OBB_Weights(WeightsEnum):

    COCO = Weights(
        path=K.ZOO_ROOT / "framework/ultralytics/yolo12/yolo12l_obb/dota_v1/yolo12l_obb_dota_v1.pt",
        url=Path("https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo12l-obb.pt"),
        num_classes=15,
        transforms=None,
        meta={}
    )
    DEFAULT = COCO


@WEIGHTS.register(name="yolo12x_obb")
class YOLO12x_OBB_Weights(WeightsEnum):

    COCO = Weights(
        path=K.ZOO_ROOT / "framework/ultralytics/yolo12/yolo12x_obb/dota_v1/yolo12x_obb_dota_v1.pt",
        url=Path("https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo12x-obb.pt"),
        num_classes=15,
        transforms=None,
        meta={}
    )
    DEFAULT = COCO


# --- Model Variants ---

@MODELS.register(arch="yolo12", name="yolo12n_obb", metaclass=YOLO)
def yolo12n_obb(weights: WeightsLike = "default", *args, **kwargs):
    """Create an Ultralytics YOLO model.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
    """
    _ = kwargs.pop("name", "yolo12n_obb")
    return YOLO(name="yolo12n", weights=YOLO12n_OBB_Weights(weights), *args, **kwargs)


@MODELS.register(arch="yolo12", name="yolo12s_obb", metaclass=YOLO)
def yolo12s_obb(weights: WeightsLike = "default", *args, **kwargs):
    """Create an Ultralytics YOLO model.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
    """
    _ = kwargs.pop("name", "yolo12s_obb")
    return YOLO(name="yolo12s_obb", weights=YOLO12s_OBB_Weights(weights), *args, **kwargs)


@MODELS.register(arch="yolo12", name="yolo12m_obb", metaclass=YOLO)
def yolo12m_obb(weights: WeightsLike = "default", *args, **kwargs):
    """Create an Ultralytics YOLO model.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
    """
    _ = kwargs.pop("name", "yolo12m_obb")
    return YOLO(name="yolo12m_obb", weights=YOLO12m_OBB_Weights(weights), *args, **kwargs)


@MODELS.register(arch="yolo12", name="yolo12l_obb", metaclass=YOLO)
def yolo12l_obb(weights: WeightsLike = "default", *args, **kwargs):
    """Create an Ultralytics YOLO model.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
    """
    _ = kwargs.pop("name", "yolo12l_obb")
    return YOLO(name="yolo12l_obb", weights=YOLO12l_OBB_Weights(weights), *args, **kwargs)


@MODELS.register(arch="yolo12", name="yolo12x_obb", metaclass=YOLO)
def yolo12x_obb(weights: WeightsLike = "default", *args, **kwargs):
    """Create an Ultralytics YOLO model.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
    """
    _ = kwargs.pop("name", "yolo12x_obb")
    return YOLO(name="yolo12x_obb", weights=YOLO12x_OBB_Weights(weights), *args, **kwargs)

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion

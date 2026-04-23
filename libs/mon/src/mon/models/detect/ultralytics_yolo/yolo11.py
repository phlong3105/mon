#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Ultralytics YOLO11 Models.

This module provides the Ultralytics YOLO11 definition and pre-trained weights.

References:
    - Code: https://docs.ultralytics.com/models/yolo11/
"""

from __future__ import annotations

__all__ = [
    "YOLO11l_CLS_Weights",
    "YOLO11l_OBB_Weights",
    "YOLO11l_POSE_Weights",
    "YOLO11l_SEG_Weights",
    "YOLO11l_Weights",
    "YOLO11m_CLS_Weights",
    "YOLO11m_OBB_Weights",
    "YOLO11m_POSE_Weights",
    "YOLO11m_SEG_Weights",
    "YOLO11m_Weights",
    "YOLO11n_CLS_Weights",
    "YOLO11n_OBB_Weights",
    "YOLO11n_POSE_Weights",
    "YOLO11n_SEG_Weights",
    "YOLO11n_Weights",
    "YOLO11s_CLS_Weights",
    "YOLO11s_OBB_Weights",
    "YOLO11s_POSE_Weights",
    "YOLO11s_SEG_Weights",
    "YOLO11s_Weights",
    "YOLO11x_CLS_Weights",
    "YOLO11x_OBB_Weights",
    "YOLO11x_POSE_Weights",
    "YOLO11x_SEG_Weights",
    "YOLO11x_Weights",
    "yolo11l",
    "yolo11l_cls",
    "yolo11l_obb",
    "yolo11l_pose",
    "yolo11l_seg",
    "yolo11m",
    "yolo11m_cls",
    "yolo11m_obb",
    "yolo11m_pose",
    "yolo11m_seg",
    "yolo11n",
    "yolo11n_cls",
    "yolo11n_obb",
    "yolo11n_pose",
    "yolo11n_seg",
    "yolo11s",
    "yolo11s_cls",
    "yolo11s_obb",
    "yolo11s_pose",
    "yolo11s_seg",
    "yolo11x",
    "yolo11x_cls",
    "yolo11x_obb",
    "yolo11x_pose",
    "yolo11x_seg",
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

@WEIGHTS.register(name="yolo11n")
class YOLO11n_Weights(WeightsEnum):

    COCO = Weights(
        path=K.ZOO_ROOT / "ultralytics/yolo11/yolo11n/coco/yolo11n_coco.pt",
        url=Path("https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.pt"),
        num_classes=80,
        transforms=None,
        meta={}
    )
    DEFAULT = COCO


@WEIGHTS.register(name="yolo11s")
class YOLO11s_Weights(WeightsEnum):

    COCO = Weights(
        path=K.ZOO_ROOT / "ultralytics/yolo11/yolo11s/coco/yolo11s_coco.pt",
        url=Path("https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11s.pt"),
        num_classes=80,
        transforms=None,
        meta={}
    )
    DEFAULT = COCO


@WEIGHTS.register(name="yolo11m")
class YOLO11m_Weights(WeightsEnum):

    COCO = Weights(
        path=K.ZOO_ROOT / "ultralytics/yolo11/yolo11m/coco/yolo11m_coco.pt",
        url=Path("https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11m.pt"),
        num_classes=80,
        transforms=None,
        meta={}
    )
    DEFAULT = COCO


@WEIGHTS.register(name="yolo11l")
class YOLO11l_Weights(WeightsEnum):

    COCO = Weights(
        path=K.ZOO_ROOT / "ultralytics/yolo11/yolo11l/coco/yolo11l_coco.pt",
        url=Path("https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11l.pt"),
        num_classes=80,
        transforms=None,
        meta={}
    )
    DEFAULT = COCO


@WEIGHTS.register(name="yolo11x")
class YOLO11x_Weights(WeightsEnum):

    COCO = Weights(
        path=K.ZOO_ROOT / "ultralytics/yolo11/yolo11x/coco/yolo11x_coco.pt",
        url=Path("https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11x.pt"),
        num_classes=80,
        transforms=None,
        meta={}
    )
    DEFAULT = COCO


# --- Model Variants ---

@MODELS.register(arch="yolo11", name="yolo11n", metaclass=YOLO)
def yolo11n(weights: WeightsLike = "default", *args, **kwargs):
    """Create an Ultralytics YOLO model.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
    """
    _ = kwargs.pop("name", "yolo11n")
    return YOLO(name="yolo11n", weights=YOLO11n_Weights(weights), *args, **kwargs)


@MODELS.register(arch="yolo11", name="yolo11s", metaclass=YOLO)
def yolo11s(weights: WeightsLike = "default", *args, **kwargs):
    """Create an Ultralytics YOLO model.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
    """
    _ = kwargs.pop("name", "yolo11s")
    return YOLO(name="yolo11s", weights=YOLO11s_Weights(weights), *args, **kwargs)


@MODELS.register(arch="yolo11", name="yolo11m", metaclass=YOLO)
def yolo11m(weights: WeightsLike = "default", *args, **kwargs):
    """Create an Ultralytics YOLO model.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
    """
    _ = kwargs.pop("name", "yolo11m")
    return YOLO(name="yolo11m", weights=YOLO11m_Weights(weights), *args, **kwargs)


@MODELS.register(arch="yolo11", name="yolo11l", metaclass=YOLO)
def yolo11l(weights: WeightsLike = "default", *args, **kwargs):
    """Create an Ultralytics YOLO model.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
    """
    _ = kwargs.pop("name", "yolo11l")
    return YOLO(name="yolo11l", weights=YOLO11l_Weights(weights), *args, **kwargs)


@MODELS.register(arch="yolo11", name="yolo11x", metaclass=YOLO)
def yolo11x(weights: WeightsLike = "default", *args, **kwargs):
    """Create an Ultralytics YOLO model.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
    """
    _ = kwargs.pop("name", "yolo11x")
    return YOLO(name="yolo11x", weights=YOLO11x_Weights(weights), *args, **kwargs)

# endregion


# ==============================================================================
# region SEGMENTATION
# ==============================================================================

# --- Pre-trained Weights ---

@WEIGHTS.register(name="yolo11n_seg")
class YOLO11n_SEG_Weights(WeightsEnum):

    COCO = Weights(
        path=K.ZOO_ROOT / "ultralytics/yolo11/yolo11n_seg/coco/yolo11n_seg_coco.pt",
        url=Path("https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n-seg.pt"),
        num_classes=80,
        transforms=None,
        meta={}
    )
    DEFAULT = COCO


@WEIGHTS.register(name="yolo11s_seg")
class YOLO11s_SEG_Weights(WeightsEnum):

    COCO = Weights(
        path=K.ZOO_ROOT / "ultralytics/yolo11/yolo11s_seg/coco/yolo11s_seg_coco.pt",
        url=Path("https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11s-seg.pt"),
        num_classes=80,
        transforms=None,
        meta={}
    )
    DEFAULT = COCO


@WEIGHTS.register(name="yolo11m_seg")
class YOLO11m_SEG_Weights(WeightsEnum):

    COCO = Weights(
        path=K.ZOO_ROOT / "ultralytics/yolo11/yolo11m_seg/coco/yolo11m_seg_coco.pt",
        url=Path("https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11m-seg.pt"),
        num_classes=80,
        transforms=None,
        meta={}
    )
    DEFAULT = COCO


@WEIGHTS.register(name="yolo11l_seg")
class YOLO11l_SEG_Weights(WeightsEnum):

    COCO = Weights(
        path=K.ZOO_ROOT / "ultralytics/yolo11/yolo11l_seg/coco/yolo11l_seg_coco.pt",
        url=Path("https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11l-seg.pt"),
        num_classes=80,
        transforms=None,
        meta={}
    )
    DEFAULT = COCO


@WEIGHTS.register(name="yolo11x_seg")
class YOLO11x_SEG_Weights(WeightsEnum):

    COCO = Weights(
        path=K.ZOO_ROOT / "ultralytics/yolo11/yolo11x_seg/coco/yolo11x_seg_coco.pt",
        url=Path("https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11x-seg.pt"),
        num_classes=80,
        transforms=None,
        meta={}
    )
    DEFAULT = COCO


# --- Model Variants ---

@MODELS.register(arch="yolo11", name="yolo11n_seg", metaclass=YOLO)
def yolo11n_seg(weights: WeightsLike = "default", *args, **kwargs):
    """Create an Ultralytics YOLO model.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
    """
    _ = kwargs.pop("name", "yolo11n_seg")
    return YOLO(name="yolo11n", weights=YOLO11n_SEG_Weights(weights), *args, **kwargs)


@MODELS.register(arch="yolo11", name="yolo11s_seg", metaclass=YOLO)
def yolo11s_seg(weights: WeightsLike = "default", *args, **kwargs):
    """Create an Ultralytics YOLO model.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
    """
    _ = kwargs.pop("name", "yolo11s_seg")
    return YOLO(name="yolo11s_seg", weights=YOLO11s_SEG_Weights(weights), *args, **kwargs)


@MODELS.register(arch="yolo11", name="yolo11m_seg", metaclass=YOLO)
def yolo11m_seg(weights: WeightsLike = "default", *args, **kwargs):
    """Create an Ultralytics YOLO model.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
    """
    _ = kwargs.pop("name", "yolo11m_seg")
    return YOLO(name="yolo11m_seg", weights=YOLO11m_SEG_Weights(weights), *args, **kwargs)


@MODELS.register(arch="yolo11", name="yolo11l_seg", metaclass=YOLO)
def yolo11l_seg(weights: WeightsLike = "default", *args, **kwargs):
    """Create an Ultralytics YOLO model.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
    """
    _ = kwargs.pop("name", "yolo11l_seg")
    return YOLO(name="yolo11l_seg", weights=YOLO11l_SEG_Weights(weights), *args, **kwargs)


@MODELS.register(arch="yolo11", name="yolo11x_seg", metaclass=YOLO)
def yolo11x_seg(weights: WeightsLike = "default", *args, **kwargs):
    """Create an Ultralytics YOLO model.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
    """
    _ = kwargs.pop("name", "yolo11x_seg")
    return YOLO(name="yolo11x_seg", weights=YOLO11x_SEG_Weights(weights), *args, **kwargs)

# endregion


# ==============================================================================
# region CLASSIFICATION
# ==============================================================================

# --- Pre-trained Weights ---

@WEIGHTS.register(name="yolo11n_cls")
class YOLO11n_CLS_Weights(WeightsEnum):

    COCO = Weights(
        path=K.ZOO_ROOT / "ultralytics/yolo11/yolo11n_cls/imagenet/yolo11n_cls_imagenet.pt",
        url=Path("https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n-cls.pt"),
        num_classes=1000,
        transforms=None,
        meta={}
    )
    DEFAULT = COCO


@WEIGHTS.register(name="yolo11s_cls")
class YOLO11s_CLS_Weights(WeightsEnum):

    COCO = Weights(
        path=K.ZOO_ROOT / "ultralytics/yolo11/yolo11s_cls/imagenet/yolo11s_cls_imagenet.pt",
        url=Path("https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11s-cls.pt"),
        num_classes=1000,
        transforms=None,
        meta={}
    )
    DEFAULT = COCO


@WEIGHTS.register(name="yolo11m_cls")
class YOLO11m_CLS_Weights(WeightsEnum):

    COCO = Weights(
        path=K.ZOO_ROOT / "ultralytics/yolo11/yolo11m_cls/imagenet/yolo11m_cls_imagenet.pt",
        url=Path("https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11m-cls.pt"),
        num_classes=1000,
        transforms=None,
        meta={}
    )
    DEFAULT = COCO


@WEIGHTS.register(name="yolo11l_cls")
class YOLO11l_CLS_Weights(WeightsEnum):

    COCO = Weights(
        path=K.ZOO_ROOT / "ultralytics/yolo11/yolo11l_cls/imagenet/yolo11l_cls_imagenet.pt",
        url=Path("https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11l-cls.pt"),
        num_classes=1000,
        transforms=None,
        meta={}
    )
    DEFAULT = COCO


@WEIGHTS.register(name="yolo11x_cls")
class YOLO11x_CLS_Weights(WeightsEnum):

    COCO = Weights(
        path=K.ZOO_ROOT / "ultralytics/yolo11/yolo11x_cls/imagenet/yolo11x_cls_imagenet.pt",
        url=Path("https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11x-cls.pt"),
        num_classes=1000,
        transforms=None,
        meta={}
    )
    DEFAULT = COCO


# --- Model Variants ---

@MODELS.register(arch="yolo11", name="yolo11n_cls", metaclass=YOLO)
def yolo11n_cls(weights: WeightsLike = "default", *args, **kwargs):
    """Create an Ultralytics YOLO model.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
    """
    _ = kwargs.pop("name", "yolo11n_cls")
    return YOLO(name="yolo11n", weights=YOLO11n_CLS_Weights(weights), *args, **kwargs)


@MODELS.register(arch="yolo11", name="yolo11s_cls", metaclass=YOLO)
def yolo11s_cls(weights: WeightsLike = "default", *args, **kwargs):
    """Create an Ultralytics YOLO model.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
    """
    _ = kwargs.pop("name", "yolo11s_cls")
    return YOLO(name="yolo11s_cls", weights=YOLO11s_CLS_Weights(weights), *args, **kwargs)


@MODELS.register(arch="yolo11", name="yolo11m_cls", metaclass=YOLO)
def yolo11m_cls(weights: WeightsLike = "default", *args, **kwargs):
    """Create an Ultralytics YOLO model.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
    """
    _ = kwargs.pop("name", "yolo11m_cls")
    return YOLO(name="yolo11m_cls", weights=YOLO11m_CLS_Weights(weights), *args, **kwargs)


@MODELS.register(arch="yolo11", name="yolo11l_cls", metaclass=YOLO)
def yolo11l_cls(weights: WeightsLike = "default", *args, **kwargs):
    """Create an Ultralytics YOLO model.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
    """
    _ = kwargs.pop("name", "yolo11l_cls")
    return YOLO(name="yolo11l_cls", weights=YOLO11l_CLS_Weights(weights), *args, **kwargs)


@MODELS.register(arch="yolo11", name="yolo11x_cls", metaclass=YOLO)
def yolo11x_cls(weights: WeightsLike = "default", *args, **kwargs):
    """Create an Ultralytics YOLO model.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
    """
    _ = kwargs.pop("name", "yolo11x_cls")
    return YOLO(name="yolo11x_cls", weights=YOLO11x_CLS_Weights(weights), *args, **kwargs)

# endregion


# ==============================================================================
# region HUMAN POSE ESTIMATION
# ==============================================================================

# --- Pre-trained Weights ---

@WEIGHTS.register(name="yolo11n_pose")
class YOLO11n_POSE_Weights(WeightsEnum):

    COCO = Weights(
        path=K.ZOO_ROOT / "ultralytics/yolo11/yolo11n_pose/coco/yolo11n_pose_coco.pt",
        url=Path("https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n-pose.pt"),
        num_classes=1,
        transforms=None,
        meta={}
    )
    DEFAULT = COCO


@WEIGHTS.register(name="yolo11s_pose")
class YOLO11s_POSE_Weights(WeightsEnum):

    COCO = Weights(
        path=K.ZOO_ROOT / "ultralytics/yolo11/yolo11s_pose/coco/yolo11s_pose_coco.pt",
        url=Path("https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11s-pose.pt"),
        num_classes=1,
        transforms=None,
        meta={}
    )
    DEFAULT = COCO


@WEIGHTS.register(name="yolo11m_pose")
class YOLO11m_POSE_Weights(WeightsEnum):

    COCO = Weights(
        path=K.ZOO_ROOT / "ultralytics/yolo11/yolo11m_pose/coco/yolo11m_pose_coco.pt",
        url=Path("https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11m-pose.pt"),
        num_classes=1,
        transforms=None,
        meta={}
    )
    DEFAULT = COCO


@WEIGHTS.register(name="yolo11l_pose")
class YOLO11l_POSE_Weights(WeightsEnum):

    COCO = Weights(
        path=K.ZOO_ROOT / "ultralytics/yolo11/yolo11l_pose/coco/yolo11l_pose_coco.pt",
        url=Path("https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11l-pose.pt"),
        num_classes=1,
        transforms=None,
        meta={}
    )
    DEFAULT = COCO


@WEIGHTS.register(name="yolo11x_pose")
class YOLO11x_POSE_Weights(WeightsEnum):

    COCO = Weights(
        path=K.ZOO_ROOT / "ultralytics/yolo11/yolo11x_pose/coco/yolo11x_pose_coco.pt",
        url=Path("https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11x-pose.pt"),
        num_classes=1,
        transforms=None,
        meta={}
    )
    DEFAULT = COCO


# --- Model Variants ---

@MODELS.register(arch="yolo11", name="yolo11n_pose", metaclass=YOLO)
def yolo11n_pose(weights: WeightsLike = "default", *args, **kwargs):
    """Create an Ultralytics YOLO model.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
    """
    _ = kwargs.pop("name", "yolo11n_pose")
    return YOLO(name="yolo11n", weights=YOLO11n_POSE_Weights(weights), *args, **kwargs)


@MODELS.register(arch="yolo11", name="yolo11s_pose", metaclass=YOLO)
def yolo11s_pose(weights: WeightsLike = "default", *args, **kwargs):
    """Create an Ultralytics YOLO model.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
    """
    _ = kwargs.pop("name", "yolo11s_pose")
    return YOLO(name="yolo11s_pose", weights=YOLO11s_POSE_Weights(weights), *args, **kwargs)


@MODELS.register(arch="yolo11", name="yolo11m_pose", metaclass=YOLO)
def yolo11m_pose(weights: WeightsLike = "default", *args, **kwargs):
    """Create an Ultralytics YOLO model.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
    """
    _ = kwargs.pop("name", "yolo11m_pose")
    return YOLO(name="yolo11m_pose", weights=YOLO11m_POSE_Weights(weights), *args, **kwargs)


@MODELS.register(arch="yolo11", name="yolo11l_pose", metaclass=YOLO)
def yolo11l_pose(weights: WeightsLike = "default", *args, **kwargs):
    """Create an Ultralytics YOLO model.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
    """
    _ = kwargs.pop("name", "yolo11l_pose")
    return YOLO(name="yolo11l_pose", weights=YOLO11l_POSE_Weights(weights), *args, **kwargs)


@MODELS.register(arch="yolo11", name="yolo11x_pose", metaclass=YOLO)
def yolo11x_pose(weights: WeightsLike = "default", *args, **kwargs):
    """Create an Ultralytics YOLO model.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
    """
    _ = kwargs.pop("name", "yolo11x_pose")
    return YOLO(name="yolo11x_pose", weights=YOLO11x_POSE_Weights(weights), *args, **kwargs)

# endregion


# ==============================================================================
# region OBB
# ==============================================================================

# --- Pre-trained Weights ---

@WEIGHTS.register(name="yolo11n_obb")
class YOLO11n_OBB_Weights(WeightsEnum):

    COCO = Weights(
        path=K.ZOO_ROOT / "ultralytics/yolo11/yolo11n_obb/dota_v1/yolo11n_obb_dota_v1.pt",
        url=Path("https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n-obb.pt"),
        num_classes=15,
        transforms=None,
        meta={}
    )
    DEFAULT = COCO


@WEIGHTS.register(name="yolo11s_obb")
class YOLO11s_OBB_Weights(WeightsEnum):

    COCO = Weights(
        path=K.ZOO_ROOT / "ultralytics/yolo11/yolo11s_obb/dota_v1/yolo11s_obb_dota_v1.pt",
        url=Path("https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11s-obb.pt"),
        num_classes=15,
        transforms=None,
        meta={}
    )
    DEFAULT = COCO


@WEIGHTS.register(name="yolo11m_obb")
class YOLO11m_OBB_Weights(WeightsEnum):

    COCO = Weights(
        path=K.ZOO_ROOT / "ultralytics/yolo11/yolo11m_obb/dota_v1/yolo11m_obb_dota_v1.pt",
        url=Path("https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11m-obb.pt"),
        num_classes=15,
        transforms=None,
        meta={}
    )
    DEFAULT = COCO


@WEIGHTS.register(name="yolo11l_obb")
class YOLO11l_OBB_Weights(WeightsEnum):

    COCO = Weights(
        path=K.ZOO_ROOT / "ultralytics/yolo11/yolo11l_obb/dota_v1/yolo11l_obb_dota_v1.pt",
        url=Path("https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11l-obb.pt"),
        num_classes=15,
        transforms=None,
        meta={}
    )
    DEFAULT = COCO


@WEIGHTS.register(name="yolo11x_obb")
class YOLO11x_OBB_Weights(WeightsEnum):

    COCO = Weights(
        path=K.ZOO_ROOT / "ultralytics/yolo11/yolo11x_obb/dota_v1/yolo11x_obb_dota_v1.pt",
        url=Path("https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11x-obb.pt"),
        num_classes=15,
        transforms=None,
        meta={}
    )
    DEFAULT = COCO


# --- Model Variants ---

@MODELS.register(arch="yolo11", name="yolo11n_obb", metaclass=YOLO)
def yolo11n_obb(weights: WeightsLike = "default", *args, **kwargs):
    """Create an Ultralytics YOLO model.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
    """
    _ = kwargs.pop("name", "yolo11n_obb")
    return YOLO(name="yolo11n", weights=YOLO11n_OBB_Weights(weights), *args, **kwargs)


@MODELS.register(arch="yolo11", name="yolo11s_obb", metaclass=YOLO)
def yolo11s_obb(weights: WeightsLike = "default", *args, **kwargs):
    """Create an Ultralytics YOLO model.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
    """
    _ = kwargs.pop("name", "yolo11s_obb")
    return YOLO(name="yolo11s_obb", weights=YOLO11s_OBB_Weights(weights), *args, **kwargs)


@MODELS.register(arch="yolo11", name="yolo11m_obb", metaclass=YOLO)
def yolo11m_obb(weights: WeightsLike = "default", *args, **kwargs):
    """Create an Ultralytics YOLO model.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
    """
    _ = kwargs.pop("name", "yolo11m_obb")
    return YOLO(name="yolo11m_obb", weights=YOLO11m_OBB_Weights(weights), *args, **kwargs)


@MODELS.register(arch="yolo11", name="yolo11l_obb", metaclass=YOLO)
def yolo11l_obb(weights: WeightsLike = "default", *args, **kwargs):
    """Create an Ultralytics YOLO model.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
    """
    _ = kwargs.pop("name", "yolo11l_obb")
    return YOLO(name="yolo11l_obb", weights=YOLO11l_OBB_Weights(weights), *args, **kwargs)


@MODELS.register(arch="yolo11", name="yolo11x_obb", metaclass=YOLO)
def yolo11x_obb(weights: WeightsLike = "default", *args, **kwargs):
    """Create an Ultralytics YOLO model.

    Args:
        weights (Weights, optional): Pre-trained weights to load.
            Defaults to "default".
    """
    _ = kwargs.pop("name", "yolo11x_obb")
    return YOLO(name="yolo11x_obb", weights=YOLO11x_OBB_Weights(weights), *args, **kwargs)

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion

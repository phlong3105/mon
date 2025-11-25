#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This package provides various loss functions for training machine learning models.

References:
    - Definition: https://www.ibm.com/think/topics/loss-function#1580786328
"""

__all__ = [
    "BCELoss",
    "BCEWithLogitsLoss",
    "BaseLoss",
    "CTCLoss",
    "CharbonnierLoss",
    "ColorConstancyLoss",
    "CosineEmbeddingLoss",
    "CosineSimilarityLoss",
    "CrossEntropyLoss",
    "DepthAwareIlluminationLoss",
    "EdgeLoss",
    "ExposureControlLoss",
    "ExposureValueControlLoss",
    "ExtendedL1Loss",
    "GaussianNLLLoss",
    "HingeEmbeddingLoss",
    "HuberLoss",
    "KLDivLoss",
    "L1Loss",
    "MSELoss",
    "MarginRankingLoss",
    "MultiLabelMarginLoss",
    "MultiLabelSoftMarginLoss",
    "MultiMarginLoss",
    "NLLLoss",
    "NLLLoss2d",
    "PSNRLoss",
    "PoissonNLLLoss",
    "SmoothL1Loss",
    "SoftMarginLoss",
    "SpatialConsistencyLoss",
    "StructureTextureDecompositionLoss",
    "TotalVariationLoss",
    "TripletMarginLoss",
    "TripletMarginWithDistanceLoss",
]

from .base import BaseLoss
from .core import (
    BCELoss,
    BCEWithLogitsLoss,
    CharbonnierLoss,
    CosineEmbeddingLoss,
    CosineSimilarityLoss,
    CrossEntropyLoss,
    CTCLoss,
    ExtendedL1Loss,
    GaussianNLLLoss,
    HingeEmbeddingLoss,
    HuberLoss,
    KLDivLoss,
    L1Loss,
    MarginRankingLoss,
    MSELoss,
    MultiLabelMarginLoss,
    MultiLabelSoftMarginLoss,
    MultiMarginLoss,
    NLLLoss,
    NLLLoss2d,
    PoissonNLLLoss,
    SmoothL1Loss,
    SoftMarginLoss,
    TripletMarginLoss,
    TripletMarginWithDistanceLoss,
)
from .image import (
    ColorConstancyLoss,
    DepthAwareIlluminationLoss,
    EdgeLoss,
    ExposureControlLoss,
    ExposureValueControlLoss,
    PSNRLoss,
    SpatialConsistencyLoss,
    StructureTextureDecompositionLoss,
    TotalVariationLoss,
)

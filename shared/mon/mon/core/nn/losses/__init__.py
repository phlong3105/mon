#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements loss functions for neural network training."""

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

from .base import *
from .core import *
from .image import *

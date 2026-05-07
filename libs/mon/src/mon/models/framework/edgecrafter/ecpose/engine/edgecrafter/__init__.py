"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""


from .detrpose_criterion import DETRPoseCriterion
from .detrpose_matcher import DETRPoseHungarianMatcher
from .detrpose_postprocesses import DETRPosePostProcessor
from .detrpose_transformer import DETRTransformer
from .ecpose import ECPose
from .ecvit import ViTAdapter
from .hybrid_encoder import HybridEncoder

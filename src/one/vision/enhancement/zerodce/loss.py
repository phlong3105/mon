#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

from typing import Optional

from torch import nn
from torch import Tensor

from one.core import Tensors
from one.core import Weights
from one.nn import ColorConstancyLoss
from one.nn import ExposureControlLoss
from one.nn import IlluminationSmoothnessLoss
from one.nn import SpatialConsistencyLoss

__all__ = [
	"CombinedLoss",
]


# MARK: - Modules

class CombinedLoss(nn.Module):
    """loss = loss_spa + loss_exp + loss_col + loss_tv"""
    
    # MARK: Magic Functions

    def __init__(
        self,
        spa_weight    : float = 1.0,
	    exp_patch_size: int   = 16,
	    exp_mean_val  : float = 0.6,
        exp_weight    : float = 10.0,
        col_weight    : float = 5.0,
        tv_weight     : float = 200.0,
    ):
        super().__init__()
        self.loss_spa = SpatialConsistencyLoss(loss_weight=spa_weight)
        self.loss_exp = ExposureControlLoss(
	        patch_size  = exp_patch_size,
	        mean_val    = exp_mean_val,
	        loss_weight = exp_weight
        )
        self.loss_col = ColorConstancyLoss(loss_weight=col_weight)
        self.loss_tv  = IlluminationSmoothnessLoss(loss_weight=tv_weight)
    
    # MARK: Forward Pass
    
    def forward(
        self,
        input             : Tensor,
        pred              : Tensors,
        input_weight      : Optional[Weights] = None,
        elementwise_weight: Optional[Weights] = None,
        **_
    ) -> Tensor:
        r, enhanced_image = pred[0], pred[-1]
        loss_spa = self.loss_spa(input, enhanced_image, input_weight, elementwise_weight)
        loss_exp = self.loss_exp(enhanced_image, input_weight, elementwise_weight)
        loss_col = self.loss_col(enhanced_image, input_weight, elementwise_weight)
        loss_tv  = self.loss_tv(r, input_weight, elementwise_weight)
        loss     = loss_spa + loss_exp + loss_col + loss_tv
        return loss

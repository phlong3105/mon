#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Edge Loss.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.loss import _Loss

from one.core import LOSSES
from one.core import Tensors
from one.core import Weights
from one.nn.loss.charbonnier_loss import charbonnier_loss
from one.nn.loss.utils import weighted_loss
from one.nn.loss.utils import weighted_sum

__all__ = [
    "edge_loss",
    "elementwise_edge_loss",
    "EdgeLoss",
]


# MARK: - Functional

k 	   = Tensor([[0.05, 0.25, 0.4, 0.25, 0.05]])
kernel = torch.matmul(k.t(), k).unsqueeze(0).repeat(3, 1, 1, 1)


def _conv_gauss(image: Tensor) -> Tensor:
    global kernel
    if kernel.device != image.device:
        kernel = kernel.to(image.device)
    n_channels, _, kw, kh = kernel.shape
    image = F.pad(image, (kw // 2, kh // 2, kw // 2, kh // 2), mode="replicate")
    return F.conv2d(image, kernel, groups=n_channels)


def _laplacian_kernel(image: Tensor) -> Tensor:
    filtered   = _conv_gauss(image)  		# filter
    down 	   = filtered[:, :, ::2, ::2]   # downsample
    new_filter = torch.zeros_like(filtered)
    new_filter[:, :, ::2, ::2] = down * 4   # upsample
    filtered   = _conv_gauss(new_filter)    # filter
    diff 	   = image - filtered
    return diff


@weighted_loss
def elementwise_edge_loss(
    input: Tensor, target: Tensor, eps: float = 1e-3
) -> Tensor:
    """Apply elementwise weight and reduce loss between a batch of input and
    a batch of target.
    
    Args:
    	input (Tensor):
    	    Either the prediction or the original input (unsupervised learning)
    	    of shape [B, C, H, W]
    	target (Tensor):
    	    Ground-truth of shape [B, C, H, W]. Sometimes can be `None`
    	    (unsupervised learning).
    	eps (float):
    	    Small value for numerically stability when dividing.
            Default: `1e-3`.
            
    Returns:
    	loss (Tensor):
    	    Single reduced loss value.
    """
    return charbonnier_loss(
        input  = _laplacian_kernel(input),
        target = _laplacian_kernel(target),
        eps    = eps
    )


def edge_loss(
    input             : Tensors,
    target            : Tensor,
    eps               : float             = 1e-3,
    input_weight      : Optional[Weights] = None,
    elementwise_weight: Optional[Weights] = None,
    reduction         : str               = "mean",
) -> Tensor:
    """Measures the loss value.

    Args:
        input (Tensors):
            Either the prediction or the original input (unsupervised learning).
            Can be a single/collection of batches of shape [B, C, H, W].
        target (Tensor):
            Ground-truth of shape [B, C, H, W]. Sometimes can be `None`
    	    (unsupervised learning).
        eps (float):
    	    Small value for numerically stability when dividing.
            Default: `1e-3`.
        input_weight (Weights, optional):
            If `input` is a single batch, then set to `None` (or `1.0`).
            If `input` is a collection of batches, apply weighted sum on the
            returned loss values. Default: `None`.
        elementwise_weight (Weights, optional):
            Elementwise weights for each `input` batch of shape [B, C, H, W].
            Default: `None`.
        reduction (str):
            Specifies the reduction to apply to the output.
            One of: [`none`, `mean`, `sum`].
            - `none`: No reduction will be applied.
            - `mean`: The sum of the output will be divided by the number of
                      elements in the output.
            - `sum`: The output will be summed.
            Default: `mean`.
    
    Returns:
    	loss (Tensor):
    	    Loss value.
    """
    if isinstance(input, Tensor):  # Single output
        input = [input]
    elif isinstance(input, dict):
        input = list(input.values())
    if not isinstance(input, (list, tuple)):
        raise ValueError(f"`input` must be a `list` or `tuple`. But got: {type(input)}.")

    losses = [
        elementwise_edge_loss(
            input     = i,
            target    = target,
            eps       = eps,
            weight    = elementwise_weight,
            reduction = reduction
        ) for i in input
    ]
    return weighted_sum(losses, input_weight)


# MARK: - Modules

@LOSSES.register(name="edge_loss")
class EdgeLoss(_Loss):
    """Edge loss.
    
    Attributes:
        name (str):
            Name of the loss. Default: `edge_loss`.
        eps (float):
            Small value for numerically stability when dividing.
            Default: `1e-3`.
        loss_weight (Weights, optional):
			Some loss function is the sum of other loss functions.
			This provides weight for each loss component. Default: `1.0`.
        reduction (str):
            reduction (str):
            Specifies the reduction to apply to the output.
            One of: [`none`, `mean`, `sum`].
            - `none`: No reduction will be applied.
            - `mean`: The sum of the output will be divided by the number of
                      elements in the output.
            - `sum`: The output will be summed.
            Default: `mean`.
    """
    
    reductions = ["none", "mean", "sum"]
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        eps        : float             = 1e-3,
        loss_weight: Optional[Weights] = 1.0,
        reduction  : str               = "mean"
    ):
        super().__init__(reduction=reduction)
        self.name        = "edge_loss"
        self.eps         = eps
        self.loss_weight = loss_weight
        
        if self.reduction not in self.reductions:
            raise ValueError(f"`reduction` must be one of: {self.reductions}. "
                             f"But got: {self.reduction}.")

    # MARK: Forward Pass
    
    def forward(
        self,
        input             : Tensors,
        target            : Tensor,
        input_weight      : Optional[Weights] = None,
        elementwise_weight: Optional[Weights] = None,
        **_
    ) -> Tensor:
        """Measures the loss value.

        Args:
            input (Tensors):
                Either the prediction or the original input (unsupervised
                learning). Can be a single/collection of batches of shape
                [B, C, H, W].
            target (Tensor):
                Ground-truth of shape [B, C, H, W]. Sometimes can be `None`
        	    (unsupervised learning).
            input_weight (Weights, optional):
                If `input` is a single batch, then set to `None` (or `1.0`).
                If `input` is a collection of batches, apply weighted sum on the
                returned loss values. Default: `None`.
            elementwise_weight (Weights, optional):
                Elementwise weights for each `input` batch of shape [B, C, H, W].
                Default: `None`.
    
        Returns:
        	loss (Tensor):
        	    Loss value.
        """
        return self.loss_weight * edge_loss(
            input              = input,
            target             = target,
            eps                = self.eps,
            input_weight       = input_weight,
            elementwise_weight = elementwise_weight,
            reduction          = self.reduction,
        )

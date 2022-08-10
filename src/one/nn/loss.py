#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Loss Functions
"""

from __future__ import annotations

import functools
from abc import ABCMeta
from abc import abstractmethod

import numpy as np
import torch
from torch import FloatTensor
from torch import nn
from torch import Tensor
from torch.nn import functional
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss

from one.constants import LOSSES
from one.core import assert_sequence
from one.core import Callable
from one.core import Ints
from one.core import Reduction
from one.core import Reduction_
from one.core import Tensors
from one.nn.metric import psnr
from one.nn.metric import ssim


# H1: - Helper Functions -------------------------------------------------------


def reduce_loss(
    loss     : Tensor,
    weight   : Tensor | None = None,
    reduction: Reduction_    = "mean",
) -> Tensor:
    """
    Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (Reduction_): Reduction value to use.
        weight (Tensor | None): Element-wise weights. Defaults to None.
        
    Returns:
        Reduced loss.
    """
    reduction = Reduction.from_value(reduction)
    if reduction == Reduction.NONE:
        return loss
    if reduction == Reduction.MEAN:
        return loss.mean()
    if reduction == Reduction.SUM:
        return loss.sum()
    if reduction == Reduction.WEIGHTED_SUM:
        if weight is None:
            return loss.sum()
        else:
            if weight.device != loss.device:
                weight.to(loss.device)
            if weight.ndim != loss.ndim:
                raise ValueError(
                    f"`weight` and `loss` must have the same ndim."
                    f" But got: {weight.dim()} != {loss.dim()}"
                )
            loss *= weight
            return loss.sum()


def weighted_loss(f: Callable):
    """
    A decorator that allows weighted loss calculation between multiple inputs
    (predictions from  multistage models) and a single target.
    """
    @functools.wraps(f)
    def wrapper(
        input             : Tensors,
        target            : Tensor | None,
        input_weight      : Tensor | None = None,
        elementwise_weight: Tensor | None = None,
        reduction         : Reduction_    = "mean",
        *args, **kwargs
    ) -> Tensor:
        """
        
        Args:
            input (Tensors): A collection of tensors resulted from multistage
                predictions. Each tensor is of shape [B, C, H, W].
            target (Tensor | None): A tensor of shape [B, C, H, W] containing
                the ground-truth. If None, then it is used for unsupervised
                learning task.
            input_weight (Tensor | None): A manual scaling weight given to each
                tensor in `input`. If specified, it must have the length equals
                to the length of `input`.
            elementwise_weight (Tensor | None): A manual scaling weight given
                to each item in the batch. For classification, it is the weight
                for each class.
            reduction (Reduction_): Reduction option.
           
        Returns:
            Loss tensor.
        """
        if isinstance(input, Tensor):  # Single output
            input = [input]
        elif isinstance(input, dict):
            input = list(input.values())
        assert_sequence(input)
        
        losses = torch.FloatTensor([
            reduce_loss(
                loss      = f(input=i, target=target, *args, **kwargs),
                weight    = elementwise_weight,
                reduction = reduction
            ) for i in input
        ])
        
        return reduce_loss(
            loss      = losses,
            weight    = input_weight,
            reduction = Reduction.WEIGHTED_SUM if input_weight else Reduction.SUM,
        )

    return wrapper
    
    
# H1: - Base Loss --------------------------------------------------------------

class BaseLoss(_Loss, metaclass=ABCMeta):
    """
    Base Loss class.
    
    Args:
        weight (Tensor | None): Some loss function is the combination of other
            loss functions. This provides weight for each loss component.
            Defaults to [1.0].
        input_weight (Tensor | None): A manual scaling weight given to each
            tensor in `input`. If specified, it must have the length equals
            to the length of `input`.
        elementwise_weight (Tensor | None): A manual scaling weight given
            to each item in the batch. For classification, it is the weight
            for each class.
        reduction (str): Specifies the reduction to apply to the output.
            One of: [`none`, `mean`, `sum`].
            - none: No reduction will be applied.
            - mean: The sum of the output will be divided by the number of
              elements in the output.
            - sum: The output will be summed.
            Defaults to mean.
    """
    
    # If your loss function only support some reduction. Consider overwriting
    # this value.
    reductions = ["none", "mean", "sum", "weighted_sum"]
    
    def __init__(
        self,
        weight            : Tensor | None = Tensor([1.0]),
        input_weight      : Tensor | None = None,
        elementwise_weight: Tensor | None = None,
        reduction         : str           = "mean",
    ):
        super().__init__(reduction=reduction)
        self.weight             = weight
        self.input_weight       = input_weight
        self.elementwise_weight = elementwise_weight
        if self.reduction not in self.reductions:
            raise ValueError(
                f"`reduction` must be one of: {self.reductions}. "
                f"But got: {self.reduction}."
            )
    
    @abstractmethod
    def forward(self, input: Tensors, target: Tensor, **_) -> Tensor:
        pass
    

# H1: - Loss -------------------------------------------------------------------

@weighted_loss
def charbonnier_loss(
    input: Tensor, target: Tensor, eps: float = 1e-3, **_
) -> Tensor:
    """
    Charbonnier loss.
    
    Args:
        input (Tensor): The input tensor of shape [B, C, H, W].
        target (Tensor): The target tensor of shape [B, C, H, W].
        eps (float): Small value for numerically stability when dividing.
            Defaults to 1e-3.
            
    Returns:
        The loss tensor of shape [B].
    """
    return torch.sqrt((input - target) ** 2 + (eps * eps))


@weighted_loss
def color_constancy_loss(input: Tensor, target: None = None, **_) -> Tensor:
    """
    A color constancy loss to correct the potential color deviations in the
    enhanced image and also build the relations among the three adjusted
    channels.

    References:
        https://github.com/Li-Chongyi/Zero-DCE/blob/master/Zero-DCE_code/Myloss.py

    Args:
        input (Tensor): The input tensor of shape [B, C, H, W].
        target (Tensor): The target tensor of shape [B, C, H, W].
            Defaults to None.
        
    Returns:
        The loss tensor of shape [B].
    """
    mean_rgb   = torch.mean(input, [2, 3], keepdim=True)
    mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
    d_rg       = torch.pow(mr - mg, 2)
    d_rb       = torch.pow(mr - mb, 2)
    d_gb       = torch.pow(mb - mg, 2)
    loss = torch.pow(
        torch.pow(d_rg, 2) + torch.pow(d_rb, 2) + torch.pow(d_gb, 2), 0.5
    )
    return loss


@weighted_loss
def edge_loss(input: Tensor, target: Tensor, eps: float = 1e-3, **_) -> Tensor:
    """
    Edge loss.
    
    Args:
        input (Tensor): The input tensor of shape [B, C, H, W].
        target (Tensor): The target tensor of shape [B, C, H, W].
        eps (float): Small value for numerically stability when dividing.
            Defaults to 1e-3.
            
    Returns:
        The loss tensor of shape [B].
    """
    k 	   = Tensor([[0.05, 0.25, 0.4, 0.25, 0.05]])
    kernel = torch.matmul(k.t(), k).unsqueeze(0).repeat(3, 1, 1, 1)

    def conv_gauss(image: Tensor) -> Tensor:
        global kernel
        if kernel.device != image.device:
            kernel = kernel.to(image.device)
        n_channels, _, kw, kh = kernel.shape
        image = F.pad(image, (kw // 2, kh // 2, kw // 2, kh // 2), mode="replicate")
        return F.conv2d(image, kernel, groups=n_channels)
    
    def laplacian_kernel(image: Tensor) -> Tensor:
        filtered   = conv_gauss(image)  		# filter
        down 	   = filtered[:, :, ::2, ::2]   # downsample
        new_filter = torch.zeros_like(filtered)
        new_filter[:, :, ::2, ::2] = down * 4   # upsample
        filtered   = conv_gauss(new_filter)    # filter
        diff 	   = image - filtered
        return diff
    
    return torch.sqrt(
        (laplacian_kernel(input) - laplacian_kernel(target)) ** 2 + (eps * eps)
    )


@weighted_loss
def exposure_control_loss(
    input     : Tensor,
    patch_size: Ints,
    mean_val  : float,
    target    : None = None,
    **_
) -> Tensor:
    """
    Exposure Control Loss measures the distance between the average intensity
    value of a local region to the well-exposedness level E.

    References:
        https://github.com/Li-Chongyi/Zero-DCE/blob/master/Zero-DCE_code/Myloss.py

    Args:
        input (Tensor): The input tensor of shape [B, C, H, W].
        target (None): The target tensor of shape [B, C, H, W]. Default to None.
        patch_size (Ints): Kernel size for pooling layer.
    	mean_val (float):
            
    Returns:
        The loss tensor of shape [B].
    """
    pool = nn.AvgPool2d(patch_size)
    mean = torch.mean(input, 1, keepdim=True)
    mean = pool(mean)
    loss = torch.pow(mean - torch.FloatTensor([mean_val]).to(input.device), 2)
    return loss


@weighted_loss
def gray_loss(input: Tensor, target: None = None, **_) -> Tensor:
    """
    Gray loss.

    Args:
        input (Tensor): The input tensor of shape [B, C, H, W].
        target (Tensor): The target tensor of shape [B, C, H, W].
            Defaults to None.
        
    Returns:
        The loss tensor of shape [B].
    """
    return 1.0 / mse_loss(
        input     = input,
        target    = torch.ones_like(input) * 0.5,
        reduction = "mean"
    )


@weighted_loss
def illumination_smoothness_loss(
    input: Tensor, tv_loss_weight: int, target: None = None, **_
) -> Tensor:
    """
    Illumination Smoothness Loss preserve the mono-tonicity relations between
    neighboring pixels, we add an illumination smoothness loss to each curve
    parameter map A.
    
    References:
        https://github.com/Li-Chongyi/Zero-DCE/blob/master/Zero-DCE_code/Myloss.py

    Args:
        input (Tensor): The input tensor of shape [B, C, H, W].
        target (Tensor): The target tensor of shape [B, C, H, W].
            Defaults to None.
        tv_loss_weight (int):
        
    Returns:
        The loss tensor of shape [B].
    """
    x          = input
    batch_size = x.size()[0]
    h_x        = x.size()[2]
    w_x        = x.size()[3]
    count_h    = (x.size()[2] - 1) * x.size()[3]
    count_w    = x.size()[2] * (x.size()[3] - 1)
    h_tv       = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x-1, :]), 2).sum()
    w_tv       = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x-1]), 2).sum()
    return tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size


@weighted_loss
def mae_loss(input: Tensor, target: Tensor, **_) -> Tensor:
    """
    MAE (Mean Absolute Error or L1) loss.
    
    Args:
        input (Tensor): The input tensor of shape [B, C, H, W].
        target (Tensor): The target tensor of shape [B, C, H, W].
    
    Returns:
        The loss tensor of shape [B].
    """
    return F.l1_loss(input=input, target=target, reduction="none")


@weighted_loss
def mse_loss(input: Tensor, target: Tensor, **_) -> Tensor:
    """
    MSE (Mean Squared Error or L2) loss.
    
    Args:
        input (Tensor): The input tensor of shape [B, C, H, W].
        target (Tensor): The target tensor of shape [B, C, H, W].
    
    Returns:
        The loss tensor of shape [B].
    """
    return F.mse_loss(input=input, target=target, reduction="none")


@weighted_loss
def non_blurry_loss(input: Tensor, target: None = None, **_) -> Tensor:
    """
    Non-blurry Loss.
    
    Args:
        input (Tensor): The input tensor of shape [B, C, H, W].
        target (Tensor): The target tensor of shape [B, C, H, W].
            Defaults to None.
    
    Returns:
        The loss tensor of shape [B].
    """
    return 1.0 - F.mse_loss(
        input     = input,
        target    = torch.ones_like(input) * 0.5,
        reduction = "mean"
    )


@weighted_loss
def psnr_loss(
    input: Tensor, target: Tensor, max_val: float = 1.0, **_
) -> Tensor:
    """
    PSNR loss. Modified from BasicSR: https://github.com/xinntao/BasicSR
    
    Args:
        input (Tensor): The input tensor of shape [B, C, H, W].
        target (Tensor): The target tensor of shape [B, C, H, W].
        max_val (float): Dynamic range of the images. Defaults to 1.0.
        
    Returns:
        The loss tensor of shape [B].
    """
    return -1.0 * psnr(input=input, target=target, max_val=max_val)


@weighted_loss
def smooth_mae_loss(
    input: Tensor, target: Tensor, beta: float = 1.0, **_
) -> Tensor:
    """
    Smooth MAE (Mean Absolute Error or L1) loss.
    
    Args:
        input (Tensor): The input tensor of shape [B, C, H, W].
        target (Tensor): The target tensor of shape [B, C, H, W].
        beta (float):
        
    Returns:
        The loss tensor of shape [B].
    """
    return F.smooth_l1_loss(
        input     = input,
        target    = target,
        beta      = beta,
        reduction = "none"
    )


@weighted_loss
def ssim_loss(
    input      : Tensor,
    target     : Tensor,
    window_size: int,
    max_val    : float = 1.0,
    eps        : float = 1e-12,
    **_
) -> Tensor:
    """
    PSNR loss. Modified from BasicSR: https://github.com/xinntao/BasicSR
    
    Args:
        input (Tensor): The input tensor of shape [B, C, H, W].
        target (Tensor): The target tensor of shape [B, C, H, W].
        window_size (int): Size of the gaussian kernel to smooth the images.
        max_val (float): Dynamic range of the images. Defaults to 1.0.
        eps (float): Small value for numerically stability when dividing.
            Defaults to 1e-12.
        
    Returns:
        The loss tensor of shape [B].
    """
    # Compute the ssim map
    ssim_map = ssim(
        input       = input,
        target      = target,
        window_size = window_size,
        max_val     = max_val,
        eps         = eps
    )
    # Compute and reduce the loss
    return torch.clamp((1.0 - ssim_map) / 2, min=0, max=1)


@weighted_loss
def spatial_consistency_loss(input: Tensor, target: Tensor, **_) -> Tensor:
    """
    Spatial Consistency Loss encourages spatial coherence of the enhanced
    image through preserving the difference of neighboring regions between the
    input image and its enhanced version.
    
    References:
        https://github.com/Li-Chongyi/Zero-DCE/blob/master/Zero-DCE_code/Myloss.py
    
    Args:
        input (Tensor): The input tensor of shape [B, C, H, W].
        target (Tensor): The target tensor of shape [B, C, H, W]. In this case,
            the enhanced image, i.e, prediction.
    
    Returns:
        The loss tensor of shape [B].
    """
    kernel_left  = FloatTensor([[0,  0, 0], [-1, 1,  0], [0,  0, 0]]).unsqueeze(0).unsqueeze(0)
    kernel_right = FloatTensor([[0,  0, 0], [ 0, 1, -1], [0,  0, 0]]).unsqueeze(0).unsqueeze(0)
    kernel_up    = FloatTensor([[0, -1, 0], [ 0, 1,  0], [0,  0, 0]]).unsqueeze(0).unsqueeze(0)
    kernel_down  = FloatTensor([[0,  0, 0], [ 0, 1,  0], [0, -1, 0]]).unsqueeze(0).unsqueeze(0)
    
    weight_left  = nn.Parameter(data=kernel_left,  requires_grad=False)
    weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
    weight_up    = nn.Parameter(data=kernel_up,    requires_grad=False)
    weight_down  = nn.Parameter(data=kernel_down,  requires_grad=False)
    pool         = nn.AvgPool2d(4)
    
    if weight_left.device != input.device:
        weight_left = weight_left.to(input.device)
    if weight_right.device != input.device:
        weight_right = weight_right.to(input.device)
    if weight_up.device != input.device:
        weight_up = weight_up.to(input.device)
    if weight_down.device != input.device:
        weight_down = weight_down.to(input.device)
    
    input_mean      = torch.mean(input,  1, keepdim=True)
    target_mean     = torch.mean(target, 1, keepdim=True)

    input_pool      = pool(input_mean)
    target_pool     = pool(target_mean)

    d_org_left      = F.conv2d(input_pool, weight_left,  padding=1)
    d_org_right     = F.conv2d(input_pool, weight_right, padding=1)
    d_org_up        = F.conv2d(input_pool, weight_up,    padding=1)
    d_org_down      = F.conv2d(input_pool, weight_down,  padding=1)

    d_enhance_left  = F.conv2d(target_pool, weight_left,  padding=1)
    d_enhance_right = F.conv2d(target_pool, weight_right, padding=1)
    d_enhance_up    = F.conv2d(target_pool, weight_up,    padding=1)
    d_enhance_down  = F.conv2d(target_pool, weight_down,  padding=1)

    d_left          = torch.pow(d_org_left  - d_enhance_left,  2)
    d_right         = torch.pow(d_org_right - d_enhance_right, 2)
    d_up            = torch.pow(d_org_up    - d_enhance_up,    2)
    d_down          = torch.pow(d_org_down  - d_enhance_down,  2)
    loss            = d_left + d_right + d_up + d_down
    
    return loss


class CharbonnierLoss(BaseLoss):
    
    def __init__(self, eps: float = 1e-3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eps  = eps
        self.name = "charbonnier_loss"
     
    def forward(self, input: Tensors, target: Tensor = None, **_) -> Tensor:
        return self.weight * charbonnier_loss(
            input              = input,
            target             = target,
            eps                = self.eps,
            input_weight       = self.input_weight,
            elementwise_weight = self.elementwise_weight,
            reduction          = self.reduction,
        )


class CharbonnierEdgeLoss(BaseLoss):
    """
    Implementation of the loss function proposed in the paper "Multi-Stage
    Progressive Image Restoration".
    """
    
    def __init__(
        self,
        eps   : float = 1e-3,
        weight: Tensor = Tensor([1.0, 0.05]),
        *args, **kwargs
    ):
        super().__init__(weight=weight, *args, **kwargs)
        self.eps  = eps
        self.name = "charbonnier_edge_loss"
     
    def forward(self, input: Tensors, target: Tensor = None, **_) -> Tensor:
        return \
            self.weight[0] * charbonnier_loss(
                input              = input,
                target             = target,
                eps                = self.eps,
                input_weight       = self.input_weight,
                elementwise_weight = self.elementwise_weight,
                reduction          = self.reduction,
            ) + \
            self.weight[1] * edge_loss(
                input              = input,
                target             = target,
                eps                = self.eps,
                input_weight       = self.input_weight,
                elementwise_weight = self.elementwise_weight,
                reduction          = self.reduction,
            )


class ColorConstancyLoss(BaseLoss):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "color_constancy_loss"
     
    def forward(self, input: Tensors, target: Tensor = None, **_) -> Tensor:
        return self.weight * color_constancy_loss(
            input              = input,
            target             = target,
            input_weight       = self.input_weight,
            elementwise_weight = self.elementwise_weight,
            reduction          = self.reduction,
        )


class EdgeLoss(BaseLoss):
    
    def __init__(self, eps: float = 1e-3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eps  = eps
        self.name = "edge_loss"
     
    def forward(self, input: Tensors, target: Tensor = None, **_) -> Tensor:
        return self.weight * edge_loss(
            input              = input,
            target             = target,
            eps                = self.eps,
            input_weight       = self.input_weight,
            elementwise_weight = self.elementwise_weight,
            reduction          = self.reduction,
        )
    
    
class ExposureControlLoss(BaseLoss):
    """
    Exposure Control Loss.
    """
    
    def __init__(self, patch_size : Ints, mean_val: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name       = "exposure_control_loss"
        self.patch_size = patch_size
        self.mean_val   = mean_val
     
    def forward(self, input: Tensors, target: None = None, **_) -> Tensor:
        return self.weight * exposure_control_loss(
            input              = input,
            target             = target,
            patch_size         = self.patch_size,
            mean_val           = self.mean_val,
            input_weight       = self.input_weight,
            elementwise_weight = self.elementwise_weight,
            reduction          = self.reduction,
        )


class GradientLoss(BaseLoss):
    """
    L1 loss on the gradient of the image.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "gradient_loss"
     
    def forward(self, input: Tensors, target: None = None, **_) -> Tensor:
        if isinstance(input, Tensor):  # Single output
            input = [input]
        elif isinstance(input, dict):
            input = list(input.values())
        assert_sequence(input)
        
        losses = []
        for i in input:
            gradient_a_x = torch.abs(i[:, :, :, :-1] - i[:, :, :, 1:])
            gradient_a_y = torch.abs(i[:, :, :-1, :] - i[:, :, 1:, :])
            losses.append(
                reduce_loss(
                    loss      = torch.mean(gradient_a_x) + torch.mean(gradient_a_y),
                    weight    = self.elementwise_weight,
                    reduction = self.reduction
                )
            )
        return self.weight * reduce_loss(
            loss      = torch.FloatTensor(losses),
            weight    = self.input_weight,
            reduction = Reduction.WEIGHTED_SUM
        )
        

class GrayLoss(BaseLoss):
    """
    Gray Loss.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "gray_loss"
     
    def forward(self, input: Tensors, target: None = None, **_) -> Tensor:
        return self.weight * mae_loss(
            input              = input,
            target             = target,
            input_weight       = self.input_weight,
            elementwise_weight = self.elementwise_weight,
            reduction          = self.reduction,
        )


class GrayscaleLoss(BaseLoss):
    """
    Grayscale Loss.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "grayscale_loss"
     
    def forward(self, input: Tensors, target: Tensor, **_) -> Tensor:
        if isinstance(input, Tensor):  # Single output
            input = [input]
        elif isinstance(input, dict):
            input = list(input.values())
        assert_sequence(input)
        
        input_g  = [torch.mean(i,  1, keepdim=True) for i in input]
        target_g = torch.mean(target, 1, keepdim=True)
        
        return self.weight * mse_loss(
            input              = input_g,
            target             = target_g,
            input_weight       = self.input_weight,
            elementwise_weight = self.elementwise_weight,
            reduction          = self.reduction,
        )


class IlluminationSmoothnessLoss(BaseLoss):
    
    def __init__(self, tv_loss_weight: int = 1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name           = "illumination_smoothness_loss"
        self.tv_loss_weight = tv_loss_weight
     
    def forward(self, input: Tensors, target: None = None, **_) -> Tensor:
        return self.weight * illumination_smoothness_loss(
            input              = input,
            target             = target,
            tv_loss_weight     = self.tv_loss_weight,
            input_weight       = self.input_weight,
            elementwise_weight = self.elementwise_weight,
            reduction          = self.reduction,
        )

    
class MAELoss(BaseLoss):
    """
    MAE (Mean Absolute Error or L1) loss.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "mae_loss"
     
    def forward(self, input: Tensors, target: Tensor, **_) -> Tensor:
        return self.weight * mae_loss(
            input              = input,
            target             = target,
            input_weight       = self.input_weight,
            elementwise_weight = self.elementwise_weight,
            reduction          = self.reduction,
        )


class MSELoss(BaseLoss):
    """
    MSE (Mean Squared Error or L2) loss.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "mse_loss"
     
    def forward(self, input: Tensors, target: Tensor, **_) -> Tensor:
        return self.weight * mse_loss(
            input              = input,
            target             = target,
            input_weight       = self.input_weight,
            elementwise_weight = self.elementwise_weight,
            reduction          = self.reduction,
        )


class NonBlurryLoss(BaseLoss):
    """
    MSELoss on the distance to 0.5
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "non_blurry_loss"
     
    def forward(self, input: Tensors, target: None = None, **_) -> Tensor:
        return self.weight * non_blurry_loss(
            input              = input,
            target             = target,
            input_weight       = self.input_weight,
            elementwise_weight = self.elementwise_weight,
            reduction          = self.reduction,
        )


class PerceptualL1Loss(BaseLoss):
    """
    Loss = weights[0] * Perceptual Loss + weights[1] * L1 Loss.
    """
    
    def __init__(
        self,
        vgg   : nn.Module,
        weight: Tensor = Tensor([1.0, 1.0]),
        *args, **kwargs
    ):
        super().__init__(weight=weight, *args, **kwargs)
        self.name     = "perceptual_Loss"
        self.per_loss = PerceptualLoss(vgg=vgg, *args, **kwargs)
        self.l1_loss  = L1Loss(*args, **kwargs)
        self.layer_name_mapping = {
            "3" : "relu1_2",
            "8" : "relu2_2",
            "15": "relu3_3"
        }
        
        if self.weight is None:
            self.weight = Tensor([1.0, 1.0])
        elif len(self.weight) != 2:
            raise ValueError(f"Length of `weight` must be 2. "
                             f"But got: {len(self.weight)}." )
     
    def forward(self, input: Tensors, target: Tensor, **_) -> Tensor:
        return \
            self.weight[0] * self.per_loss(
                input              = input,
                target             = target,
                eps                = self.eps,
                input_weight       = self.input_weight,
                elementwise_weight = self.elementwise_weight,
                reduction          = self.reduction,
            ) + \
            self.weight[1] * self.l1_loss(
                input              = input,
                target             = target,
                eps                = self.eps,
                input_weight       = self.input_weight,
                elementwise_weight = self.elementwise_weight,
                reduction          = self.reduction,
            )


class PerceptualLoss(BaseLoss):
    """
    Perceptual Loss.
    """
    
    def __init__(self, vgg: nn.Module, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "perceptual_Loss"
        self.vgg  = vgg
        self.vgg.freeze()
     
    def forward(self, input: Tensors, target: Tensor, **_) -> Tensor:
        if isinstance(input, Tensor):
            input = [input]
        elif isinstance(input, dict):
            input = list(input.values())
        assert_sequence(input)
        
        if self.vgg.device != input[0].device:
            self.vgg = self.vgg.to(input[0].device)
        
        losses = []
        for i in input:
            input_features  = self.vgg.forward_features(i)
            target_features = self.vgg.forward_features(target)
            losses.append(
                reduce_loss(
                    loss      = F.mse_loss(input_features, target_features),
                    weight    = self.elementwise_weight,
                    reduction = self.reduction
                )
            )
        return self.weight * reduce_loss(
            loss      = torch.FloatTensor(losses),
            weight    = self.input_weight,
            reduction = Reduction.WEIGHTED_SUM
        )


class PSNRLoss(BaseLoss):
    """
    PSNR Loss.
    """
    
    def __init__(self, max_val: float = 1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name    = "psnr_loss"
        self.max_val = max_val
     
    def forward(self, input: Tensors, target: Tensor, **_) -> Tensor:
        return self.weight * psnr_loss(
            input              = input,
            target             = target,
            max_val            = self.max_val,
            input_weight       = self.input_weight,
            elementwise_weight = self.elementwise_weight,
            reduction          = self.reduction,
        )


class SmoothMAELoss(BaseLoss):
    """
    Smooth MAE (Mean Absolute Error or L1) loss.
    """
    
    def __init__(self, beta: float = 1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "smooth_mae_loss"
        self.beta = beta
     
    def forward(self, input: Tensors, target: Tensor, **_) -> Tensor:
        return self.weight * smooth_mae_loss(
            input              = input,
            target             = target,
            beta               = self.beta,
            input_weight       = self.input_weight,
            elementwise_weight = self.elementwise_weight,
            reduction          = self.reduction,
        )


class SpatialConsistencyLoss(BaseLoss):
    """
    Spatial Consistency Loss (SPA) Loss.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "spatial_consistency_loss"
     
    def forward(self, input: Tensors, target: Tensor, **_) -> Tensor:
        return self.weight * spatial_consistency_loss(
            input              = input,
            target             = target,
            input_weight       = self.input_weight,
            elementwise_weight = self.elementwise_weight,
            reduction          = self.reduction,
        )


class SSIMLoss(BaseLoss):
    """
    SSIM Loss.
    
    Args:
        window_size (int): Size of the gaussian kernel to smooth the images.
        max_val (float): Dynamic range of the images. Defaults to 1.0.
        eps (float): Small value for numerically stability when dividing.
            Defaults to 1e-12.
    """
    
    def __init__(
        self,
        window_size: int,
        max_val    : float = 1.0,
        eps        : float = 1e-12,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.name        = "ssim_loss"
        self.window_size = window_size
        self.max_val     = max_val
        self.eps         = eps
    
    def forward(self, input: Tensors, target: Tensor, **_) -> Tensor:
        return self.weight * ssim_loss(
            input              = input,
            target             = target,
            window_size        = self.window_size,
            max_val            = self.max_val,
            eps                = self.eps,
            input_weight       = self.input_weight,
            elementwise_weight = self.elementwise_weight,
            reduction          = self.reduction,
        )


class StdLoss(BaseLoss):
    """
    Loss on the variance of the image. Works in the grayscale. If the image is
    smooth, gets zero.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "std_loss"
        
        blur        = (1 / 25) * np.ones((5, 5))
        blur        = blur.reshape(1, 1, blur.shape[0], blur.shape[1])
        blur        = nn.Parameter(data=torch.FloatTensor(blur), requires_grad=False)
        self.blur   = blur
        image       = np.zeros((5, 5))
        image[2, 2] = 1
        # noinspection PyArgumentList
        image       = image.reshape(1, 1, image.shape[0], image.shape[1])
        image       = nn.Parameter(data=torch.FloatTensor(image), requires_grad=False)
        self.image  = image
     
    def forward(self, input: Tensors, target: Tensor, **_) -> Tensor:
        if isinstance(input, Tensor):
            input = [input]
        elif isinstance(input, dict):
            input = list(input.values())
        assert_sequence(input)

        if self.blur.device != input[0].device:
            self.blur = self.blur.to(input[0].device)
        if self.image.device != input[0].device:
            self.image = self.image.to(input[0].device)
        
        losses = []
        for i in input:
            i_mean = torch.mean(i, 1, keepdim=True)
            losses.append(
                reduce_loss(
                    loss      = F.mse_loss(
                        functional.conv2d(i_mean, self.image),
                        functional.conv2d(i_mean, self.blur),
                    ),
                    weight    = self.elementwise_weight,
                    reduction = self.reduction
                )
            )
        return self.weight * reduce_loss(
            loss      = torch.FloatTensor(losses),
            weight    = self.input_weight,
            reduction = Reduction.WEIGHTED_SUM
        )


L1Loss       = MAELoss
L2Loss       = MSELoss
SmoothL1Loss = SmoothMAELoss

LOSSES.register(name="bce_loss", 			              module=nn.BCELoss)
LOSSES.register(name="bce_with_logits_loss",              module=nn.BCEWithLogitsLoss)
LOSSES.register(name="charbonnier_loss",                  module=CharbonnierLoss)
LOSSES.register(name="charbonnier_edge_loss",             module=CharbonnierEdgeLoss)
LOSSES.register(name="color_constancy_loss",              module=ColorConstancyLoss)
LOSSES.register(name="cosine_embedding_loss",             module=nn.CosineEmbeddingLoss)
LOSSES.register(name="cross_entropy_loss",	              module=nn.CrossEntropyLoss)
LOSSES.register(name="ctc_loss",	  		              module=nn.CTCLoss)
LOSSES.register(name="edge_loss",	  		              module=EdgeLoss)
LOSSES.register(name="exposure_control_loss",	  		  module=ExposureControlLoss)
LOSSES.register(name="gaussian_nll_loss",                 module=nn.GaussianNLLLoss)
LOSSES.register(name="gradient_loss",                     module=GradientLoss)
LOSSES.register(name="gray_loss",                         module=GrayLoss)
LOSSES.register(name="grayscale_loss",                    module=GrayscaleLoss)
LOSSES.register(name="hinge_embedding_loss",              module=nn.HingeEmbeddingLoss)
LOSSES.register(name="huber_loss",                        module=nn.HuberLoss)
LOSSES.register(name="illumination_smoothness_loss",      module=IlluminationSmoothnessLoss)
LOSSES.register(name="kl_div_loss",	  		              module=nn.KLDivLoss)
LOSSES.register(name="l1_loss",	  		                  module=L1Loss)
LOSSES.register(name="l2_loss",	  		                  module=L2Loss)
LOSSES.register(name="margin_ranking_loss",	              module=nn.MarginRankingLoss)
LOSSES.register(name="mae_loss",	  		              module=MAELoss)
LOSSES.register(name="mse_loss",	  		              module=MSELoss)
LOSSES.register(name="multi_label_margin_loss",           module=nn.MultiLabelMarginLoss)
LOSSES.register(name="multi_label_soft_margin_loss",      module=nn.MultiLabelSoftMarginLoss)
LOSSES.register(name="multi_margin_loss",                 module=nn.MultiMarginLoss)
LOSSES.register(name="nll_loss",   	   		              module=nn.NLLLoss)
LOSSES.register(name="nll_loss2d",   	   		          module=nn.NLLLoss2d)
LOSSES.register(name="non_blurry_loss",   	   		      module=NonBlurryLoss)
LOSSES.register(name="perceptual_l1_loss",   	          module=PerceptualL1Loss)
LOSSES.register(name="perceptual_loss",   	              module=PerceptualLoss)
LOSSES.register(name="poisson_nll_loss",   	              module=nn.PoissonNLLLoss)
LOSSES.register(name="psnr_loss",   	                  module=PSNRLoss)
LOSSES.register(name="smooth_l1_loss",   	              module=SmoothL1Loss)
LOSSES.register(name="smooth_mae_loss",   	              module=SmoothMAELoss)
LOSSES.register(name="soft_margin_loss",   	              module=nn.SoftMarginLoss)
LOSSES.register(name="spatial_consistency_loss",   	      module=SpatialConsistencyLoss)
LOSSES.register(name="ssim_loss",   	                  module=SSIMLoss)
LOSSES.register(name="std_loss",   	                      module=StdLoss)
LOSSES.register(name="triplet_margin_loss",               module=nn.TripletMarginLoss)
LOSSES.register(name="triplet_margin_with_distance_Loss", module=nn.TripletMarginWithDistanceLoss)

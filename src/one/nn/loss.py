#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Loss Functions
"""

from __future__ import annotations

from abc import ABCMeta

from torch import FloatTensor
from torch import Tensor
from torch.nn import functional
from torch.nn.modules.loss import *
from torch.nn.modules.loss import _Loss

from one.constants import *
from one.core import *
from one.nn.metric import psnr
from one.nn.metric import ssim


# H1: - Base Loss --------------------------------------------------------------

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
        return torch.mean(loss)
    if reduction == Reduction.SUM:
        return torch.sum(loss)
    if reduction == Reduction.WEIGHTED_SUM:
        if weight is None:
            return torch.sum(loss)
        else:
            if weight.device != loss.device:
                weight.to(loss.device)
            if weight.ndim != loss.ndim:
                raise ValueError(
                    f"`weight` and `loss` must have the same ndim."
                    f" But got: {weight.dim()} != {loss.dim()}"
                )
            loss *= weight
            return torch.sum(loss)


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
        
        losses = Tensor([
            reduce_loss(
                loss      = f(input=i, target=target, *args, **kwargs),
                weight    = elementwise_weight,
                reduction = reduction
            ) for i in input
        ])
        
        loss = reduce_loss(
            loss      = losses,
            weight    = input_weight,
            reduction = Reduction.WEIGHTED_SUM if input_weight else Reduction.SUM,
        )
        loss.requires_grad = True
        return loss

    return wrapper
    

class BaseLoss(_Loss, metaclass=ABCMeta):
    """
    Base Loss class.
    
    Args:
        weight (Tensor): Some loss function is the combination of other loss
            functions. This provides weight for each loss component.
            Defaults to 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            One of: [`none`, `mean`, `sum`, `weighted_sum`].
            - none: No reduction will be applied.
            - mean: The sum of the output will be divided by the number of
              elements in the output.
            - sum: The output will be summed.
            Defaults to mean.
    """
    
    # If your loss function only support some reduction.
    # Consider overwriting this value.
    reductions = ["none", "mean", "sum", "weighted_sum"]
    
    def __init__(
        self,
        weight   : Floats = 1.0,
        reduction: str    = "mean",
        *args, **kwargs
    ):
        super().__init__(reduction=reduction)
        weight = weight or [1.0]
        if not isinstance(weight, Sequence):
            weight = [weight]
        self.weight = weight
        if self.reduction not in self.reductions:
            raise ValueError(
                f"`reduction` must be one of: {self.reductions}. "
                f"But got: {self.reduction}."
            )
    
    @classmethod
    @property
    def classname(cls) -> str:
        """
        Returns the name of the class of the object passed to it.
        
        Returns:
            The class name of the object.
        """
        return cls.__name__
    
    @abstractmethod
    def forward(self, input: Tensor, target: Tensor, **_) -> Tensor:
        pass
    

# H1: - Loss -------------------------------------------------------------------

@LOSSES.register(name="charbonnier_loss")
class CharbonnierLoss(BaseLoss):
    
    def __init__(
        self,
        eps      : float  = 1e-3,
        weight   : Floats = 1.0,
        reduction: str    = "mean",
        *args, **kwargs
    ):
        super().__init__(
            weight    = weight,
            reduction = reduction,
            *args, **kwargs
        )
        self.eps  = eps
        self.name = "charbonnier_loss"
     
    def forward(self, input: Tensors, target: Tensor = None, **_) -> Tensor:
        loss = torch.sqrt((input - target) ** 2 + (self.eps * self.eps))
        loss = reduce_loss(loss=loss, reduction=self.reduction)
        return self.weight[0] * loss


@LOSSES.register(name="charbonnier_edge_loss")
class CharbonnierEdgeLoss(BaseLoss):
    """
    Implementation of the loss function proposed in the paper "Multi-Stage
    Progressive Image Restoration".
    """
    
    def __init__(
        self,
        eps      : float  = 1e-3,
        weight   : Floats = [1.0, 0.05],
        reduction: str    = "mean",
        *args, **kwargs
    ):
        super().__init__(
            weight    = weight,
            reduction = reduction,
            *args, **kwargs
        )
        self.eps  = eps
        self.name = "charbonnier_edge_loss"
        self.charbonnier_loss = CharbonnierLoss(eps=eps, reduction=reduction)
        self.edge_loss        = EdgeLoss(eps=eps, reduction=reduction)
        
    def forward(self, input: Tensors, target: Tensor = None, **_) -> Tensor:
        charbonnier_loss = self.charbonnier_loss(input=input, target=target)
        edge_loss        = self.edge_loss(input=input, target=target)
        return self.weight[0] * charbonnier_loss \
               + self.weight[1] * edge_loss


@LOSSES.register(name="color_constancy_loss")
class ColorConstancyLoss(BaseLoss):
    """
    A color constancy loss to correct the potential color deviations in the
    enhanced image and also build the relations among the three adjusted
    channels.

    References:
        https://github.com/Li-Chongyi/Zero-DCE/blob/master/Zero-DCE_code/Myloss.py
    """
    
    def __init__(
        self,
        weight   : Floats = 1.0,
        reduction: str    = "mean",
        *args, **kwargs
    ):
        super().__init__(
            weight    = weight,
            reduction = reduction,
            *args, **kwargs
        )
        self.name = "color_constancy_loss"
     
    def forward(self, input: Tensors, target: Tensor = None, **_) -> Tensor:
        mean_rgb   = torch.mean(input, [2, 3], keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        d_rg       = torch.pow(mr - mg, 2)
        d_rb       = torch.pow(mr - mb, 2)
        d_gb       = torch.pow(mb - mg, 2)
        loss = torch.pow(
            torch.pow(d_rg, 2) + torch.pow(d_rb, 2) + torch.pow(d_gb, 2), 0.5
        )
        loss = reduce_loss(loss=loss, reduction=self.reduction)
        return self.weight[0] * loss


@LOSSES.register(name="edge_loss")
class EdgeLoss(BaseLoss):
    
    def __init__(
        self,
        eps      : float  = 1e-3,
        weight   : Floats = 1.0,
        reduction: str    = "mean",
        *args, **kwargs
    ):
        super().__init__(
            weight    = weight,
            reduction = reduction,
            *args, **kwargs
        )
        self.eps    = eps
        self.name   = "edge_loss"
        k 	        = Tensor([[0.05, 0.25, 0.4, 0.25, 0.05]])
        self.kernel = torch.matmul(k.t(), k).unsqueeze(0).repeat(3, 1, 1, 1)
        
    def forward(self, input: Tensors, target: Tensor = None, **_) -> Tensor:
        loss = torch.sqrt(
            (self.laplacian_kernel(input) - self.laplacian_kernel(target)) ** 2
            + (self.eps * self.eps)
        )
        loss = reduce_loss(loss=loss, reduction=self.reduction)
        return self.weight[0] * loss
    
    def conv_gauss(self, image: Tensor) -> Tensor:
        if self.kernel.device != image.device:
            self.kernel = self.kernel.to(image.device)
        n_channels, _, kw, kh = self.kernel.shape
        image = F.pad(image, (kw // 2, kh // 2, kw // 2, kh // 2), mode="replicate")
        return F.conv2d(image, self.kernel, groups=n_channels)
    
    def laplacian_kernel(self, image: Tensor) -> Tensor:
        filtered   = self.conv_gauss(image)  		# filter
        down 	   = filtered[:, :, ::2, ::2]       # downsample
        new_filter = torch.zeros_like(filtered)
        new_filter[:, :, ::2, ::2] = down * 4       # upsample
        filtered   = self.conv_gauss(new_filter)    # filter
        diff 	   = image - filtered
        return diff
    
    
@LOSSES.register(name="exposure_control_loss")
class ExposureControlLoss(BaseLoss):
    """
    Exposure Control Loss measures the distance between the average intensity
    value of a local region to the well-exposedness level E.

    References:
        https://github.com/Li-Chongyi/Zero-DCE/blob/master/Zero-DCE_code/Myloss.py

    Args:
        patch_size (Ints): Kernel size for pooling layer.
        mean_val (float):
        reduction (Reduction_): Reduction value to use.
    """
    
    def __init__(
        self,
        patch_size: Ints,
        mean_val  : float,
        weight    : Floats = 1.0,
        reduction : str    = "mean",
        *args, **kwargs
    ):
        super().__init__(
            weight    = weight,
            reduction = reduction,
            *args, **kwargs
        )
        self.name       = "exposure_control_loss"
        self.patch_size = patch_size
        self.mean_val   = mean_val
        self.pool       = nn.AvgPool2d(self.patch_size)
     
    def forward(self, input: Tensors, target: None = None, **_) -> Tensor:
        x    = input
        x    = torch.mean(x, 1, keepdim=True)
        mean = self.pool(x)
        loss = torch.pow(mean - torch.FloatTensor([self.mean_val]).to(input.device), 2)
        loss = reduce_loss(loss=loss, reduction=self.reduction)
        return self.weight[0] * loss
        

@LOSSES.register(name="gradient_loss")
class GradientLoss(BaseLoss):
    """
    L1 loss on the gradient of the image.
    """
    
    def __init__(
        self,
        weight   : Tensor = 1.0,
        reduction: str    = "mean",
        *args, **kwargs
    ):
        super().__init__(
            weight    = weight,
            reduction = reduction,
            *args, **kwargs
        )
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
                    reduction = self.reduction
                )
            )
        return self.weight[0] * reduce_loss(
            loss      = torch.FloatTensor(losses),
            reduction = Reduction.WEIGHTED_SUM
        )
        

@LOSSES.register(name="gray_loss")
class GrayLoss(BaseLoss):
    """
    Gray Loss.
    """
    
    def __init__(
        self,
        weight   : Floats = 1.0,
        reduction: str    = "mean",
        *args, **kwargs
    ):
        super().__init__(
            weight    = weight,
            reduction = reduction,
            *args, **kwargs
        )
        self.name = "gray_loss"
     
    def forward(self, input: Tensors, target: None = None, **_) -> Tensor:
        mse_loss = F.mse_loss(
            input     = input,
            target    = torch.ones_like(input) * 0.5,
            reduction = self.reduction
        )
        loss = 1.0 / mse_loss
        return self.weight[0] * loss


@LOSSES.register(name="grayscale_loss")
class GrayscaleLoss(BaseLoss):
    """
    Grayscale Loss.
    """
    
    def __init__(
        self,
        weight   : Floats = 1.0,
        reduction: str    = "mean",
        *args, **kwargs
    ):
        super().__init__(
            weight    = weight,
            reduction = reduction,
            *args, **kwargs
        )
        self.name = "grayscale_loss"
     
    def forward(self, input: Tensors, target: Tensor, **_) -> Tensor:
        if isinstance(input, Tensor):  # Single output
            input = [input]
        elif isinstance(input, dict):
            input = list(input.values())
        assert_sequence(input)
        
        input_g  = [torch.mean(i,  1, keepdim=True) for i in input]
        target_g = torch.mean(target, 1, keepdim=True)
        
        loss = Tensor([
            F.mse_loss(
                input     = i,
                target    = target_g,
                reduction = self.reduction
            )
            for i in input_g
        ])
        loss = reduce_loss(loss=loss, reduction=self.reduction)
        return self.weight[0] * loss


@LOSSES.register(name="illumination_smoothness_loss")
class IlluminationSmoothnessLoss(BaseLoss):
    """
    Illumination Smoothness Loss preserve the mono-tonicity relations between
    neighboring pixels, we add an illumination smoothness loss to each curve
    parameter map A.
    
    References:
        https://github.com/Li-Chongyi/Zero-DCE/blob/master/Zero-DCE_code/Myloss.py
    """
    
    def __init__(
        self,
        tv_loss_weight: int    = 1,
        weight        : Floats = 1.0,
        reduction     : str    = "mean",
        *args, **kwargs
    ):
        super().__init__(
            weight    = weight,
            reduction = reduction,
            *args, **kwargs
        )
        self.name           = "illumination_smoothness_loss"
        self.tv_loss_weight = tv_loss_weight
     
    def forward(self, input: Tensors, target: None = None, **_) -> Tensor:
        x       = input
        b       = x.size()[0]
        h_x     = x.size()[2]
        w_x     = x.size()[3]
        count_h = (x.size()[2]-1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv    = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x-1, :]), 2).sum()
        w_tv    = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x-1]), 2).sum()
        loss    = self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / b
        return self.weight[0] * loss


@LOSSES.register(name="mae_loss")
@LOSSES.register(name="l1_loss")
class MAELoss(BaseLoss):
    """
    MAE (Mean Absolute Error or L1) loss.
    """
    
    def __init__(
        self,
        weight   : Floats = 1.0,
        reduction: str    = "mean",
        *args, **kwargs
    ):
        super().__init__(
            weight    = weight,
            reduction = reduction,
            *args, **kwargs
        )
        self.name = "mae_loss"
     
    def forward(self, input: Tensors, target: Tensor, **_) -> Tensor:
        loss = F.l1_loss(input=input, target=target, reduction=self.reduction)
        return self.weight[0] * loss


@LOSSES.register(name="mse_loss")
@LOSSES.register(name="l2_loss")
class MSELoss(BaseLoss):
    """
    MSE (Mean Squared Error or L2) loss.
    """
    
    def __init__(
        self,
        weight   : Floats = 1.0,
        reduction: str    = "mean",
        *args, **kwargs
    ):
        super().__init__(
            weight    = weight,
            reduction = reduction,
            *args, **kwargs
        )
        self.name = "mse_loss"
     
    def forward(self, input: Tensors, target: Tensor, **_) -> Tensor:
        loss = F.mse_loss(input=input, target=target, reduction=self.reduction)
        return self.weight[0] * loss


@LOSSES.register(name="non_blurry_loss")
class NonBlurryLoss(BaseLoss):
    """
    MSELoss on the distance to 0.5
    """
    
    def __init__(
        self,
        weight   : Floats = 1.0,
        reduction: str    = "mean",
        *args, **kwargs
    ):
        super().__init__(
            weight    = weight,
            reduction = reduction,
            *args, **kwargs
        )
        self.name = "non_blurry_loss"
     
    def forward(self, input: Tensors, target: None = None, **_) -> Tensor:
        mse_loss = F.mse_loss(
            input     = input,
            target    = torch.ones_like(input) * 0.5,
            reduction = self.reduction
        )
        loss = 1.0 - mse_loss
        return self.weight[0] * loss


@LOSSES.register(name="perceptual_l1_loss")
class PerceptualL1Loss(BaseLoss):
    """
    Loss = weights[0] * Perceptual Loss + weights[1] * L1 Loss.
    """
    
    def __init__(
        self,
        vgg      : nn.Module,
        weight   : Floats = [1.0, 1.0],
        reduction: str    = "mean",
        *args, **kwargs
    ):
        super().__init__(
            weight    = weight,
            reduction = reduction,
            *args, **kwargs
        )
        self.name     = "perceptual_Loss"
        self.per_loss = PerceptualLoss(vgg=vgg, *args, **kwargs)
        self.l1_loss  = L1Loss(*args, **kwargs)
        self.layer_name_mapping = {
            "3" : "relu1_2",
            "8" : "relu2_2",
            "15": "relu3_3"
        }
        
        if self.weight is None:
            self.weight = [1.0, 1.0]
        elif len(self.weight) != 2:
            raise ValueError(
                f"Length of `weight` must be 2. "
                f"But got: {len(self.weight)}."
            )
     
    def forward(self, input: Tensors, target: Tensor, **_) -> Tensor:
        return \
            self.weight[0] * self.per_loss(
                input     = input,
                target    = target,
                eps       = self.eps,
                reduction = self.reduction,
            ) + \
            self.weight[1] * self.l1_loss(
                input     = input,
                target    = target,
                eps       = self.eps,
                reduction = self.reduction,
            )


@LOSSES.register(name="perceptual_loss")
class PerceptualLoss(BaseLoss):
    """
    Perceptual Loss.
    """
    
    def __init__(
        self,
        vgg      : nn.Module,
        weight   : Floats = 1.0,
        reduction: str    = "mean",
        *args, **kwargs
    ):
        super().__init__(
            weight    = weight,
            reduction = reduction,
            *args, **kwargs
        )
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
                    reduction = self.reduction
                )
            )
        loss = reduce_loss(
            loss      = torch.FloatTensor(losses),
            reduction = Reduction.WEIGHTED_SUM
        )
        return self.weight[0] * loss


@LOSSES.register(name="psnr_loss")
class PSNRLoss(BaseLoss):
    """
    PSNR loss.
    
    Modified from BasicSR: https://github.com/xinntao/BasicSR
    """
    
    def __init__(
        self,
        max_val  : float  = 1.0,
        weight   : Floats = 1.0,
        reduction: str    = "mean",
        *args, **kwargs
    ):
        super().__init__(
            weight    = weight,
            reduction = reduction,
            *args, **kwargs
        )
        self.name    = "psnr_loss"
        self.max_val = max_val
     
    def forward(self, input: Tensors, target: Tensor, **_) -> Tensor:
        loss = -1.0 * psnr(input=input, target=target, max_val=self.max_val)
        loss = reduce_loss(loss=loss, reduction=self.reduction)
        return self.weight[0] * loss


@LOSSES.register(name="smooth_mae_loss")
@LOSSES.register(name="smooth_l1_loss")
class SmoothMAELoss(BaseLoss):
    """
    Smooth MAE (Mean Absolute Error or L1) loss.
    """
    
    def __init__(
        self,
        beta     : float  = 1.0,
        weight   : Floats = 1.0,
        reduction: str    = "mean",
        *args, **kwargs
    ):
        super().__init__(
            weight    = weight,
            reduction = reduction,
            *args, **kwargs
        )
        self.name = "smooth_mae_loss"
        self.beta = beta
     
    def forward(self, input: Tensors, target: Tensor, **_) -> Tensor:
        loss = F.smooth_l1_loss(
            input     = input,
            target    = target,
            beta      = self.beta,
            reduction = self.reduction
        )
        return self.weight[0] * loss


@LOSSES.register(name="spatial_consistency_loss")
class SpatialConsistencyLoss(BaseLoss):
    """
    Spatial Consistency Loss (SPA) Loss.
    """
    
    def __init__(
        self,
        weight   : Floats = 1.0,
        reduction: str    = "mean",
        *args, **kwargs
    ):
        super().__init__(
            weight    = weight,
            reduction = reduction,
            *args, **kwargs
        )
        self.name         = "spatial_consistency_loss"
        kernel_left       = FloatTensor([[0,  0, 0], [-1, 1,  0], [0,  0, 0]]).unsqueeze(0).unsqueeze(0)
        kernel_right      = FloatTensor([[0,  0, 0], [ 0, 1, -1], [0,  0, 0]]).unsqueeze(0).unsqueeze(0)
        kernel_up         = FloatTensor([[0, -1, 0], [ 0, 1,  0], [0,  0, 0]]).unsqueeze(0).unsqueeze(0)
        kernel_down       = FloatTensor([[0,  0, 0], [ 0, 1,  0], [0, -1, 0]]).unsqueeze(0).unsqueeze(0)
        self.weight_left  = nn.Parameter(data=kernel_left,  requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up    = nn.Parameter(data=kernel_up,    requires_grad=False)
        self.weight_down  = nn.Parameter(data=kernel_down,  requires_grad=False)
        self.pool         = nn.AvgPool2d(4)
     
    def forward(self, input: Tensors, target: Tensor, **_) -> Tensor:
        if self.weight_left.device != input.device:
            self.weight_left = self.weight_left.to(input.device)
        if self.weight_right.device != input.device:
            self.weight_right = self.weight_right.to(input.device)
        if self.weight_up.device != input.device:
            self.weight_up = self.weight_up.to(input.device)
        if self.weight_down.device != input.device:
            self.weight_down = self.weight_down.to(input.device)

        org_mean     = torch.mean(input,  1, keepdim=True)
        enhance_mean = torch.mean(target, 1, keepdim=True)

        org_pool     = self.pool(org_mean)
        enhance_pool = self.pool(enhance_mean)

        d_org_left      = F.conv2d(org_pool,     self.weight_left,  padding=1)
        d_org_right     = F.conv2d(org_pool,     self.weight_right, padding=1)
        d_org_up        = F.conv2d(org_pool,     self.weight_up,    padding=1)
        d_org_down      = F.conv2d(org_pool,     self.weight_down,  padding=1)
    
        d_enhance_left  = F.conv2d(enhance_pool, self.weight_left,  padding=1)
        d_enhance_right = F.conv2d(enhance_pool, self.weight_right, padding=1)
        d_enhance_up    = F.conv2d(enhance_pool, self.weight_up,    padding=1)
        d_enhance_down  = F.conv2d(enhance_pool, self.weight_down,  padding=1)
    
        d_left          = torch.pow(d_org_left  - d_enhance_left,  2)
        d_right         = torch.pow(d_org_right - d_enhance_right, 2)
        d_up            = torch.pow(d_org_up    - d_enhance_up,    2)
        d_down          = torch.pow(d_org_down  - d_enhance_down,  2)
        loss            = d_left + d_right + d_up + d_down
        loss            = reduce_loss(loss=loss, reduction=self.reduction)
        return self.weight[0] * loss
    

@LOSSES.register(name="ssim_loss")
class SSIMLoss(BaseLoss):
    """
    SSIM Loss.
    
    Modified from BasicSR: https://github.com/xinntao/BasicSR
    
    Args:
        window_size (int): Size of the gaussian kernel to smooth the images.
        max_val (float): Dynamic range of the images. Defaults to 1.0.
        eps (float): Small value for numerically stability when dividing.
            Defaults to 1e-12.
    """
    
    def __init__(
        self,
        window_size: int,
        max_val  : float  = 1.0,
        eps      : float  = 1e-12,
        weight   : Floats = 1.0,
        reduction: str    = "mean",
        *args, **kwargs
    ):
        super().__init__(
            weight    = weight,
            reduction = reduction,
            *args, **kwargs
        )
        self.name        = "ssim_loss"
        self.window_size = window_size
        self.max_val     = max_val
        self.eps         = eps
    
    def forward(self, input: Tensors, target: Tensor, **_) -> Tensor:
        # Compute the ssim map
        ssim_map = ssim(
            input       = input,
            target      = target,
            window_size = self.window_size,
            max_val     = self.max_val,
            eps         = self.eps
        )
        # Compute and reduce the loss
        loss = torch.clamp((1.0 - ssim_map) / 2, min=0, max=1)
        loss = reduce_loss(loss=loss, reduction=self.reduction)
        return self.weight[0] * loss


@LOSSES.register(name="std_loss")
class StdLoss(BaseLoss):
    """
    Loss on the variance of the image. Works in the grayscale. If the image is
    smooth, gets zero.
    """
    
    def __init__(
        self,
        weight   : Floats = 1.0,
        reduction: str    = "mean",
        *args, **kwargs
    ):
        super().__init__(
            weight    = weight,
            reduction = reduction,
            *args, **kwargs
        )
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
                    reduction = self.reduction
                )
            )
        return self.weight[0] * reduce_loss(
            loss      = torch.FloatTensor(losses),
            reduction = Reduction.WEIGHTED_SUM
        )


L1Loss       = MAELoss
L2Loss       = MSELoss
SmoothL1Loss = SmoothMAELoss

LOSSES.register(name="bce_loss", 			              module=BCELoss)
LOSSES.register(name="bce_with_logits_loss",              module=BCEWithLogitsLoss)
LOSSES.register(name="cosine_embedding_loss",             module=CosineEmbeddingLoss)
LOSSES.register(name="cross_entropy_loss",	              module=CrossEntropyLoss)
LOSSES.register(name="ctc_loss",	  		              module=CTCLoss)
LOSSES.register(name="gaussian_nll_loss",                 module=GaussianNLLLoss)
LOSSES.register(name="hinge_embedding_loss",              module=HingeEmbeddingLoss)
LOSSES.register(name="huber_loss",                        module=HuberLoss)
LOSSES.register(name="kl_div_loss",	  		              module=KLDivLoss)
LOSSES.register(name="margin_ranking_loss",	              module=MarginRankingLoss)
LOSSES.register(name="multi_label_margin_loss",           module=MultiLabelMarginLoss)
LOSSES.register(name="multi_label_soft_margin_loss",      module=MultiLabelSoftMarginLoss)
LOSSES.register(name="multi_margin_loss",                 module=MultiMarginLoss)
LOSSES.register(name="nll_loss",   	   		              module=NLLLoss)
LOSSES.register(name="nll_loss2d",   	   		          module=NLLLoss2d)
LOSSES.register(name="poisson_nll_loss",   	              module=PoissonNLLLoss)
LOSSES.register(name="soft_margin_loss",   	              module=SoftMarginLoss)
LOSSES.register(name="triplet_margin_loss",               module=TripletMarginLoss)
LOSSES.register(name="triplet_margin_with_distance_Loss", module=TripletMarginWithDistanceLoss)

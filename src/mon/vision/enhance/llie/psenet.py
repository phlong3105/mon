#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements PSENet (PSENet: Progressive Self-Enhancement Network
for Unsupervised Extreme-Light Image Enhancement) models.

Reference:
    `<https://github.com/VinAIResearch/PSENet-Image-Enhancement>`__
"""

from __future__ import annotations

__all__ = [
    "PSENet",
]
from typing import Any, Literal

import lightning.pytorch.utilities.types
import torch

from mon import core, nn
from mon.core import _callable, _size_2_t
from mon.globals import MODELS, Scheme
from mon.nn import functional as F
from mon.vision.enhance.llie import base

console     = core.console
StepOutput  = lightning.pytorch.utilities.types.STEP_OUTPUT
EpochOutput = Any  # lightning.pytorch.utilities.types.EPOCH_OUTPUT


# region Loss

class Loss(nn.Loss):
    
    def __init__(
        self,
        tv_weight: float = 5.0,
        reduction: Literal["none", "mean", "sum"] = "mean",
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs, reduction=reduction)
        self.tv_weight = tv_weight
        self.l2_loss   = nn.L2Loss(reduction=reduction)
    
    def forward(
        self,
        input : torch.Tensor,
        target: torch.Tensor,
        gamma : torch.Tensor,
        *_
    ) -> torch.Tensor:
        rec_loss = self.l2_loss(input, target)
        tv_loss  = self.tv_loss(gamma)
        loss     = rec_loss + tv_loss * self.tv_weight
        return loss
    
    @staticmethod
    def tv_loss(input: torch.Tensor) -> torch.Tensor:
        x    = input
        x    = torch.log(x + 1e-3)
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :-1, :]), 2)
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :-1]), 2)
        return torch.mean(h_tv) + torch.mean(w_tv)
    
# endregion


# region Module

class IQA(nn.Module):
    
    def __init__(self):
        super().__init__()
        ps = 25
        self.exposed_level = 0.5
        self.mean_pool     = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(ps // 2),
            torch.nn.AvgPool2d(ps, stride=1)
        )
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        eps         = 1 / 255.0
        max_rgb     = torch.max(input, dim=1, keepdim=True)[0]
        min_rgb     = torch.min(input, dim=1, keepdim=True)[0]
        saturation  = (max_rgb - min_rgb + eps) / (max_rgb + eps)
        
        mean_rgb    = self.mean_pool(input).mean(dim=1, keepdim=True)
        exposedness = torch.abs(mean_rgb - self.exposed_level) + eps
        
        contrast    = self.mean_pool(input * input).mean(dim=1, keepdim=True) - mean_rgb ** 2
        return torch.mean((saturation * contrast) / exposedness, dim=[1], keepdim=True)

    
class MobileBottleneck(nn.Module):
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        mid_channels: int,
        kernel_size : _size_2_t,
        stride      : _size_2_t,
        se          : str | bool = "SE",
        nl: Literal["ReLU", "Hardswish", "LeakyReLU", "Hardsigmoid", "NegHardsigmoid"] = "ReLU"
    ):
        super().__init__()
        assert stride in [1, 2]
        # assert kernel in [3, 5, 7]
        
        padding = (kernel_size - 1) // 2
        self.with_residual = stride == 1 and in_channels == out_channels
        # self.use_res_connect = False
        
        conv_layer = nn.Conv2d
        if nl == "ReLU":
            nonlinear_layer = nn.ReLU  # or ReLU6
        elif nl == "Hardswish":
            nonlinear_layer = nn.Hardswish
        elif nl == "LeakyReLU":
            nonlinear_layer = nn.LeakyReLU
        elif nl == "Hardsigmoid":
            nonlinear_layer = nn.Hardsigmoid
        elif nl == "NegHardsigmoid":
            nonlinear_layer = nn.NegHardsigmoid
        else:
            raise NotImplementedError
        
        if se == "SE":
            se_layer = nn.SqueezeExciteC
        else:
            se_layer = nn.Identity
        
        if mid_channels != out_channels:
            self.conv = nn.Sequential(
                # PW
                conv_layer(in_channels, mid_channels, 1, 1, 0, bias=True, padding_mode="reflect"),
                nonlinear_layer(inplace=True),
                # DW
                conv_layer(mid_channels, mid_channels, kernel_size, stride=stride, padding=padding, groups=mid_channels, bias=True, padding_mode="reflect"),
                se_layer(mid_channels, reduction_ratio=1, bias=True),
                nonlinear_layer(inplace=True),
                # PW Linear
                conv_layer(mid_channels, out_channels, 1, 1, 0, bias=True, padding_mode="reflect"),
            )
        else:
            self.conv = nn.Sequential(
                # PW
                conv_layer(in_channels, mid_channels, 1, 1, 0, bias=True),
                nonlinear_layer(inplace=False),
                conv_layer(mid_channels, out_channels, 1, 1, 0, bias=True),
            )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        if self.with_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)


class UnetTMO(nn.Module):
    
    def __init__(self, in_channels: int = 3, out_channels: int = 3):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        base_number       = 16
        self.first_conv   = MobileBottleneck(self.in_channels, self.out_channels, 6,          3, 1, "SE",  "LeakyReLU")
        self.conv1        = MobileBottleneck(self.in_channels, base_number,       int(base_number * 1.5), 3, 2, False, "LeakyReLU")
        self.conv2        = MobileBottleneck(base_number,      base_number,       int(base_number * 1.5), 3, 1, False, "LeakyReLU")
        self.conv3        = MobileBottleneck(base_number,      base_number * 2,   base_number * 3,        3, 2, False, "LeakyReLU")
        self.conv5        = MobileBottleneck(base_number * 2,  base_number * 2,   base_number * 3,        3, 1, False, "LeakyReLU")
        self.conv6        = MobileBottleneck(base_number * 2,  base_number,       base_number * 3,        3, 1, False, "LeakyReLU")
        self.conv7        = MobileBottleneck(base_number * 2,  base_number,       base_number * 3,        3, 1, False, "LeakyReLU")
        self.conv8        = MobileBottleneck(base_number,      self.out_channels, int(base_number * 1.5), 3, 1, False, "LeakyReLU")
        self.last_conv    = MobileBottleneck(6,     self.out_channels, 9,          3, 1, "SE",  "LeakyReLU")
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x      = input
        x_down = x
        x_1    = self.first_conv(x)
        r      = self.conv1(x_1)
        r      = self.conv2(r)
        r_d2   = r
        r      = self.conv3(r)
        r      = self.conv5(r)
        r      = self.conv6(r)
        r      = F.interpolate(r, (r_d2.shape[2], r_d2.shape[3]), mode="bilinear", align_corners=True)
        r      = self.conv7(torch.cat([r_d2, r], dim=1))
        r      = self.conv8(r)
        r      = F.interpolate(r, (x_down.shape[2], x_down.shape[3]), mode="bilinear", align_corners=True)
        r      = self.last_conv(torch.cat([x_1, r], dim=1))
        r      = torch.abs(r + 1)
        y      = 1 - (1 - x) ** r
        return r, y

# endregion


# region Model

@MODELS.register(name="psenet")
class PSENet(base.LowLightImageEnhancementModel):
    """PSENet (PSENet: Progressive Self-Enhancement Network for Unsupervised
    Extreme-Light Image Enhancement) models.
    
    Reference:
        `<https://github.com/VinAIResearch/PSENet-Image-Enhancement>`__
     
    See Also: :class:`base.LowLightImageEnhancementModel`
    """
    
    _scheme: list[Scheme] = [Scheme.UNSUPERVISED, Scheme.ZEROSHOT]
    _zoo   : dict = {}
    
    def __init__(
        self,
        in_channels     : int   = 3,
        gamma_lower     : int   = -2,
        gamma_upper     : int   = 3,
        number_refs     : int   = 1,
        afifi_evaluation: bool  = False,
        tv_weight       : float = 5.0,
        weights         : Any   = None,
        *args, **kwargs
    ):
        super().__init__(
            name        = "psenet",
            in_channels = in_channels,
            weights     = weights,
            *args, **kwargs
        )
        
        # Populate hyperparameter values from pretrained weights
        if isinstance(self.weights, dict):
            in_channels      = self.weights.get("in_channels"     , in_channels)
            gamma_lower      = self.weights.get("gamma_lower"     , gamma_lower)
            gamma_upper      = self.weights.get("gamma_upper"     , gamma_upper)
            number_refs      = self.weights.get("number_refs"     , number_refs)
            afifi_evaluation = self.weights.get("afifi_evaluation", afifi_evaluation)
            tv_weight        = self.weights.get("tv_weight"       , tv_weight)
        self.in_channels      = in_channels
        self.gamma_lower      = gamma_lower
        self.gamma_upper      = gamma_upper
        self.number_refs      = number_refs
        self.afifi_evaluation = afifi_evaluation
        self.prev_input       = None
        self.prev_pseudo_gt   = None
        
        # Construct model
        self.model = UnetTMO()
        self.iqa   = IQA()
        
        # Loss
        self._loss = Loss(tv_weight=tv_weight)
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        pass
    
    # region Forward Pass
    
    def forward_loss(
        self,
        input : torch.Tensor,
        target: torch.Tensor | None,
        *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        pred           = self.forward(input=input, *args, **kwargs)
        gamma, enhance = pred
        loss           = self.loss(enhance, target, gamma)
        return enhance, loss

    def forward(
        self,
        input    : torch.Tensor,
        augment  : _callable = None,
        profile  : bool      = False,
        out_index: int       = -1,
        *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x    = input
        r, y = self.model(x)
        return r, y
    
    def generate_pseudo_gt(self, input: torch.Tensor) -> torch.Tensor:
        b, c, h, w = input.shape
        #
        underexposed_ranges  = torch.linspace(0, self.gamma_upper, steps=self.number_refs + 1).to(input.device)[:-1]
        step_size            = self.gamma_upper / self.number_refs
        underexposed_gamma   = torch.exp(torch.rand([b, self.number_refs], device=input.device) * step_size + underexposed_ranges[None, :])
        #
        overrexposed_ranges  = torch.linspace(self.gamma_lower, 0, steps=self.number_refs + 1).to(input.device)[:-1]
        step_size            = - self.gamma_lower / self.number_refs
        overrexposed_gamma   = torch.exp(torch.rand([b, self.number_refs], device=input.device) * overrexposed_ranges[None, :])
        #
        gammas               = torch.cat([underexposed_gamma, overrexposed_gamma], dim=1)
        # gammas: [b, nref], im: [b, c, h, w] -> synthetic_references: [b, nref, c, h, w]
        synthetic_references = 1 - (1 - input[:, None]) ** gammas[:, :, None, None, None]
        previous_iter_output = self.model(input)[-1].clone().detach()
        references           = torch.cat([input[:, None], previous_iter_output[:, None], synthetic_references], dim=1)
        nref                 = references.shape[1]
        scores               = self.iqa(references.view(b * nref, c, h, w))
        scores               = scores.view(b, nref, 1, h, w)
        max_idx              = torch.argmax(scores, dim=1)
        max_idx              = max_idx.repeat(1, c, 1, 1)[:, None]
        pseudo_gt            = torch.gather(references, 1, max_idx)
        return pseudo_gt.squeeze(1)
    
    # endregion
    
    # region Training
    
    def training_step(self, batch: Any, batch_idx: int, *args, **kwargs) -> StepOutput | None:
        """Here you compute and return the training loss, and some additional
        metrics for e.g., the progress bar or logger.

        Args:
            batch: The output of :class:`~torch.utils.data.DataLoader`. It can
                be a :class:`torch.Tensor`, :class:`tuple` or :class:`list`.
            batch_idx: An integer displaying index of this batch.
            
        Return:
            Any of:
                - The loss tensor.
                - A :class:`dict`. Can include any keys, but must include the
                  key ``'loss'``.
                - ``None``, training will skip to the next batch.
        """
        # input, target, extra = batch[0], batch[1], batch[2:]
        
        # Saving n-th input and n-th pseudo gt
        nth_input     = batch[0]
        nth_pseudo_gt = self.generate_pseudo_gt(batch[0])
        
        # Forward pass
        if self.prev_input is not None:
            # Getting (n - 1)th input and (n - 1)-th pseudo gt -> calculate loss -> update model weight (handled automatically by pytorch lightning)
            input      = self.prev_input
            pseudo_gt  = self.prev_pseudo_gt
            pred, loss = self.forward_loss(input=input, target=pseudo_gt, *args, **kwargs)
            # Log
            log_dict = {
                f"step"      : self.current_epoch,
                f"train/loss": loss,
            }
            self.log_dict(
                dictionary     = log_dict,
                prog_bar       = False,
                logger         = True,
                on_step        = False,
                on_epoch       = True,
                sync_dist      = True,
                rank_zero_only = False,
            )
        else:
            # Skip updating model's weight at the first batch
            loss = None
           
        # Saving n-th input and n-th pseudo gt
        self.prev_input     = nth_input
        self.prev_pseudo_gt = nth_pseudo_gt
        
        return loss
    
    '''
    def on_train_epoch_end(self):
        """Called in the training loop at the very end of the epoch."""
        lr_scheduler = self.lr_schedulers()
        if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            lr_scheduler.step(self.trainer.callback_metrics["train/loss"])
            
        if self.train_metrics:
            for i, metric in enumerate(self.train_metrics):
                metric.reset()
    '''
    
    # endregion
    
# endregion

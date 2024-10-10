#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""PSENet

This module implements the paper: "PSENet: Progressive Self-Enhancement Network
for Unsupervised Extreme-Light Image Enhancement".

References:
    https://github.com/whai362/PSENet
"""

from __future__ import annotations

__all__ = [
    "PSENet",
]

from typing import Any, Literal

import torch

from mon import core, nn
from mon.globals import MODELS, Scheme, Task
from mon.nn import functional as F
from mon.vision.enhance import base, utils

console      = core.console
current_file = core.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# region Loss

class TVLoss(nn.Loss):

    def __init__(
        self,
        loss_weight: float = 1.0,
        reduction  : Literal["none", "mean", "sum"] = "mean",
    ):
        super().__init__(loss_weight=loss_weight, reduction=reduction)
        
    def forward(self, input: torch.Tensor, target: torch.Tensor = None) -> torch.Tensor:
        x    = input
        x    = torch.log(x + 1e-3)
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :-1, :]), 2)
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :-1]), 2)
        loss = torch.mean(h_tv) + torch.mean(w_tv)
        loss = loss * self.loss_weight
        return loss
        
# endregion


# region Module

class Hswish(nn.Module):
    
    def __init__(self, inplace: bool = True):
        super().__init__()
        self.inplace = inplace

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input * F.relu6(input + 3.0, inplace=self.inplace) / 6.0


class Hsigmoid(nn.Module):
    
    def __init__(self, inplace: bool = True):
        super().__init__()
        self.inplace = inplace

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.relu6(3 * input + 3.0, inplace=self.inplace) / 6.0


class HTanh(nn.Module):
    
    def __init__(self, inplace: bool = True):
        super().__init__()
        self.inplace = inplace

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.relu6(input, inplace=self.inplace) / 3.0 - 1.0


class NegHsigmoid(nn.Module):
    
    def __init__(self, inplace: bool = True):
        super().__init__()
        self.inplace = inplace

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.relu6(3 * input + 3.0, inplace=self.inplace) / 6.0 - 0.5


class SEModule(nn.Module):
    
    def __init__(self, channel: int, reduction: int = 1):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc       = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, 1, 0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, 1, 0, bias=True),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = self.avg_pool(input)
        y = self.fc(y)
        return x * y


class MobileBottleneck(nn.Module):
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel      : int,
        stride      : int,
        exp         : int,
        se          : str = "SE",
        nl          : str = "RE"
    ):
        super().__init__()
        assert stride in [1, 2]
        # assert kernel in [3, 5, 7]
        padding = (kernel - 1) // 2
        self.use_res_connect = stride == 1 and in_channels == out_channels
        # self.use_res_connect = False
        conv_layer = nn.Conv2d
        if nl == "RE":
            nlin_layer = nn.ReLU  # or ReLU6
        elif nl == "HS":
            nlin_layer = Hswish
        elif nl == "LeRE":
            nlin_layer = nn.LeakyReLU
        elif nl == "HSig":
            nlin_layer = Hsigmoid
        elif nl == "NegHSig":
            nlin_layer = NegHsigmoid
        else:
            raise NotImplementedError
        if se == "SE":
            SELayer = SEModule
        else:
            SELayer = nn.Identity
        if exp != out_channels:
            self.conv = nn.Sequential(
                # pw
                conv_layer(in_channels, exp, 1, 1, 0, bias=True, padding_mode="reflect"),
                nlin_layer(inplace=True),
                # dw
                conv_layer(exp, exp, kernel, stride=stride, padding=padding, groups=exp, bias=True, padding_mode="reflect"),
                SELayer(exp),
                nlin_layer(inplace=True),
                # pw-linear
                conv_layer(exp, out_channels, 1, 1, 0, bias=True, padding_mode="reflect"),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                conv_layer(in_channels, exp, 1, 1, 0, bias=True),
                nlin_layer(inplace=False),
                conv_layer(exp, out_channels, 1, 1, 0, bias=True),
            )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.use_res_connect:
            return input + self.conv(input)
        else:
            return self.conv(input)


class UnetTMO(nn.Module):
    
    def __init__(
        self,
        in_channels : int = 3,
        out_channels: int = 3,
        base_number : int = 16,
    ):
        super().__init__()
        self.first_conv = MobileBottleneck(in_channels,     out_channels,    3, 1, 6, nl="LeRE")
        self.conv1      = MobileBottleneck(in_channels,     base_number,     3, 2, int(base_number * 1.5), False, "LeRE")
        self.conv2      = MobileBottleneck(base_number,     base_number,     3, 1, int(base_number * 1.5), False, "LeRE")
        self.conv3      = MobileBottleneck(base_number,     base_number * 2, 3, 2, base_number * 3,        False, "LeRE")
        self.conv5      = MobileBottleneck(base_number * 2, base_number * 2, 3, 1, base_number * 3,        False, "LeRE")
        self.conv6      = MobileBottleneck(base_number * 2, base_number,     3, 1, base_number * 3,        False, "LeRE")
        self.conv7      = MobileBottleneck(base_number * 2, base_number,     3, 1, base_number * 3,        False, "LeRE")
        self.conv8      = MobileBottleneck(base_number,     out_channels,    3, 1, int(base_number * 1.5), False, "LeRE")
        self.last_conv  = MobileBottleneck(in_channels * 2, out_channels,    3, 1, 9, nl="LeRE")

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
        x      = 1 - (1 - x) ** r
        return x, r
    
# endregion


# region Model

@MODELS.register(name="psenet", arch="psenet")
class PSENet(base.ImageEnhancementModel):
    """PSENet: Progressive Self-Enhancement Network for Unsupervised
    Extreme-Light Image Enhancement.
    
    References:
        https://github.com/whai362/PSENet
    """
    
    model_dir: core.Path    = current_dir
    arch     : str          = "psenet"
    tasks    : list[Task]   = [Task.LLIE]
    schemes  : list[Scheme] = [Scheme.UNSUPERVISED, Scheme.ZERO_REFERENCE]
    zoo      : dict         = {}
    
    def __init__(
        self,
        in_channels     : int   = 3,
        out_channels    : int   = 3,
        base_channels   : int   = 16,
        tv_weight       : float = 5,
        gamma_lower     : float = -2,
        gamma_upper     : float = 3,
        number_refs     : int   = 1,
        lr              : float = 5e-4,
        afifi_evaluation: bool  = False,
        weights         : Any   = None,
        *args, **kwargs
    ):
        super().__init__(
            name         = "psenet",
            in_channels  = in_channels,
            out_channels = out_channels,
            weights      = weights,
            *args, **kwargs
        )
        self.in_channels      = in_channels
        self.out_channels     = out_channels
        self.base_channels    = base_channels
        self.tv_weight        = tv_weight
        self.gamma_lower      = gamma_lower
        self.gamma_upper      = gamma_upper
        self.number_refs      = number_refs
        self.lr               = lr
        self.afifi_evaluation = afifi_evaluation
        self.saved_input      = None
        self.saved_pseudo_gt  = None
        
        # Construct model
        self.model = UnetTMO(self.in_channels, self.out_channels, self.base_channels)
        
        # Loss
        self.mse  = nn.MSELoss()
        self.loss = TVLoss(reduction="mean")
        
        self.pseudo_gt_generator = utils.PseudoGTGenerator(
            number_refs   = self.number_refs,
            gamma_upper   = self.gamma_upper,
            gamma_lower   = self.gamma_lower,
            exposed_level = 0.5,
            pool_size     = 25,
        )
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)

    def init_weights(self, m: nn.Module):
        pass
    
    def configure_optimizers(self):
        optimizer = nn.Adam(self.model.parameters(), lr=self.lr, betas=[0.9, 0.99])
        scheduler = nn.ReduceLROnPlateau(optimizer, patience=10, factor=0.5, verbose=True)
        return {
            "optimizer"   : optimizer,
            "lr_scheduler": scheduler,
            "monitor"     : "train/loss",
        }
    
    def forward_loss(self, datapoint: dict, *args, **kwargs) -> dict | None:
        # Forward
        self.assert_datapoint(datapoint)
        image = datapoint.get("image")
        # Saving n-th input and n-th pseudo gt
        nth_input     = image
        nth_output    = self.model(image)[0].clone().detach()
        nth_pseudo_gt = self.pseudo_gt_generator(image, nth_output)
        if self.saved_input is not None:
            # Getting (n - 1)th input and (n - 1)-th pseudo gt -> calculate loss -> update model weight (handled automatically by pytorch lightning)
            x          = self.saved_input
            y, r       = self.model(x)
            pseudo_gt  = self.saved_pseudo_gt
            recon_loss = self.mse(y, pseudo_gt)
            tv_loss    = self.loss(r)
            loss       = recon_loss + tv_loss * self.tv_weight
            outputs = {
                "adjust"  : r,
                "enhanced": y,
                "loss"    : loss,
            }
        else:  # Skip updating model's weight at the first batch
            outputs = {"loss": None}
        # Saving n-th input and n-th pseudo gt
        self.saved_input     = nth_input
        self.saved_pseudo_gt = nth_pseudo_gt
        # Return
        return outputs
    
    def forward(self, datapoint: dict, *args, **kwargs) -> dict:
        self.assert_datapoint(datapoint)
        x    = datapoint.get("image")
        y, r = self.model(x)
        return {
            "adjust"  : r,
            "enhanced": y,
        }
    
    def on_train_epoch_end(self):
        sch = self.lr_schedulers()
        if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
            sch.step(self.trainer.callback_metrics["train/loss"])
        super().on_train_epoch_end()
        
# endregion

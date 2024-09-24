#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Zero-Restore

This module implements the paper: Zero-shot Single Image Restoration through
Controlled Perturbation of Koschmieder's Model.

References:
    https://github.com/aupendu/zero-restore
"""

from __future__ import annotations

__all__ = [
    "ZeroRestoreLLIE",
]

import random
from typing import Any, Literal

import numpy as np
import torch

from mon import core, nn
from mon.globals import MODELS, Scheme, Task
from mon.vision.enhance import base

torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
np.random.seed(1)
random.seed(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

console      = core.console
current_file = core.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# region Loss

class TotalVariationLoss(nn.Loss):
    """Total Variation Loss on the Illumination (Illumination Smoothness Loss)
    `\mathcal{L}_{tvA}` preserve the monotonicity relations between
    neighboring pixels. It is used to avoid aggressive and sharp changes between
    neighboring pixels.
    
    References:
        `<https://github.com/Li-Chongyi/Zero-DCE/blob/master/Zero-DCE_code/Myloss.py>`__
    """
    
    def __init__(
        self,
        loss_weight: float = 1.0,
        reduction  : Literal["none", "mean", "sum"] = "mean",
    ):
        super().__init__(loss_weight=loss_weight, reduction=reduction)
    
    def forward(
        self,
        input : torch.Tensor,
        target: torch.Tensor = None
    ) -> torch.Tensor:
        x       = input
        b       = x.size()[0]
        h_x     = x.size()[2]
        w_x     = x.size()[3]
        count_h = (h_x-1) * w_x
        count_w = h_x * (w_x - 1)
        h_tv    = torch.pow((x[:, :, 1:,  :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv    = torch.pow((x[:, :,  :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        loss    = (h_tv / count_h + w_tv / count_w) / b
        loss    = self.loss_weight * loss
        return loss

# endregion


# region Module

class DoubleConv(nn.Module):
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False, padding_mode="reflect"),
            nn.GroupNorm(num_channels=out_channels, num_groups=8, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False, padding_mode="reflect"),
            nn.GroupNorm(num_channels=out_channels, num_groups=8, affine=True),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.conv(input)


class InDoubleConv(nn.Module):
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False, padding_mode='reflect'),
            nn.GroupNorm(num_channels=out_channels, num_groups=8, affine=True),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.conv(input)


class InConv(nn.Module):
    
    def __init__(self, in_channels: int = 1, out_channels: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 7, stride=4, padding=3, bias=False, padding_mode="reflect"),
            nn.GroupNorm(num_channels=out_channels, num_groups=8, affine=True),
            nn.ReLU(inplace=True)
        )
        self.convf = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False, padding_mode="reflect"),
            nn.GroupNorm(num_channels=out_channels, num_groups=8, affine=True),
            nn.ReLU(inplace=False)
        )
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x    = input
        R    = x[:, 0:1, :, :]
        G    = x[:, 1:2, :, :]
        B    = x[:, 2:3, :, :]
        xR   = torch.unsqueeze(self.conv(R), 1)
        xG   = torch.unsqueeze(self.conv(G), 1)
        xB   = torch.unsqueeze(self.conv(B), 1)
        x    = torch.cat([xR, xG, xB], 1)
        x, _ = torch.max(x, dim=1)
        return self.convf(x)


class SKConv(nn.Module):
    
    def __init__(
        self,
        in_channels : int = 1,
        out_channels: int = 64,
        M           : int = 4,
        L           : int = 32
    ):
        super().__init__()
        self.M     = M
        self.convs = nn.ModuleList([])
        in_conv    = InConv(in_channels, out_channels)
        for i in range(M):
            if i == 0:
                self.convs.append(in_conv)
            else:
                self.convs.append(
                    nn.Sequential(
                        nn.Upsample(scale_factor=1 / (2 ** i), mode="bilinear", align_corners=True),
                        in_conv,
                        nn.Upsample(scale_factor=2 ** i, mode="bilinear", align_corners=True)
                    )
                )
        self.fc  = nn.Linear(out_channels, L)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(nn.Linear(L, out_channels))
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        for i, conv in enumerate(self.convs):
            fea = conv(x)
            fea = torch.unsqueeze(fea, 1)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)
        fea_u = torch.sum(feas, dim=1)
        fea_s = fea_u.mean(-1).mean(-1)
        fea_z = self.fc(fea_s)
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z)
            vector = torch.unsqueeze(vector, 1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = torch.unsqueeze(attention_vectors, -1)
        attention_vectors = torch.unsqueeze(attention_vectors, -1)
        fea_v = (feas * attention_vectors).sum(dim=1)
        return fea_v


class Estimation(nn.Module):

    def __init__(self, num_channels: int = 64):
        super().__init__()
        self.num_channels = num_channels
        self.in_conv      = SKConv(1, num_channels, 3, 32)
        # Transmission Map
        self.conv_t1 = DoubleConv(num_channels, num_channels)
        self.conv_t2 = nn.Conv2d(num_channels, 1, 3, padding=1, stride=1, bias=False, padding_mode="reflect")
        self.up      = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
        # Atmospheric Light
        self.conv_a1 = InDoubleConv(3, num_channels)
        self.conv_a2 = DoubleConv(num_channels, num_channels)
        self.conv_a3 = nn.Conv2d(num_channels, 1, 3, padding=1, stride=1, bias=False, padding_mode="reflect")
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x     = input
        x_min = self.in_conv(x)
        trans = self.conv_t2(self.up(self.conv_t1(x_min)))
        trans = torch.sigmoid(trans) + 1e-12
        atm   = self.conv_a1(x)
        atm   = torch.mul(atm, self.up(x_min))
        atm   = self.conv_a3(self.conv_a2(atm))
        atm   = torch.sigmoid(atm)
        return trans, atm

# endregion


# region Model

@MODELS.register(name="zero_restore_llie", arch="zero_restore")
class ZeroRestoreLLIE(base.ImageEnhancementModel):
    """Zero-shot Single Image Restoration through Controlled Perturbation of
    Koschmieder's Model.
    """
    
    model_dir: core.Path    = current_dir
    arch     : str          = "zero_restore"
    tasks    : list[Task]   = [Task.LLIE]
    schemes  : list[Scheme] = [Scheme.ZERO_REFERENCE]
    zoo      : dict         = {}
    
    def __init__(
        self,
        name        : str = "zero_restore_llie",
        in_channels : int = 3,
        num_channels: int = 64,
        weights     : Any = None,
        *args, **kwargs
    ):
        super().__init__(
            name        = name,
            in_channels = in_channels,
            weights     = weights,
            *args, **kwargs
        )
        self.in_channels  = in_channels
        self.num_channels = num_channels
        
        # Construct model
        self.estimation = Estimation(self.num_channels)
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)
        self.initial_state_dict = self.state_dict()
        
    def init_weights(self, m: nn.Module):
        classname = m.__class__.__name__
        if classname.find("Conv2d") != -1:  # 0.02
            m.weight.data.normal_(0.0, 0.001)
        if classname.find("Linear") != -1:  # 0.02
            m.weight.data.normal_(0.0, 0.001)
    
    def forward_loss(self, datapoint: dict, *args, **kwargs) -> dict:
        # Forward 1
        self.assert_datapoint(datapoint)
        outputs   = self.forward(datapoint=datapoint, *args, **kwargs)
        image     = datapoint.get("image")
        trans_map = outputs["trans"]
        atm_map   = outputs["atm"]
        enhanced  = outputs["enhanced"]
        # Forward 2
        p_x         = 0.9
        image_x     = image * p_x + (1 - p_x) * atm_map
        outputs_x   = self.forward(datapoint={"image": image_x}, *args, **kwargs)
        trans_map_x = outputs_x["trans"]
        atm_map_x   = outputs_x["atm"]
        enhanced_x  = outputs_x["enhanced"]
        # Loss
        o_tensor  = torch.ones(enhanced.shape).to(self.device)
        z_tensor  = torch.zeros(enhanced.shape).to(self.device)
        loss_t    = torch.sum((trans_map_x - p_x * trans_map) ** 2)
        loss_a    = torch.sum((atm_map - atm_map_x) ** 2)
        loss_mx_r = torch.sum(torch.max(enhanced[:, 0:1, :, :], o_tensor)) + torch.sum(torch.max(enhanced_x[:, 0:1, :, :], o_tensor)) - 2 * torch.sum(o_tensor)
        loss_mx_g = torch.sum(torch.max(enhanced[:, 1:2, :, :], o_tensor)) + torch.sum(torch.max(enhanced_x[:, 1:2, :, :], o_tensor)) - 2 * torch.sum(o_tensor)
        loss_mx_b = torch.sum(torch.max(enhanced[:, 2:3, :, :], o_tensor)) + torch.sum(torch.max(enhanced_x[:, 2:3, :, :], o_tensor)) - 2 * torch.sum(o_tensor)
        loss_mx   = loss_mx_r + loss_mx_g + 10 * loss_mx_b
        loss_mn_r = -torch.sum(torch.min(enhanced[:, 0:1, :, :], z_tensor)) - torch.sum(torch.min(enhanced_x[:, 0:1, :, :], z_tensor))
        loss_mn_g = -torch.sum(torch.min(enhanced[:, 1:2, :, :], z_tensor)) - torch.sum(torch.min(enhanced_x[:, 1:2, :, :], z_tensor))
        loss_mn_b = -torch.sum(torch.min(enhanced[:, 2:3, :, :], z_tensor)) - torch.sum(torch.min(enhanced_x[:, 2:3, :, :], z_tensor))
        loss_mn   = loss_mn_r + loss_mn_g + 10 * loss_mn_b
        loss_tv   = nn.TotalVariationLoss()(enhanced)
        loss      = loss_t + loss_a + 0.001 * loss_mx + 0.01 * loss_mn + 0.001 * loss_tv
        outputs["loss"] = loss
        # Return
        return outputs
        
    def forward(self, datapoint: dict, *args, **kwargs) -> dict:
        # Prepare input
        self.assert_datapoint(datapoint)
        image = datapoint.get("image")
        # Forward
        trans, atm = self.estimation(image)
        atm        = atm.expand_as(image)
        trans      = trans.expand_as(image)
        enhanced   = (image - (1 - trans) * atm) / trans
        # Return
        return {
            "trans"   : trans,
            "atm"     : atm,
            "enhanced": enhanced,
        }
    
    def augment(self, image: torch.Tensor) -> torch.Tensor:
        it = random.randint(0, 7)
        if it == 1:
            image = image.rot90(1, [2, 3])
        if it == 2:
            image = image.rot90(2, [2, 3])
        if it == 3:
            image = image.rot90(3, [2, 3])
        if it == 4:
            image = image.flip(2).rot90(1, [2, 3])
        if it == 5:
            image = image.flip(3).rot90(1, [2, 3])
        if it == 6:
            image = image.flip(2)
        if it == 7:
            image = image.flip(3)
        return image
    
    def infer(
        self,
        datapoint    : dict,
        epochs       : int   = 1000,
        lr           : float = 1e-3,
        weight_decay : float = 1e-2,
        reset_weights: bool  = True,
        *args, **kwargs
    ) -> dict:
        # Initialize training components
        self.train()
        if reset_weights:
            self.load_state_dict(self.initial_state_dict)
        if isinstance(self.optims, dict):
            optimizer = self.optims.get("optimizer", None)
        else:
            optimizer = nn.Adam(
                self.parameters(),
                lr           = lr,
                betas        = (0.9, 0.999),
                eps          = 1e-8,
                weight_decay = weight_decay,
            )
        
        # Pre-processing
        self.assert_datapoint(datapoint)
        for k, v in datapoint.items():
            if isinstance(v, torch.Tensor):
                datapoint[k] = v.to(self.device)
        image = datapoint.get("image")
        image = core.resize(image, divisible_by=32)
        
        # Training
        for _ in range(epochs):
            image   = self.augment(image)
            outputs = self.forward_loss(datapoint={"image": image})
            optimizer.zero_grad()
            loss = outputs["loss"]
            loss.backward()
            optimizer.step()
            
        # Inference
        self.eval()
        image    = datapoint.get("image")
        h, w     = core.get_image_size(image)
        image    = core.resize(image, divisible_by=32)
        timer    = core.Timer()
        timer.tick()
        outputs  = self.forward(datapoint={"image": image})
        timer.tock()
        
        # Post-processing
        enhanced = outputs["enhanced"]
        enhanced = core.resize(enhanced, (h, w))
        outputs["enhanced"] = torch.clamp(enhanced, 0, 1)
        self.assert_outputs(outputs)
        
        # Return
        outputs["time"] = timer.avg_time
        return outputs
    
# endregion

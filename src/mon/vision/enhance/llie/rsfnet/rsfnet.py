#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""RSFNet.

This module implement the paper: RSFNet: Specularity Factorization for Low Light
Enhancement.

Reference:
    https://github.com/sophont01/RSFNet
"""

from __future__ import annotations

__all__ = []

from typing import Any, Literal

import kornia
import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import functional as F
import torch.linalg as la
from mon import core, nn
from mon.globals import MODELS, Scheme, Task
from mon.vision import filtering
from mon.vision.enhance import base

console      = core.console
current_file = core.Path(__file__).absolute()
current_dir  = current_file.parents[0]
eps          = torch.finfo(torch.float32).eps


# region Loss

class Loss(nn.Loss):
    
    def __init__(
        self,
        col_weight : float = 10,
        exp_weight : float = 2,
        tv_weight  : float = 2,
        fact_weight: float = 2,
        loss_weight: float = 1.0,
        reduction  : Literal["none", "mean", "sum"] = "mean",
        *args, **kwargs
    ):
        super().__init__(loss_weight=loss_weight, reduction=reduction)
        self.col_weight  = col_weight
        self.exp_weight  = exp_weight
        self.tv_weight   = tv_weight
        self.fact_weight = fact_weight
        self.l_color     = nn.ColorConstancyLoss
        self.l_exp       = nn.ExposureValueControlLoss(patch_size=16, mean_val=0.6)
        self.l_tv        = nn.TotalVariationLoss()
        
    def forward(
        self,
        illu_lr         : torch.Tensor,
        image_v_lr      : torch.Tensor,
        image_v_fixed_lr: torch.Tensor,
    ) -> torch.Tensor:
        loss_spa      = torch.mean(torch.abs(torch.pow(illu_lr - image_v_lr, 2)))
        loss_tv       = self.l_tv(illu_lr)
        loss_exp      = torch.mean(self.l_exp(illu_lr))
        loss_sparsity = torch.mean(image_v_fixed_lr)
        loss = (
                   loss_spa * self.alpha
                  + loss_tv * self.beta
                 + loss_exp * self.gamma
            + loss_sparsity * self.delta
        )
        return loss

# endregion


# region Module

class Factorization(nn.Module):
    
    def __init__(
        self,
        factors      : int   = 5,
        num_iters    : int   = 3,
        freeze_epochs: int   = 25,
        eta_a        : float = 0.5,
        is_train     : bool  = True,
    ):
        super().__init__()
        self.factors       = factors
        self.num_iters     = num_iters
        self.freeze_epochs = freeze_epochs
        self.eta_a         = eta_a
        self.is_train      = is_train
        self.initialize    = True
        self.epoch         = 0
        self.et_mean       = [[] for _ in range(self.factors)]
        self.x_mean        = 0
        self.relu          = nn.ReLU(inplace=True)
        self.lambda_a      = nn.ModuleList()
        self.lambda_e      = nn.ModuleList()
        self.step          = nn.ModuleList()
        for i in range(self.factors):
            self.lambda_a.append((nn.ParameterList([nn.Parameter(Variable(torch.tensor(0.0, dtype=torch.float32), requires_grad=True)) for t in range(self.num_iters)])))
            self.lambda_e.append((nn.ParameterList([nn.Parameter(Variable(torch.tensor(0.0, dtype=torch.float32), requires_grad=True)) for t in range(self.num_iters)])))
            self.step.append((    nn.ParameterList([nn.Parameter(Variable(torch.tensor(1.0, dtype=torch.float32), requires_grad=True)) for t in range(self.num_iters)])))
        self.lambda_a_backup = [[torch.tensor(0.0, dtype=torch.float32, requires_grad=False) for t in range(self.num_iters)] for tt in range(self.factors)]
        self.lambda_e_backup = [[torch.tensor(0.0, dtype=torch.float32, requires_grad=False) for t in range(self.num_iters)] for tt in range(self.factors)]
        self.step_backup     = [[torch.tensor(0.0, dtype=torch.float32, requires_grad=False) for t in range(self.num_iters)] for tt in range(self.factors)]
       
    def thres_e(self, inputs: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
        norm_inputs = la.vector_norm(inputs, dim=1)
        out         = torch.max(1 - torch.div(threshold, (norm_inputs + eps)), torch.zeros([1, 1])).unsqueeze(1).repeat(1, inputs.shape[1], 1, 1) * inputs
        return out

    def thres_a(self, inputs: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
        norm_inputs      = la.vector_norm(inputs, dim=1)
        norm_inputs_norm = torch.sqrt(torch.sum(norm_inputs, dim=[1, 2]) + eps)
        out              = torch.max(1 - torch.div(threshold, (norm_inputs_norm + eps)), torch.zeros([1, 1])).t().repeat(1, inputs.shape[1]).unsqueeze(2).unsqueeze(3) * inputs
        return out
    
    def check_negative(self, f: int) -> bool:
        is_negative = False
        if self.is_train:
            for t in range(self.num_iters):
                if (self.lambda_a[f][t].data < 0) or (self.lambda_e[f][t].data < 0):
                    is_negative = True
            if is_negative:
                for t in range(self.num_iters):
                    self.lambda_a[f][t].data = self.lambda_a_backup[f][t]
                    self.lambda_e[f][t].data = self.lambda_e_backup[f][t]
                    self.step[f][t].data     = self.step_backup[f][t]
            else:
                for t in range(self.num_iters):
                    self.lambda_a_backup[f][t] = self.lambda_a[f][t].data
                    self.lambda_e_backup[f][t] = self.lambda_e[f][t].data
                    self.step_backup[f][t]     = self.step[f][t].data
        return is_negative
    
    def initialize_ths(self, f: int, s: int, x_mean = None):
        eta_a = self.eta_a
        eta_b = (f + 1) / self.factors
        if s == 0:
            if self.initialize:
                self.lambda_a[f][0].data = torch.clone(eta_a * self.lambda_a[f][0].data + (1 - eta_a) * eta_b * x_mean)
                self.lambda_e[f][0].data = torch.clone(eta_a * self.lambda_e[f][0].data + (1 - eta_a) * (1 - eta_b) * x_mean)
        if s > 0 and self.initialize:
            # THIS SHOULD HAPPEN ONLY FOR ONE TIME
            # print('INITIALIZING STAGE')
            self.lambda_a[f][s].data = torch.clone(self.lambda_a[f][s - 1].data)
            self.lambda_e[f][s].data = torch.clone(self.lambda_e[f][s - 1].data)
            self.step[f][s].data     = torch.clone(self.step[f][s-1].data)
            if s == self.num_iters - 1:  # last stage
                self.initialize = False
                # print('SWITCHING OFF INITIALIZATION !!!')
        if self.epoch > self.freeze_epochs:
            self.lambda_a[f][s].requires_grad_ = False
            self.lambda_e[f][s].requires_grad_ = False
            self.step[f][s].requires_grad_     = False

    def factorize(self, x: torch.Tensor, f: int) -> tuple[torch.Tensor, torch.Tensor]:
        eta_b  = (f + 1) / self.factors
        x_2    = la.vector_norm(x, ord=2)
        x_mean = torch.mean(torch.ravel(x))
        
        self.initialize_ths(f, 0, x_mean)
        e_ths = self.lambda_e[f][0] / self.step[f][0]
        e_t   = self.thres_e(x, e_ths)
        a_ths = self.lambda_a[f][0] / self.step[f][0]
        a_t   = self.thres_a(x - e_t, a_ths)
        y_t   = torch.div(x, x_2 + eps)
        for t in range(1, self.num_iters):
            self.initialize_ths(f, t)
            e_ths = self.lambda_e[f][t] / self.step[f][t]
            e_t   = self.thres_e(x - a_t - y_t / self.step[f][t], e_ths)
            a_ths = self.lambda_a[f][t] / self.step[f][t]
            a_t   = self.thres_a(x - e_t - y_t / self.step[f][t], a_ths)
            y_t   = y_t + self.step[f][t] * (e_t + a_t - x)

        e_t  = self.relu(e_t)
        a_t  = self.relu(a_t)
        loss = torch.abs(torch.sum(e_t) / (torch.sum(x) + eps) - eta_b)
        return e_t, loss
    
    def forward(self, input: torch.Tensor, epoch: int) -> tuple[torch.Tensor, torch.Tensor]:
        self.epoch = epoch
        all_e = torch.Tensor().to(input.device)
        loss  = 0
        # Factorization
        a = input
        for i in range(self.factors):
            e, l = self.factorize(a, i)
            if self.is_train:
                self.et_mean[i].append(torch.sum(e) / torch.sum(input))
                loss += l
            a = a - e
            if i > 0:
                e = torch.abs(e - all_e[:, 3 * (i - 1):3 * i, :, :])
            all_e = torch.cat([all_e, e], dim=1)
        # Update x_mean
        if self.is_train:
            self.x_mean = (self.x_mean + torch.mean(input)) * 0.5
        # Return
        return all_e, loss


class Fusion(nn.Module):
    
    def __init__(self, factors: int):
        super().__init__()
        self.factors = factors
        num_filters  = 3
        in_channels  = 3 * (self.factors + 1)
        out_channels = 3 * (self.factors + 1)
        self.e_conv1 = nn.Conv2d(in_channels,     num_filters,  3, 1, 1, bias=True)
        self.e_conv2 = nn.Conv2d(num_filters,     num_filters,  3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(num_filters,     num_filters,  3, 1, 1, bias=True)
        self.e_conv4 = nn.Conv2d(num_filters,     num_filters,  3, 1, 1, bias=True)
        self.d_conv5 = nn.Conv2d(num_filters * 2, num_filters,  3, 1, 1, bias=True)
        self.d_conv6 = nn.Conv2d(num_filters * 2, num_filters,  3, 1, 1, bias=True)
        self.d_conv7 = nn.Conv2d(num_filters * 2, out_channels, 3, 1, 1, bias=True)
        self.encoder = nn.ModuleList([self.e_conv1, self.e_conv2, self.e_conv3, self.e_conv4])
        self.decoder = nn.ModuleList([self.d_conv5, self.d_conv6, self.d_conv7])
        self.relu    = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, S: torch.Tensor) -> torch.Tensor:
        s  = list(torch.split(S, 3, dim=1))
        w  = [1, 1, 1, 1, 1, 1]
        for i in range(len(s)):
            s[i] = s[i] * w[i]
        S  = torch.cat(s, dim=1)
        e1 = self.relu(self.e_conv1(S))
        e2 = self.relu(self.e_conv2(e1))
        e3 = self.relu(self.e_conv3(e2))
        e4 = self.relu(self.e_conv3(e3))
        d1 = self.relu(self.d_conv5(torch.cat([e3, e4], 1)))
        d2 = self.relu(self.d_conv6(torch.cat([e2, d1], 1)))
        o  = torch.tanh(self.d_conv7(torch.cat([e1, d2], 1)))
        #
        r  = list(torch.split(o, 3, dim=1))
        x  = s[0]
        for i in range(5):
            for j in range(self.factors + 1):
                x = x + r[j] * (torch.pow(x, 2) - x)
        #
        return x

# endregion


# region Model

@MODELS.register(name="rsfnet", arch="rsfnet")
class RSFNet(base.ImageEnhancementModel):

    model_dir: core.Path    = current_dir
    arch     : str          = "rsfnet"
    tasks    : list[Task]   = [Task.LLIE]
    schemes  : list[Scheme] = [Scheme.ZERO_REFERENCE, Scheme.INSTANCE]
    zoo      : dict         = {}
    
    def __init__(
        self,
        name         : str   = "rsfnet",
        factors      : int   = 5,
        num_iters    : int   = 3,
        freeze_epochs: int   = 25,
        eta_a        : float = 0.5,
        is_train     : bool  = True,
        denoise      : bool  = True,
        weights      : Any   = None,
        *args, **kwargs
    ):
        super().__init__(
            name    = name,
            weights = weights,
            *args, **kwargs
        )
        self.factors       = factors
        self.num_iters     = num_iters
        self.freeze_epochs = freeze_epochs
        self.eta_a         = eta_a
        self.is_train      = is_train
        self.denoise       = denoise
        
        # Loss
        # self.loss = Loss(L, alpha, beta, gamma, delta)
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)
    
    def init_weights(self, m: nn.Module):
        pass
    
    def freeze(self, epoch: int):
        if epoch > self.freeze:
            for i in range(self.factors):
                for j in range(self.num_iters):
                    self.fact_net.lambda_A[i][j].requires_grad = False
                    self.fact_net.lambda_E[i][j].requires_grad = False
                    self.fact_net.step[i][j].requires_grad     = False
            for param in self.fuse_net.parameters():
                param.requires_grad = True
        
    def forward(self, datapoint: dict, *args, **kwargs) -> dict:
        self.assert_datapoint(datapoint)
        
        
# endregion

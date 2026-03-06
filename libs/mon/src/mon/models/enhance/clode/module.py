#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Modules.

This module provides various layers, blocks, and modules for the CLODE model.
"""

from __future__ import annotations

__all__ = [
    "NODE",
]

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torchdiffeq import odeint_adjoint

from .loss import L_tv

MAX_NUM_STEPS = 1000  # 30  # 50  # 100


# ==============================================================================
# region UTILITIES
# ==============================================================================

def normalize_minmax(x: Tensor, scale: float = 1) -> Tensor:
    x = x * scale
    return (x - x.min()) / (x.max() - x.min())

# endregion


# ==============================================================================
# region MODULES
# ==============================================================================

class Conv2dTime(nn.Conv2d):

    # --- Lifecycle & Initialization ---
    def __init__(self, in_channels: int, *args, **kwargs):
        super().__init__(in_channels + 1, *args, **kwargs)

    # --- Callable & Context Manager ---
    def forward(self, t: Tensor, x: Tensor) -> Tensor:
        t_img = torch.ones_like(x[:, :1, :, :]) * t  # (B, 1, H, W)
        t_and_x = torch.cat([t_img, x], 1)  # (B, C + 1, H, W)
        return super(Conv2dTime, self).forward(t_and_x)


class Network(nn.Module):

    # --- Lifecycle & Initialization ---
    def __init__(self, in_channels: int, embed_channels: int = 48):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, embed_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(embed_channels, embed_channels, 3, padding=1)
        self.conv3 = nn.Conv2d(embed_channels, in_channels, 1)
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    # --- Callable & Context Manager ---
    def forward(self, x: Tensor) -> Tensor:
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.conv3(x)
        return x


class EnhanceFunc(nn.Module):

    # --- Lifecycle & Initialization ---
    def __init__(self, num_filters: int = 32):
        super().__init__()
        self.nfe = 0
        self.pred_t = []
        self.last_curve_map = None

        in_channels = 6
        out_channels = 3
        self.up_conv = Conv2dTime(in_channels, num_filters, kernel_size=3, padding=1, padding_mode="reflect")
        self.conv_3_1 = Conv2dTime(num_filters,  num_filters, kernel_size=3, padding=3 // 2, padding_mode="reflect")
        self.conv_5_1 = Conv2dTime(num_filters, num_filters, kernel_size=5, padding=5 // 2, padding_mode="reflect")
        self.conv_3_2 = Conv2dTime(num_filters * 2, num_filters * 2, kernel_size=3, padding=3 // 2, padding_mode="reflect")
        self.conv_5_2 = Conv2dTime(num_filters * 2, num_filters * 2, kernel_size=5, padding=5 // 2, padding_mode="reflect")
        self.confusion = Conv2dTime(num_filters * 4, num_filters, kernel_size=1, padding=0, padding_mode="reflect")
        self.down_conv = Conv2dTime(num_filters, out_channels, kernel_size=3, padding=1, padding_mode="reflect")
        self.norm32 = nn.GroupNorm(1, 32)
        self.norm64 = nn.GroupNorm(1, 64)
        self.relu = nn.ReLU(inplace=True)

        self.denoise = Network(3)
        self.L_tv = L_tv()

    # --- Callable & Context Manager ---
    def forward(self, t: Tensor, x: Tensor) -> Tensor:
        self.nfe  += 1

        _x = x[:, :3 , :, :]
        _, c, h, w = _x.shape

        noise_map = self.loss_func(_x)
        p_x = _x - self.denoise(_x)
        _in = torch.cat([p_x, 1 - p_x], 1)

        input_1 = self.relu(self.norm32(self.up_conv(t, _in)))
        output_3_1 = self.relu(self.norm32(self.conv_3_1(t, input_1)))
        output_5_1 = self.relu(self.norm32(self.conv_5_1(t, input_1)))
        input_2 = torch.cat([output_3_1, output_5_1], 1)
        output_3_2 = self.relu(self.norm64(self.conv_3_2(t, input_2)))
        output_5_2 = self.relu(self.norm64(self.conv_5_2(t, input_2)))
        input_3 = torch.cat([output_3_2, output_5_2], 1)
        output = self.relu(self.norm32(self.confusion(t, input_3)))
        _A = F.tanh(self.down_conv(t, output))

        pred = _A * (torch.pow(_x, 2) - _x)
        l_tv = torch.ones_like(_A) * self.L_tv(_A)
        noise_map = torch.ones_like(_A) * noise_map

        self.last_curve_map = _A
        self.pred_t.append(t.item())

        return torch.cat([pred, l_tv, noise_map], 1)

    # --- Denoise ---
    def pair_downsampler(self, image: Tensor) -> tuple[Tensor, Tensor]:
        c = image.shape[1]
        filter1 = torch.FloatTensor([[[[0, 0.5],[0.5, 0]]]]).to(image.device)
        filter1 = filter1.repeat(c,1, 1, 1)
        filter2 = torch.FloatTensor([[[[0.5, 0],[0, 0.5]]]]).to(image.device)
        filter2 = filter2.repeat(c,1, 1, 1)
        output1 = F.conv2d(image, filter1, stride=2, groups=c)
        output2 = F.conv2d(image, filter2, stride=2, groups=c)
        return output1, output2

    def loss_func(self, noisy_image: Tensor) -> Tensor:
        mse = nn.MSELoss()

        noisy1, noisy2 = self.pair_downsampler(noisy_image)
        pred1 = noisy1 - self.denoise(noisy1)
        pred2 = noisy2 - self.denoise(noisy2)
        loss_res = 0.5 * (mse(noisy1, pred2) + mse(noisy2, pred1))

        noisy_denoised = noisy_image - self.denoise(noisy_image)
        denoised1, denoised2 = self.pair_downsampler(noisy_denoised)
        loss_cons = 0.5 * (mse(pred1, denoised1) + mse(pred2, denoised2))

        loss = loss_res + loss_cons
        return loss

    def add_noise(self, x: Tensor, noise_level: float) -> Tensor:
        noisy = x + torch.normal(0, noise_level / 255, x.shape).to(x.device)
        noisy = torch.clamp(noisy,0,1)
        return noisy


class ODEBlock(nn.Module):

    # --- Lifecycle & Initialization ---
    def __init__(self, odefunc: nn.Module, tol: float = 1e-3, adjoint: bool = False):
        super().__init__()
        self.odefunc = odefunc
        self.tol = tol
        self.adjoint = adjoint

    # --- Callable & Context Manager ---
    def forward(self, x: Tensor, eval_time: Tensor = None) -> Tensor:
        if eval_time is None:
            t = torch.tensor([0, 1]).float().type_as(x)
        else:
            t = eval_time

        self.odefunc.nfe = 0
        x_aug = x

        return odeint_adjoint(
            func=self.odefunc,
            y0=x_aug,
            t=t,
            rtol=self.tol,
            atol=self.tol,
            method="dopri5",  # "dopri5", "euler", "rk4"
            options={"max_num_steps": MAX_NUM_STEPS, },
        )


class NODE(nn.Module):

    # --- Lifecycle & Initialization ---
    def __init__(
        self,
        num_filters: int = 32,
        tol: float = 1e-5,
        adjoint: bool = True,
    ):
        super().__init__()
        self.num_filters = num_filters
        self.odefunc = EnhanceFunc(num_filters)
        self.odeblock = ODEBlock(self.odefunc, tol=tol, adjoint=adjoint)

    # --- Callable & Context Manager ---
    def forward(
        self,
        x: Tensor,
        eval_time: Tensor = None,
        inference: bool = False
    ) -> dict:
        _input = torch.cat([x, torch.zeros_like(x), torch.zeros_like(x)], 1)
        preds = self.odeblock(_input, eval_time)
        pred = preds[-1]
        curve_map = self.odefunc.last_curve_map

        if inference:
            return {
                "output": torch.clamp(pred[:, 0:3, :, :]  - self.odefunc.denoise(pred[:, 0:3, :, :]), 0, 1),
                "curve_map": normalize_minmax(curve_map),
                "noise_map": self.odefunc.denoise(pred[:, 0:3, :, :]),  # normalize_minmax(self.odefunc.denoise(pred[:, 0:3, :, :]), 255),  # self.odefunc.denoise(pred[:, 0:3, :, :]),
                "all": [torch.clamp(pred[:, 0:3, :, :] - self.odefunc.denoise(pred[:, 0:3, :, :]), 0, 1) for pred in preds],
            }
        else:
            return {
                "output": pred[:, 0:3, :, :],
                "curve_map": pred[:, 3:6, :, :],
                "noise_map": pred[:, 6:9, :, :],
            }

# endregion


# ==============================================================================
# region UNIT TEST
# ==============================================================================

if __name__ == "__main__":
    pass

# endregion

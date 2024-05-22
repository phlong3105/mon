#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements KAN (Kolmogorov Arnold Networks) convolutional layers.
"""

from __future__ import annotations

__all__ = [
    "FastKANConv2d",
    "SplineConv2d",
]

import torch
import torch.nn.functional as F
from torch import nn

from mon.core import _size_2_t


# region KA-Conv: Kolmogorov-Arnold Convolutional Networks with Various Basis Functions

class PolynomialFunction(nn.Module):
    
    def __init__(self, degree: int = 3):
        super().__init__()
        self.degree = degree
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.stack([input ** i for i in range(self.degree)], dim=-1)


class BSplineFunction(nn.Module):
    
    def __init__(
        self,
        grid_min : float = -2.0,
        grid_max : float = 2.0,
        degree   : int   = 3,
        num_basis: int   = 8,
    ):
        super().__init__()
        self.degree    = degree
        self.num_basis = num_basis
        self.knots     = torch.linspace(grid_min, grid_max, num_basis + degree + 1)  # Uniform knots
    
    def basis_function(self, i, k, t):
        if k == 0:
            return ((self.knots[i] <= t) & (t < self.knots[i + 1])).float()
        else:
            left_num  = (t - self.knots[i]) * self.basis_function(i, k - 1, t)
            left_den  = self.knots[i + k] - self.knots[i]
            left      = left_num / left_den if left_den != 0 else 0
            
            right_num = (self.knots[i + k + 1] - t) * self.basis_function(i + 1, k - 1, t)
            right_den = self.knots[i + k + 1] - self.knots[i + 1]
            right     = right_num / right_den if right_den != 0 else 0
            
            return left + right
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input.squeeze()  # Assuming x is of shape (B, 1)
        basis_functions = torch.stack([self.basis_function(i, self.degree, x) for i in range(self.num_basis)], dim=-1)
        return basis_functions


class ChebyshevFunction(nn.Module):
    
    def __init__(self, degree: int = 4):
        super().__init__()
        self.degree = degree
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        chebyshev_polynomials = [torch.ones_like(x), x]
        for n in range(2, self.degree):
            chebyshev_polynomials.append(2 * x * chebyshev_polynomials[-1] - chebyshev_polynomials[-2])
        return torch.stack(chebyshev_polynomials, dim=-1)


class FourierBasisFunction(nn.Module):
    
    def __init__(self, num_frequencies: int = 4, period: float = 1.0):
        super().__init__()
        assert num_frequencies % 2 == 0, ":param:`num_frequencies` must be even"
        self.num_frequencies = num_frequencies
        self.period          = nn.Parameter(torch.Tensor([period]), requires_grad=False)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        frequencies     = torch.arange(1, self.num_frequencies // 2 + 1, device=x.device)
        sin_components  = torch.sin(2 * torch.pi * frequencies * x[..., None] / self.period)
        cos_components  = torch.cos(2 * torch.pi * frequencies * x[..., None] / self.period)
        basis_functions = torch.cat([sin_components, cos_components], dim=-1)
        return basis_functions


class RadialBasisFunction(nn.Module):
    
    def __init__(
        self,
        grid_min   : float = -2.0,
        grid_max   : float = 2.0,
        num_grids  : int   = 4,
        denominator: float = None,
    ):
        super().__init__()
        grid      = torch.linspace(grid_min, grid_max, num_grids)
        self.grid = torch.nn.Parameter(grid, requires_grad=False)
        self.denominator = denominator or (grid_max - grid_min) / (num_grids - 1)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.exp(-((input[..., None] - self.grid) / self.denominator) ** 2)


class SplineConv2d(nn.Conv2d):
    """
    
    References:
        `<https://github.com/JaouadT/KANU_Net/blob/main/src/fastkanconv.py>`__
    """
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : _size_2_t = 3,
        stride      : _size_2_t = 1,
        padding     : _size_2_t = 0,
        dilation    : _size_2_t = 1,
        groups      : int       = 1,
        bias        : bool      = True,
        init_scale  : float     = 0.1,
        padding_mode: str       = "zeros",
        **kwargs
    ):
        self.init_scale = init_scale
        super().__init__(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = kernel_size,
            stride       = stride,
            padding      = padding,
            dilation     = dilation,
            groups       = groups,
            bias         = bias,
            padding_mode = padding_mode,
            **kwargs
        )
    
    def reset_parameters(self):
        nn.init.trunc_normal_(self.weight, mean=0, std=self.init_scale)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class FastKANConv2d(nn.Module):
    """KA-Conv: Kolmogorov-Arnold Convolutional Networks with Various Basis
    Functions.
    
    Args:
        kan_type: The type of KAN basis function to use. The choices are:
            ``"RBF"``, ``"Fourier"``, ``"Poly"``, ``"Chebyshev"``, ``"BSpline"``.
            Default: ``"RBF"``.
    
    References:
        `<https://github.com/XiangboGaoBarry/KA-Conv/blob/main/kaconv/kaconv/fastkanconv.py>`__
    """
    
    def __init__(
        self,
        in_channels             : int,
        out_channels            : int,
        kernel_size             : _size_2_t = 3,
        stride                  : _size_2_t = 1,
        padding                 : _size_2_t = 0,
        dilation                : _size_2_t = 1,
        groups                  : int       = 1,
        bias                    : bool      = True,
        grid_min                : float     = -2.0,
        grid_max                : float     = 2.0,
        num_grids               : int       = 4,
        use_base_update         : bool      = True,
        base_activation         : nn.Module = F.silu,
        spline_weight_init_scale: float     = 0.1,
        padding_mode            : str       = "zeros",
        kan_type                : str       = "RBF",
    ):
        super().__init__()
        if kan_type == "RBF":
            self.rbf = RadialBasisFunction(grid_min, grid_max, num_grids)
        elif kan_type == "Fourier":
            self.rbf = FourierBasisFunction(num_grids)
        elif kan_type == "Poly":
            self.rbf = PolynomialFunction(num_grids)
        elif kan_type == "Chebyshev":
            self.rbf = ChebyshevFunction(num_grids)
        elif kan_type == "BSpline":
            self.rbf = BSplineFunction(grid_min, grid_max, 3, num_grids)
        else:
            raise ValueError(f"KAN type {kan_type} not supported.")
        
        self.spline_conv = SplineConv2d(
            in_channels  = in_channels * num_grids,
            out_channels = out_channels,
            kernel_size  = kernel_size,
            stride       = stride,
            padding      = padding,
            dilation     = dilation,
            groups       = groups,
            bias         = bias,
            init_scale   = spline_weight_init_scale,
            padding_mode = padding_mode,
        )
        
        self.use_base_update = use_base_update
        if use_base_update:
            self.base_activation = base_activation
            self.base_conv = nn.Conv2d(
                in_channels  = in_channels,
                out_channels = out_channels,
                kernel_size  = kernel_size,
                stride       = stride,
                padding      = padding,
                dilation     = dilation,
                groups       = groups,
                bias         = bias,
                padding_mode = padding_mode,
            )
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x          = input
        b, c, h, w = x.shape
        x_rbf      = self.rbf(x.view(b, c, -1)).view(b, c, h, w, -1)
        x_rbf      = x_rbf.permute(0, 4, 1, 2, 3).contiguous().view(b, -1, h, w)
        # Apply spline convolution
        ret = self.spline_conv(x_rbf)
        if self.use_base_update:
            base = self.base_conv(self.base_activation(x))
            ret  = ret + base
        return ret
    
# endregion

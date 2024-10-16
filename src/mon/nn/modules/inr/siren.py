#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""SIREN Network.

This module implements SIREN, introduced in the paper: "Implicit Neural
Representations with Periodic Activation Functions," by Sitzmann et al. (2020).

References:
    https://github.com/lucidrains/siren-pytorch
"""

from __future__ import annotations

__all__ = [
	"SIREN",
    "OutputINF",
    "PatchINF",
    "SIRENNet",
    "SIRENWrapper",
    "SpatialINF",
]

import math

import numpy as np
import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F

from mon import core
from mon.nn.modules import activation as act


# region SIREN Layer

class SIREN(nn.Module):
    """SIREN Layer.
    
    Args:
        in_channels: The number of input channels.
        out_channels: The number of output channels.
        omega_0: The frequency of the sine activation function. Defaults: ``1.0``.
        C: The constant factor for the weight initialization. Defaults: ``6.0``.
        is_first: Whether this is the first layer. Defaults: ``False``.
        is_last: Whether this is the last layer. Defaults: ``False``.
        use_bias: Whether to use bias. Defaults: ``True``.
        activation: The activation function.
            - If :obj:`activation` is ``None``, then :obj:`nn.Sine(omega_0)` is
                used.
            - If :obj:`is_last` is ``True`` and :obj:`activation` is ``None``,
                then :obj:`nn.Sigmoid()` is used.
            Defaults: ``None``.
        dropout: The dropout rate. Defaults: ``0.0``.
    
    References:
        https://github.com/lucidrains/siren-pytorch
    """
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        omega_0     : int       = 1.0,
        C           : float     = 6.0,
        is_first    : bool      = False,
        is_last     : bool      = False,
        use_bias    : bool      = True,
        activation  : nn.Module = None,
        dropout     : float     = 0.0
    ):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.omega_0      = omega_0
        self.C 		      = C
        self.is_first     = is_first
        self.is_last      = is_last
        self.use_bias     = use_bias
        self.linear       = nn.Linear(self.in_channels, self.out_channels)
        if self.is_last:
            self.activation = act.Sigmoid() if activation is None else activation
        else:
            self.activation = act.Sine(self.omega_0) if activation is None else activation
        self.dropout = nn.Dropout(dropout)
        if not self.is_last:
            self.init_weights()
    
    def init_weights(self):
        if self.is_first:
            w_std = 1 / self.in_channels
        else:
            w_std = math.sqrt(self.C / self.in_channels) / self.omega_0
        with torch.no_grad():
            self.linear.weight.uniform_(-w_std, w_std)
            if self.use_bias:
                self.linear.bias.uniform_(-w_std, w_std)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = self.linear(x)
        y = self.activation(y)
        y = self.dropout(y)
        return y
    
# endregion


# region SIREN Network

class SIRENNet(nn.Module):
    """SIREN Network.
    
    Args:
        in_channels: The number of input channels.
        out_channels: The number of output channels.
        hidden_channels: The number of hidden channels.
        num_layers: The number of layers.
        omega_0: The frequency of the sine activation function. Defaults: ``1.0``.
        omega_0_initial: The initial frequency of the sine activation function.
            Defaults: ``30.0``.
        use_bias: Whether to use bias. Defaults: ``True``.
        dropout: The dropout rate. Defaults: ``0.0``.
    
    References:
        https://github.com/lucidrains/siren-pytorch/blob/master/siren_pytorch/siren_pytorch.py
    """
    
    def __init__(
        self,
        in_channels     : int,
        out_channels    : int,
        hidden_channels : int,
        num_layers      : int,
        omega_0         : float     = 1.0,
        omega_0_initial : float     = 30.0,
        use_bias        : bool      = True,
        final_activation: nn.Module = None,
        dropout         : float     = 0.0
    ):
        super().__init__()
        self.num_layers      = num_layers
        self.hidden_channels = hidden_channels
        
        self.layers = nn.ModuleList([])
        for idx in range(num_layers):
            is_first          = idx == 0
            layer_w0          = omega_0_initial if is_first else omega_0
            layer_in_channels = in_channels     if is_first else self.hidden_channels
            layer = SIREN(
                in_channels  = layer_in_channels,
                out_channels = self.hidden_channels,
                omega_0      = layer_w0,
                C            = 6.0,
                use_bias     = use_bias,
                is_first     = is_first,
                dropout      = dropout,
            )
            self.layers.append(layer)
        
        final_activation = nn.Identity() if final_activation is None else final_activation
        self.last_layer = SIREN(
            in_channels  = in_channels,
            out_channels = out_channels,
            omega_0      = omega_0,
            use_bias     = use_bias,
            activation   = final_activation,
        )
    
    def forward(self, input: torch.Tensor, mods: torch.Tensor = None):
        mods = self.cast_tuple(mods, self.num_layers)
        x    = input
        for layer, mod in zip(self.layers, mods):
            x = layer(x)
            if mod is not None:
                x *= rearrange(mod, "d -> () d")
        return self.last_layer(x)
    
    @staticmethod
    def cast_tuple(val: torch.Tensor, repeat: int = 1):
        return val if isinstance(val, tuple) else ((val,) * repeat)


class Modulator(nn.Module):
    
    def __init__(
        self,
        in_channels    : int,
        hidden_channels: int,
        num_layers     : int,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for idx in range(num_layers):
            is_first          = idx == 0
            layer_in_channels = in_channels if is_first else (hidden_channels + in_channels)
            self.layers.append(
                nn.Sequential(
                    nn.Linear(layer_in_channels, hidden_channels),
                    nn.ReLU()
                )
            )

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor]:
        x       = z
        hiddens = []
        for layer in self.layers:
            x = layer(x)
            hiddens.append(x)
            x = torch.cat((x, z))
        return tuple(hiddens)


class SIRENWrapper(nn.Module):
    
    def __init__(
        self,
        net            : SIRENNet,
        image_width    : int,
        image_height   : int,
        latent_channels: int = None
    ):
        super().__init__()
        if not isinstance(net, SIRENNet):
            raise TypeError(f"Expected `net` to be an instance of `SIRENNet`, "
                            f"but got {type(net).__name__}")

        self.net          = net
        self.image_width  = image_width
        self.image_height = image_height
        self.modulator    = None
        if latent_channels is not None:
            self.modulator = Modulator(
                in_channels     = latent_channels,
                hidden_channels = net.dim_hidden,
                num_layers      = net.num_layers
            )

        tensors = [
            torch.linspace(-1, 1, steps=image_height),
            torch.linspace(-1, 1, steps=image_width)
        ]
        mgrid   = torch.stack(torch.meshgrid(*tensors, indexing = "ij"), dim=-1)
        mgrid   = rearrange(mgrid, "h w c -> (h w) c")
        self.register_buffer("grid", mgrid)

    def forward(
        self,
        image: torch.Tensor = None,
        *,
        latent: torch.Tensor = None
    ):
        modulate = self.modulator is not None
        assert not (modulate ^ latent is not None), "latent vector must be only supplied if `latent_dim` was passed in on instantiation"
        mods   = self.modulator(latent) if modulate else None
        coords = self.grid.clone().detach().requires_grad_()
        out    = self.net(coords, mods)
        out    = rearrange(out, "(h w) c -> () c h w", h=self.image_height, w=self.image_width)
        if image is not None:
            return F.mse_loss(image, out)
        return out

# endregion


# region INF Function

class PatchINF(nn.Module):
    """Implicit Neural Function built on top of SIREN.
    
    References:
        https://github.com/ctom2/colie
    """
    
    def __init__(
        self,
        window_size : int   = 1,
        out_channels: int   = 256,
        down_size   : int   = 256,
        num_layers  : int   = 2,
        omega_0     : float = 30.0,
        siren_c     : float = 6.0,
        weight_decay: float = 0.0001,
    ):
        super().__init__()
        self.window_size = window_size
        self.patch_dim   = window_size ** 2
        self.down_size   = down_size
        patch_layers     = [SIREN(self.patch_dim, out_channels, omega_0, siren_c, is_first=True)]
        for _ in range(1, num_layers):
            patch_layers.append(SIREN(out_channels, out_channels, omega_0, siren_c))
        patch_layers.append(SIREN(out_channels, out_channels, omega_0, siren_c))
        self.patch_net = nn.Sequential(*patch_layers)
        
        weight_decay = weight_decay or 0.0001
        self.params  = [{"params": self.patch_net.parameters(), "weight_decay": weight_decay}]
        
    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        image_lr = self.interpolate_image(image)
        patch    = self.patch_net(self.get_patches(image_lr))
        return image_lr, patch
    
    def interpolate_image(self, image: torch.Tensor) -> torch.Tensor:
        """Reshapes the image based on new resolution."""
        return F.interpolate(image, size=(self.down_size, self.down_size), mode="bicubic")
    
    def get_patches(self, image: torch.Tensor) -> torch.Tensor:
        """Creates a tensor where the channel contains patch information."""
        num_channels = core.get_image_num_channels(image)
        kernel       = torch.zeros((self.window_size ** 2, num_channels, self.window_size, self.window_size)).to(image.device)
        for i in range(self.window_size):
            for j in range(self.window_size):
                kernel[int(torch.sum(kernel).item()), 0, i, j] = 1
        
        pad       = nn.ReflectionPad2d(self.window_size // 2)
        im_padded = pad(image)
        extracted = F.conv2d(im_padded, kernel, padding=0).squeeze(0)
        return torch.movedim(extracted, 0, -1)


class SpatialINF(nn.Module):
    """Implicit Neural Function for coordinates built on top of SIREN.
    
    References:
        https://github.com/ctom2/colie
    """
    
    def __init__(
        self,
        out_channels: int   = 256,
        down_size   : int   = 256,
        num_layers  : int   = 2,
        omega_0     : float = 30.0,
        siren_c     : float = 6.0,
        weight_decay: float = 0.1,
    ):
        super().__init__()
        self.down_size = down_size
        spatial_layers = [SIREN(2, out_channels, omega_0, siren_c, is_first=True)]
        for _ in range(1, num_layers):
            spatial_layers.append(SIREN(out_channels, out_channels, omega_0, siren_c))
        spatial_layers.append(SIREN(out_channels, out_channels, omega_0, siren_c))
        self.spatial_net = nn.Sequential(*spatial_layers)
        
        weight_decay = weight_decay or 0.1
        self.params  = [{"params": self.spatial_net.parameters(), "weight_decay": weight_decay}]
        
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        spatial = self.spatial_net(self.get_coords().to(image.device))
        return spatial
    
    def get_coords(self) -> torch.Tensor:
        """Creates a coordinates grid."""
        coords = np.dstack(
            np.meshgrid(
                np.linspace(0, 1, self.down_size),
                np.linspace(0, 1, self.down_size)
            )
        )
        coords = torch.from_numpy(coords).float()
        return coords


class OutputINF(nn.Module):
    """Implicit Neural Function for merging patch and spatial information built
    on top of SIREN.
    
    References:
        https://github.com/ctom2/colie
    """
    
    def __init__(
        self,
        in_channels : int   = 256,
        out_channels: int   = 3,
        num_layers  : int   = 1,
        omega_0     : float = 30.0,
        siren_c     : float = 6.0,
        weight_decay: float = 0.001,
    ):
        super().__init__()
        output_layers = []
        for _ in range(0, num_layers):
            output_layers.append(SIREN(in_channels, in_channels, omega_0, siren_c))
        output_layers.append(SIREN(in_channels, out_channels, omega_0, siren_c, is_last=True))
        self.output_net = nn.Sequential(*output_layers)
        
        weight_decay = weight_decay or 0.001
        self.params  = [{"params": self.output_net.parameters(), "weight_decay": weight_decay}]
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.output_net(input)
    
# endregion

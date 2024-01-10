#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements FINet (Fractional Instance Normalization Network)
models.
"""

from __future__ import annotations

__all__ = [
    "FINet",
]

from typing import Any, Literal

import torch

from mon.globals import MODELS
from mon.vision import core, nn
from mon.vision.enhance.universal import base

console      = core.console
_current_dir = core.Path(__file__).absolute().parent


# region Module

class UNetConvBlock(nn.Module):

    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        p           : float,
        scheme      : Literal[
                        "half",
                        "bipartite",
                        "checkerboard",
                        "random",
                        "adaptive",
                        "attention",
                    ],
        downsample  : bool,
        relu_slope  : float,
        use_csff    : bool = False,
        use_fin     : bool = False,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.downsample = downsample
        self.use_csff   = use_csff
        self.p          = p
        self.scheme     = scheme

        self.identity   = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        self.conv_1     = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True)
        self.relu_1     = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2     = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=True)
        self.relu_2     = nn.LeakyReLU(relu_slope, inplace=False)

        if downsample and use_csff:
            self.csff_enc = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
            self.csff_dec = nn.Conv2d(out_channels, out_channels, 3, 1, 1)

        if use_fin:
            self.norm = nn.FractionalInstanceNorm2d(
                num_features = out_channels,
                p            = self.p,
                scheme       = scheme,
            )
        self.use_fin = use_fin

        if downsample:
            self.downsample = nn.Conv2d(out_channels, out_channels, 4, 2, 1, bias=False)

    def forward(
        self,
        input: torch.Tensor,
        enc  : torch.Tensor | None = None,
        dec  : torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        x = input
        y = self.conv_1(x)
        if self.use_fin:
            y = self.norm(y)
        y  = self.relu_1(y)
        y  = self.relu_2(self.conv_2(y))
        y += self.identity(x)
        if enc is not None and dec is not None:
            assert self.use_csff
            y = y + self.csff_enc(enc) + self.csff_dec(dec)
        if self.downsample:
            y_down = self.downsample(y)
            return y_down, y
        else:
            return y


class UNetUpBlock(nn.Module):

    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        relu_slope  : float,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.up = nn.ConvTranspose2d(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = 2,
            stride       = 2,
            bias         = True,
        )
        self.conv_block = UNetConvBlock(
            in_channels  = in_channels,
            out_channels = out_channels,
            p            = 0,
            scheme       = "half",
            downsample   = False,
            relu_slope   = relu_slope,
            use_fin      = False,
        )

    def forward(self, input: torch.Tensor, bridge: torch.Tensor) -> torch.Tensor:
        x = input
        y = self.up(x)
        y = torch.cat([y, bridge], 1)
        y = self.conv_block(y)
        return y

# endregion


# region Model

@MODELS.register(name="finet")
class FINet(base.UniversalImageEnhancementModel):
    """Fractional-Instance Normalization Network.
    
    See Also: :class:`mon.vision.enhance.universal.UniversalImageEnhancementModel`
    """
    
    configs     = {}
    zoo         = {}
    map_weights = {}

    def __init__(
        self,
        config      : Any                = None,
        variant     :         str | None = None,
        num_channels: int   | str        = 64,
        depth       : int   | str        = 5,
        relu_slope  : float | str        = 0.2,
        in_pos_left : int   | str        = 0,
        in_pos_right: int   | str        = 4,
        p           : float | str | None = 0.5,
        scheme      : Literal[
                        "half",
                        "bipartite",
                        "checkerboard",
                        "random",
                        "adaptive",
                        "attention",
                      ]                = "half",
        *args, **kwargs
    ):
        super().__init__(config=config, *args, **kwargs)
        variant           = core.to_int(variant)
        self.variant      = f"{variant:04d}" if isinstance(variant, int) else None
        self.num_channels = core.to_int(num_channels)  or 64
        self.depth        = core.to_int(depth)         or 5
        self.relu_slope   = core.to_float(relu_slope)  or 0.2
        self.in_pos_left  = core.to_int(in_pos_left)   or 0
        self.in_pos_right = core.to_int(in_pos_right)  or 4
        self.p            = core.to_float(p)           or 0.5
        self.scheme       = scheme                     or "half"

        if self.variant is None:  # Default model
            raise ValueError(f"``variant`` must be defined.")
        else:
            self.config_model_variant()

    def config_model_variant(self):
        """Config the model based on ``self.variant``.
        Mainly used in ablation study.
        """
        self.num_channels = 64
        self.depth        = 5
        self.relu_slope   = 0.2
        self.in_pos_left  = 0
        self.in_pos_right = 4
        self.p            = 0.5
        self.scheme       = "half"

        # Variant code: [aa][p][s]
        # p: probability
        if self.variant[2] == "0":
            self.p = 0.0
        elif self.variant[2] == "1":
            self.p = 0.1
        elif self.variant[2] == "2":
            self.p = 0.2
        elif self.variant[2] == "3":
            self.p = 0.3
        elif self.variant[2] == "4":
            self.p = 0.4
        elif self.variant[2] == "5":
            self.p = 0.5
        elif self.variant[2] == "6":
            self.p = 0.6
        elif self.variant[2] == "7":
            self.p = 0.7
        elif self.variant[2] == "8":
            self.p = 0.8
        elif self.variant[2] == "9":
            self.p = 0.9
        else:
            raise ValueError

        # Variant code: [aa][p][s]
        # s: scheme
        if self.variant[3] == "0":
            self.scheme = "half"
        elif self.variant[3] == "1":
            self.scheme = "bipartite"
        elif self.variant[3] == "2":
            self.scheme = "checkerboard"
        elif self.variant[3] == "3":
            self.scheme = "random"
        elif self.variant[3] == "4":
            self.scheme = "adaptive"
        elif self.variant[3] == "5":
            self.scheme = "attentive"

        # Variant code: [aa][p][s]
        # aa: architecture
        if self.variant[0:2] == "00":
            self.down_path_1 = nn.ModuleList()
            self.down_path_2 = nn.ModuleList()
            self.conv_01     = nn.Conv2d(self.channels, self.num_channels, 3, 1, 1)
            self.conv_02     = nn.Conv2d(self.channels, self.num_channels, 3, 1, 1)

            prev_channels    = self.num_channels
            for i in range(self.depth):  # 0,1,2,3,4
                use_fin    = True if self.in_pos_left <= i <= self.in_pos_right else False
                downsample = True if (i + 1) < self.depth else False
                self.down_path_1.append(UNetConvBlock(prev_channels, (2 ** i) * self.num_channels, self.p, self.scheme, downsample, self.relu_slope, use_fin=use_fin))
                self.down_path_2.append(UNetConvBlock(prev_channels, (2 ** i) * self.num_channels, self.p, self.scheme, downsample, self.relu_slope, use_csff=downsample, use_fin=use_fin))
                prev_channels = (2 ** i) * self.num_channels

            self.up_path_1   = nn.ModuleList()
            self.up_path_2   = nn.ModuleList()
            self.skip_conv_1 = nn.ModuleList()
            self.skip_conv_2 = nn.ModuleList()
            for i in reversed(range(self.depth - 1)):
                self.up_path_1.append(UNetUpBlock(prev_channels, (2 ** i) * self.num_channels, self.relu_slope))
                self.up_path_2.append(UNetUpBlock(prev_channels, (2 ** i) * self.num_channels, self.relu_slope))
                self.skip_conv_1.append(nn.Conv2d((2 ** i) * self.num_channels, (2 ** i) * self.num_channels, 3, 1, 1))
                self.skip_conv_2.append(nn.Conv2d((2 ** i) * self.num_channels, (2 ** i) * self.num_channels, 3, 1, 1))
                prev_channels = (2 ** i) * self.num_channels

            self.sam12 = nn.SAM(prev_channels)
            self.cat12 = nn.Conv2d(prev_channels * 2, prev_channels, 1, 1, 0)
            self.last  = nn.Conv2d(prev_channels, self.channels, 3, 1, 1, bias=True)

            self.apply(self.init_weights)
        else:
            raise ValueError

    def init_weights(self, m: nn.Module):
        gain      = torch.nn.init.calculate_gain('leaky_relu', 0.20)
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            if hasattr(m, "conv"):
                nn.init.orthogonal_(m.weight, gain=gain)
                if not m.bias is None:
                    nn.init.constant_(m.bias, 0)

    def forward_loss(
        self,
        input : torch.Tensor,
        target: torch.Tensor | None,
        *args, **kwargs
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass with loss value. Loss function may need more arguments
        beside the ground-truth and prediction values. For calculating the
        metrics, we only need the final predictions and ground-truth.

        Args:
            input: An input of shape :math:`[N, C, H, W]`.
            target: A ground-truth of shape :math:`[N, C, H, W]`. Default: ``None``.

        Return:
            Predictions and loss value.
        """
        pred = self.forward(input=input, *args, **kwargs)
        loss = self.loss(input, pred) if self.loss else (None, None)
        return pred, loss

    def forward_once(
        self,
        input    : torch.Tensor,
        profile  : bool = False,
        out_index: int = -1,
        *args, **kwargs
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass once. Implement the logic for a single forward pass.

        Args:
            input: An input of shape :math:`[N, C, H, W]`.
            profile: Measure processing time. Default: ``False``.
            out_index: Return specific layer's output from :param:`out_index`.
                Default: ``-1`` means the last layer.

        Return:
            Predictions.
        """
        x = input

        # Stage 1
        x1   = self.conv_01(x)
        encs = []
        decs = []
        for i, down in enumerate(self.down_path_1):
            if (i + 1) < self.depth:
                x1, x1_up = down(x1)
                encs.append(x1_up)
            else:
                x1 = down(x1)
        for i, up in enumerate(self.up_path_1):
            x1 = up(x1, self.skip_conv_1[i](encs[-i - 1]))
            decs.append(x1)

        # SAM
        sam_feats, y1 = self.sam12(input=[x1, x])

        # Stage 2
        x2     = self.conv_02(x)
        x2     = self.cat12(torch.cat([x2, sam_feats], dim=1))
        blocks = []
        for i, down in enumerate(self.down_path_2):
            if (i + 1) < self.depth:
                x2, x2_up = down(x2, encs[i], decs[-i - 1])
                blocks.append(x2_up)
            else:
                x2 = down(x2)
        for i, up in enumerate(self.up_path_2):
            x2 = up(x2, self.skip_conv_2[i](blocks[-i - 1]))

        y2 = self.last(x2)
        y2 = y2 + x
        return y2

# endregion

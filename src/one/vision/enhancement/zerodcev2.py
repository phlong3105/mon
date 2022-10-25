#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ZeroDCEv2
"""

from __future__ import annotations

from one.nn import *

CURRENT_DIR = Path(__file__).resolve().parent
CFG_DIR     = CURRENT_DIR / "cfg"


# H1: - Loss -------------------------------------------------------------------

class CombinedLoss(BaseLoss):
    """
    Loss = loss_spa + loss_exp + loss_col + loss_tv.
    """
    
    def __init__(
        self,
        spa_weight    : Floats = 1.0,
        exp_patch_size: int    = 16,
        exp_mean_val  : float  = 0.6,
        exp_weight    : Floats = 10.0,
        col_weight    : Floats = 5.0,
        tv_weight     : Floats = 200.0,
        channel_weight: Floats = 5.0,
        edge_weight   : Floats = 5.0,
        reduction     : str    = "mean",
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.name     = "combined_loss"
        self.loss_spa = SpatialConsistencyLoss(
            weight    = spa_weight,
            reduction = reduction,
        )
        self.loss_exp = ExposureControlLoss(
            patch_size = exp_patch_size,
            mean_val   = exp_mean_val,
            weight     = exp_weight,
            reduction  = reduction,
        )
        self.loss_col = ColorConstancyLoss(
            weight    = col_weight,
            reduction = reduction,
        )
        self.loss_tv  = IlluminationSmoothnessLoss(
            weight    = tv_weight,
            reduction = reduction,
        )
        self.loss_channel = ChannelConsistencyLoss(
            weight    = channel_weight,
            reduction = reduction,
        )
        self.loss_edge = EdgeLoss(
            weight    = edge_weight,
            reduction = reduction,
        )
     
    def forward(self, input: Tensors, target: Sequence[Tensor], **_) -> Tensor:
        import one.vision.transformation as t
        
        if isinstance(target, Sequence):
            a       = target[-2]
            enhance = target[-1]
            # print(enhance)
        else:
            raise TypeError()
        # input    = t.crop_tblr(input, 6, -6, 6, -6)
        loss_spa     = self.loss_spa(input=enhance, target=input)
        loss_exp     = self.loss_exp(input=enhance)
        loss_col     = self.loss_col(input=enhance)
        loss_tv      = self.loss_tv(input=a)
        loss_channel = self.loss_channel(input=enhance, target=input)
        loss_edge    = self.loss_edge(input=enhance, target=input)
        # print(float(loss_spa), float(loss_exp), float(loss_col), float(loss_tv))
        return loss_spa + loss_exp + loss_col + loss_tv + loss_channel + loss_edge


# H1: - Model ------------------------------------------------------------------

cfgs = {
    "zerodcev2-s1-tiny": {
        "channels": 3,
        "backbone": [
            # [from,  number, module,    args(out_channels, ...)]
            [-1,      1,      Identity,  []],                     # 0  (x)
            [-1,      1,      BSConv2dS, [16, 3, 1, 1]],          # 1
            [-1,      1,      ReLU,      [True]],                 # 2  (x1)
            [-1,      1,      BSConv2dS, [16, 3, 1, 1]],          # 3
            [-1,      1,      ReLU,      [True]],                 # 4  (x2)
            [-1,      1,      BSConv2dS, [16, 3, 1, 1]],          # 5
            [-1,      1,      ReLU,      [True]],                 # 6  (x3)
            [-1,      1,      BSConv2dS, [16, 3, 1, 1]],          # 7
            [-1,      1,      ReLU,      [True]],                 # 8  (x4)
            [[6,  8], 1,      Concat,    []],                     # 9
            [-1,      1,      BSConv2dS, [16, 3, 1, 1]],          # 10
            [-1,      1,      ReLU,      [True]],                 # 11 (x5)
            [[4, 11], 1,      Concat,    []],                     # 12
            [-1,      1,      BSConv2dS, [16, 3, 1, 1]],          # 13
            [-1,      1,      ReLU,      [True]],                 # 14 (x6)
            [[2, 14], 1,      Concat,    []],                     # 15
            [-1,      1,      BSConv2dS, [3,  3, 1, 1]],          # 16 (a)
            [-1,      1,      Tanh,      []],                     # 17
        ],
        "head": [
            [[-1, 0], 1,      PixelwiseHigherOrderLECurve, [8]],  # 18
        ]
    },
    "zerodcev2-s2-tiny": {
        "channels": 3,
        "backbone": [
            # [from,  number, module,     args(out_channels, ...)]
            [-1,      1,      Identity,   []],                                                                       # 0  (x)
            [-1,      1,      BSConv2dS,  [16, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, InstanceNorm2d]],  # 1
            [-1,      1,      ReLU,       [True]],                                                                   # 2  (x1)
            [-1,      1,      BSConv2dS,  [16, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, InstanceNorm2d]],  # 3
            [-1,      1,      ReLU,       [True]],                                                                   # 4  (x2)
            [-1,      1,      BSConv2dS,  [16, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, InstanceNorm2d]],  # 5
            [-1,      1,      ReLU,       [True]],                                                                   # 6  (x3)
            [-1,      1,      BSConv2dS,  [16, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, InstanceNorm2d]],  # 7
            [-1,      1,      ReLU,       [True]],                                                                   # 8  (x4)
            [[6, 8],  1,      Concat,     []],                                                                       # 9
            [-1,      1,      BSConv2dS,  [16, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, InstanceNorm2d]],  # 10
            [-1,      1,      ReLU,       [True]],                                                                   # 11 (x5)
            [[4, 11], 1,      Concat,     []],                                                                       # 12
            [-1,      1,      BSConv2dS,  [16, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, InstanceNorm2d]],  # 13
            [-1,      1,      ReLU,       [True]],                                                                   # 14 (x6)
            [[2, 14], 1,      Concat,     []],                                                                       # 15
            [-1,      1,      BSConv2dS,  [3,  3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, InstanceNorm2d]],  # 16 (a)
            [-1,      1,      Tanh,       []],                                                                       # 17
        ],
        "head": [
            [[-1, 0], 1,      PixelwiseHigherOrderLECurve, [8]],                                                     # 18
        ]
    },
    "zerodcev2-s3-tiny": {
        "channels": 3,
        "backbone": [
            # [from,  number, module,     args(out_channels, ...)]
            [-1,      1,      Identity,   []],                                                                           # 0  (x)
            [-1,      1,      BSConv2dS,  [16, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d]],  # 1
            [-1,      1,      ReLU,       [True]],                                                                       # 2  (x1)
            [-1,      1,      BSConv2dS,  [16, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d]],  # 3
            [-1,      1,      ReLU,       [True]],                                                                       # 4  (x2)
            [-1,      1,      BSConv2dS,  [16, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d]],  # 5
            [-1,      1,      ReLU,       [True]],                                                                       # 6  (x3)
            [-1,      1,      BSConv2dS,  [16, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d]],  # 7
            [-1,      1,      ReLU,       [True]],                                                                       # 8  (x4)
            [[6, 8],  1,      Concat,     []],                                                                           # 9
            [-1,      1,      BSConv2dS,  [16, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d]],  # 10
            [-1,      1,      ReLU,       [True]],                                                                       # 11 (x5)
            [[4, 11], 1,      Concat,     []],                                                                           # 12
            [-1,      1,      BSConv2dS,  [16, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d]],  # 13
            [-1,      1,      ReLU,       [True]],                                                                       # 14 (x6)
            [[2, 14], 1,      Concat,     []],                                                                           # 15
            [-1,      1,      BSConv2dS,  [3,  3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d]],  # 16 (a)
            [-1,      1,      Tanh,       []],                                                                           # 17
        ],
        "head": [
            [[-1, 0], 1,      PixelwiseHigherOrderLECurve, [8]],                                                         # 18
        ]
    },
    "zerodcev2-s4-tiny": {
        "channels": 3,
        "backbone": [
            # [from,  number, module,     args(out_channels, ...)]
            [-1,      1,      Identity,   []],                                                                           # 0  (x)
            [-1,      1,      ABSConv2dS, [16, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d]],  # 1
            [-1,      1,      ReLU,       [True]],                                                                       # 2  (x1)
            [-1,      1,      ABSConv2dS, [16, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d]],  # 3
            [-1,      1,      ReLU,       [True]],                                                                       # 4  (x2)
            [-1,      1,      ABSConv2dS, [16, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d]],  # 5
            [-1,      1,      ReLU,       [True]],                                                                       # 6  (x3)
            [-1,      1,      ABSConv2dS, [16, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d]],  # 7
            [-1,      1,      ReLU,       [True]],                                                                       # 8  (x4)
            [[6, 8],  1,      Concat,     []],                                                                           # 9
            [-1,      1,      ABSConv2dS, [16, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d]],  # 10
            [-1,      1,      ReLU,       [True]],                                                                       # 11 (x5)
            [[4, 11], 1,      Concat,     []],                                                                           # 12
            [-1,      1,      ABSConv2dS, [16, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d]],  # 13
            [-1,      1,      ReLU,       [True]],                                                                       # 14 (x6)
            [[2, 14], 1,      Concat,     []],                                                                           # 15
            [-1,      1,      ABSConv2dS, [3,  3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d]],  # 16 (a)
            [-1,      1,      Tanh,       []],                                                                           # 17
        ],
        "head": [
            [[-1, 0], 1,      PixelwiseHigherOrderLECurve, [8]],                                                         # 18
        ]
    },
    "zerodcev2-s5-tiny": {
        "channels": 3,
        "backbone": [
            # [from,  number, module,   args(out_channels, ...)]
            [-1,      1,      Identity, []],                                                    # 0  (x)
            [-1,      1,      DCEV2,    [3, 16, partial(ABSConv2dS, act=HalfInstanceNorm2d)]],  # 1
        ],
        "head": [
            [[-1, 0], 1,      PixelwiseHigherOrderLECurve, [8]],                                # 2
        ]
    },
    "zerodcev2-s6-tiny": {
        "channels": 3,
        "backbone": [
            # [from,  number, module,                      args(out_channels, ...)]
            [-1,      1,      Identity,                    []],                                                                           # 0  (x)
            [-1,      1,      ABSConv2dS,                  [32, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d]],  # 1
            [-1,      1,      ReLU,                        [True]],                                                                       # 2  (x1)
            [-1,      1,      ABSConv2dS,                  [16, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d]],  # 3
            [-1,      1,      ReLU,                        [True]],                                                                       # 4  (x2)
            [-1,      1,      ABSConv2dS,                  [16, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d]],  # 5
            [-1,      1,      ReLU,                        [True]],                                                                       # 6  (x3)
            [-1,      1,      ABSConv2dS,                  [16, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d]],  # 7
            [-1,      1,      ReLU,                        [True]],                                                                       # 8  (x4)
            [[6, 8],  1,      Concat,                      []],                                                                           # 9
            [-1,      1,      ABSConv2dS,                  [16, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d]],  # 10
            [-1,      1,      ReLU,                        [True]],                                                                       # 11 (x5)
            [[4, 11], 1,      Concat,                      []],                                                                           # 12
            [-1,      1,      ABSConv2dS,                  [16, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d]],  # 13
            [-1,      1,      ReLU,                        [True]],                                                                       # 14 (x6)
            [[2, 14], 1,      Concat,                      []],                                                                           # 15
            [-1,      1,      ABSConv2dS,                  [16, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d]],  # 16
            [-1,      1,      ReLU,                        [True]],                                                                       # 17 (x7)
            [ 2,      1,      ExtractFeatures,             [0, 16]],                                                                      # 18
            [[2, 17], 1,      Concat,                      []],                                                                           # 19
            [-1,      1,      DepthwiseSeparableConv2d,    [3,  3, 1, 1, 1, 1, 0]],                                                       # 20 (a)
            [-1,      1,      Tanh,                        []],                                                                           # 21
        ],
        "head": [
            [[-1, 0], 1,      PixelwiseHigherOrderLECurve, [8]],                                                                          # 22
        ]
    },
    
    "zerodcev2-s1": {
        "channels": 3,
        "backbone": [
            # [from,  number, module,    args(out_channels, ...)]
            [-1,      1,      Identity,  []],                     # 0  (x)
            [-1,      1,      BSConv2dS, [32, 3, 1, 1]],          # 1
            [-1,      1,      ReLU,      [True]],                 # 2  (x1)
            [-1,      1,      BSConv2dS, [32, 3, 1, 1]],          # 3
            [-1,      1,      ReLU,      [True]],                 # 4  (x2)
            [-1,      1,      BSConv2dS, [32, 3, 1, 1]],          # 5
            [-1,      1,      ReLU,      [True]],                 # 6  (x3)
            [-1,      1,      BSConv2dS, [32, 3, 1, 1]],          # 7
            [-1,      1,      ReLU,      [True]],                 # 8  (x4)
            [[6,  8], 1,      Concat,    []],                     # 9
            [-1,      1,      BSConv2dS, [32, 3, 1, 1]],          # 10
            [-1,      1,      ReLU,      [True]],                 # 11 (x5)
            [[4, 11], 1,      Concat,    []],                     # 12
            [-1,      1,      BSConv2dS, [32, 3, 1, 1]],          # 13
            [-1,      1,      ReLU,      [True]],                 # 14 (x6)
            [[2, 14], 1,      Concat,    []],                     # 15
            [-1,      1,      BSConv2dS, [3,  3, 1, 1]],          # 16 (a)
            [-1,      1,      Tanh,      []],                     # 17
        ],
        "head": [
            [[-1, 0], 1,      PixelwiseHigherOrderLECurve, [8]],  # 18
        ]
    },
    "zerodcev2-s2": {
        "channels": 3,
        "backbone": [
            # [from,  number, module,     args(out_channels, ...)]
            [-1,      1,      Identity,   []],                                                                       # 0  (x)
            [-1,      1,      BSConv2dS,  [32, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, InstanceNorm2d]],  # 1
            [-1,      1,      ReLU,       [True]],                                                                   # 2  (x1)
            [-1,      1,      BSConv2dS,  [32, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, InstanceNorm2d]],  # 3
            [-1,      1,      ReLU,       [True]],                                                                   # 4  (x2)
            [-1,      1,      BSConv2dS,  [32, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, InstanceNorm2d]],  # 5
            [-1,      1,      ReLU,       [True]],                                                                   # 6  (x3)
            [-1,      1,      BSConv2dS,  [32, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, InstanceNorm2d]],  # 7
            [-1,      1,      ReLU,       [True]],                                                                   # 8  (x4)
            [[6, 8],  1,      Concat,     []],                                                                       # 9
            [-1,      1,      BSConv2dS,  [32, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, InstanceNorm2d]],  # 10
            [-1,      1,      ReLU,       [True]],                                                                   # 11 (x5)
            [[4, 11], 1,      Concat,     []],                                                                       # 12
            [-1,      1,      BSConv2dS,  [32, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, InstanceNorm2d]],  # 13
            [-1,      1,      ReLU,       [True]],                                                                   # 14 (x6)
            [[2, 14], 1,      Concat,     []],                                                                       # 15
            [-1,      1,      BSConv2dS,  [3,  3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, InstanceNorm2d]],  # 16 (a)
            [-1,      1,      Tanh,       []],                                                                       # 17
        ],
        "head": [
            [[-1, 0], 1,      PixelwiseHigherOrderLECurve, [8]],                                                     # 18
        ]
    },
    "zerodcev2-s3": {
        "channels": 3,
        "backbone": [
            # [from,  number, module,     args(out_channels, ...)]
            [-1,      1,      Identity,   []],                                                                           # 0  (x)
            [-1,      1,      BSConv2dS,  [32, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d]],  # 1
            [-1,      1,      ReLU,       [True]],                                                                       # 2  (x1)
            [-1,      1,      BSConv2dS,  [32, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d]],  # 3
            [-1,      1,      ReLU,       [True]],                                                                       # 4  (x2)
            [-1,      1,      BSConv2dS,  [32, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d]],  # 5
            [-1,      1,      ReLU,       [True]],                                                                       # 6  (x3)
            [-1,      1,      BSConv2dS,  [32, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d]],  # 7
            [-1,      1,      ReLU,       [True]],                                                                       # 8  (x4)
            [[6, 8],  1,      Concat,     []],                                                                           # 9
            [-1,      1,      BSConv2dS,  [32, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d]],  # 10
            [-1,      1,      ReLU,       [True]],                                                                       # 11 (x5)
            [[4, 11], 1,      Concat,     []],                                                                           # 12
            [-1,      1,      BSConv2dS,  [32, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d]],  # 13
            [-1,      1,      ReLU,       [True]],                                                                       # 14 (x6)
            [[2, 14], 1,      Concat,     []],                                                                           # 15
            [-1,      1,      BSConv2dS,  [3,  3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d]],  # 16 (a)
            [-1,      1,      Tanh,       []],                                                                           # 17
        ],
        "head": [
            [[-1, 0], 1,      PixelwiseHigherOrderLECurve, [8]],                                                         # 18
        ]
    },
    "zerodcev2-s4": {
        "channels": 3,
        "backbone": [
            # [from,  number, module,     args(out_channels, ...)]
            [-1,      1,      Identity,   []],                                                                           # 0  (x)
            [-1,      1,      ABSConv2dS, [32, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d]],  # 1
            [-1,      1,      ReLU,       [True]],                                                                       # 2  (x1)
            [-1,      1,      ABSConv2dS, [32, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d]],  # 3
            [-1,      1,      ReLU,       [True]],                                                                       # 4  (x2)
            [-1,      1,      ABSConv2dS, [32, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d]],  # 5
            [-1,      1,      ReLU,       [True]],                                                                       # 6  (x3)
            [-1,      1,      ABSConv2dS, [32, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d]],  # 7
            [-1,      1,      ReLU,       [True]],                                                                       # 8  (x4)
            [[6, 8],  1,      Concat,     []],                                                                           # 9
            [-1,      1,      ABSConv2dS, [32, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d]],  # 10
            [-1,      1,      ReLU,       [True]],                                                                       # 11 (x5)
            [[4, 11], 1,      Concat,     []],                                                                           # 12
            [-1,      1,      ABSConv2dS, [32, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d]],  # 13
            [-1,      1,      ReLU,       [True]],                                                                       # 14 (x6)
            [[2, 14], 1,      Concat,     []],                                                                           # 15
            [-1,      1,      ABSConv2dS, [3,  3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d]],  # 16 (a)
            [-1,      1,      Tanh,       []],                                                                           # 17
        ],
        "head": [
            [[-1, 0], 1,      PixelwiseHigherOrderLECurve, [8]],                                                         # 18
        ]
    },
    "zerodcev2-s5": {
        "channels": 3,
        "backbone": [
            # [from,  number, module,   args(out_channels, ...)]
            [-1,      1,      Identity, []],                                                    # 0  (x)
            [-1,      1,      DCEV2,    [3, 32, partial(ABSConv2dS, act=HalfInstanceNorm2d)]],  # 1
        ],
        "head": [
            [[-1, 0], 1,      PixelwiseHigherOrderLECurve, [8]],                                # 2
        ]
    },
    "zerodcev2-s6": {
        "channels": 3,
        "backbone": [
            # [from,  number, module,                      args(out_channels, ...)]
            [-1,      1,      Identity,                    []],                                                                           # 0  (x)
            [-1,      1,      ABSConv2dS,                  [32, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d]],  # 1
            [-1,      1,      ReLU,                        [True]],                                                                       # 2  (x1)
            [-1,      1,      ABSConv2dS,                  [32, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d]],  # 3
            [-1,      1,      ReLU,                        [True]],                                                                       # 4  (x2)
            [-1,      1,      ABSConv2dS,                  [32, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d]],  # 5
            [-1,      1,      ReLU,                        [True]],                                                                       # 6  (x3)
            [-1,      1,      ABSConv2dS,                  [32, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d]],  # 7
            [-1,      1,      ReLU,                        [True]],                                                                       # 8  (x4)
            [[6, 8],  1,      Concat,                      []],                                                                           # 9
            [-1,      1,      ABSConv2dS,                  [32, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d]],  # 10
            [-1,      1,      ReLU,                        [True]],                                                                       # 11 (x5)
            [[4, 11], 1,      Concat,                      []],                                                                           # 12
            [-1,      1,      ABSConv2dS,                  [32, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d]],  # 13
            [-1,      1,      ReLU,                        [True]],                                                                       # 14 (x6)
            [[2, 14], 1,      Concat,                      []],                                                                           # 15
            [-1,      1,      ABSConv2dS,                  [32, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d]],  # 16
            [-1,      1,      ReLU,                        [True]],                                                                       # 17 (x7)
            [ 2,      1,      ExtractFeatures,             [0, 16]],                                                                      # 18
            [[2, 17], 1,      Concat,                      []],                                                                           # 19
            [-1,      1,      DepthwiseSeparableConv2d,    [3,  3, 1, 1, 1, 1, 0]],                                                       # 20 (a)
            [-1,      1,      Tanh,                        []],                                                                           # 21
        ],
        "head": [
            [[-1, 0], 1,      PixelwiseHigherOrderLECurve, [8]],                                                                          # 22
        ]
    },
    
    "zerodcev2-u1-tiny": {
        "channels": 3,
        "backbone": [
            # [from,  number, module,    args(out_channels, ...)]
            [-1,      1,      Identity,  []],                     # 0  (x)
            [-1,      1,      BSConv2dU, [16, 3, 1, 1]],          # 1
            [-1,      1,      ReLU,      [True]],                 # 2  (x1)
            [-1,      1,      BSConv2dU, [16, 3, 1, 1]],          # 3
            [-1,      1,      ReLU,      [True]],                 # 4  (x2)
            [-1,      1,      BSConv2dU, [16, 3, 1, 1]],          # 5
            [-1,      1,      ReLU,      [True]],                 # 6  (x3)
            [-1,      1,      BSConv2dU, [16, 3, 1, 1]],          # 7
            [-1,      1,      ReLU,      [True]],                 # 8  (x4)
            [[6,  8], 1,      Concat,    []],                     # 9
            [-1,      1,      BSConv2dU, [16, 3, 1, 1]],          # 10
            [-1,      1,      ReLU,      [True]],                 # 11 (x5)
            [[4, 11], 1,      Concat,    []],                     # 12
            [-1,      1,      BSConv2dU, [16, 3, 1, 1]],          # 13
            [-1,      1,      ReLU,      [True]],                 # 14 (x6)
            [[2, 14], 1,      Concat,    []],                     # 15
            [-1,      1,      BSConv2dU, [3,  3, 1, 1]],          # 16 (a)
            [-1,      1,      Tanh,      []],                     # 17
        ],
        "head": [
            [[-1, 0], 1,      PixelwiseHigherOrderLECurve, [8]],  # 18
        ]
    },
    "zerodcev2-u2-tiny": {
        "channels": 3,
        "backbone": [
            # [from,  number, module,     args(out_channels, ...)]
            [-1,      1,      Identity,   []],                                                                       # 0  (x)
            [-1,      1,      BSConv2dU,  [16, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, InstanceNorm2d]],  # 1
            [-1,      1,      ReLU,       [True]],                                                                   # 2  (x1)
            [-1,      1,      BSConv2dU,  [16, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, InstanceNorm2d]],  # 3
            [-1,      1,      ReLU,       [True]],                                                                   # 4  (x2)
            [-1,      1,      BSConv2dU,  [16, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, InstanceNorm2d]],  # 5
            [-1,      1,      ReLU,       [True]],                                                                   # 6  (x3)
            [-1,      1,      BSConv2dU,  [16, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, InstanceNorm2d]],  # 7
            [-1,      1,      ReLU,       [True]],                                                                   # 8  (x4)
            [[6, 8],  1,      Concat,     []],                                                                       # 9
            [-1,      1,      BSConv2dU,  [16, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, InstanceNorm2d]],  # 10
            [-1,      1,      ReLU,       [True]],                                                                   # 11 (x5)
            [[4, 11], 1,      Concat,     []],                                                                       # 12
            [-1,      1,      BSConv2dU,  [16, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, InstanceNorm2d]],  # 13
            [-1,      1,      ReLU,       [True]],                                                                   # 14 (x6)
            [[2, 14], 1,      Concat,     []],                                                                       # 15
            [-1,      1,      BSConv2dU,  [3,  3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, InstanceNorm2d]],  # 16 (a)
            [-1,      1,      Tanh,       []],                                                                       # 17
        ],
        "head": [
            [[-1, 0], 1,      PixelwiseHigherOrderLECurve, [8]],                                                     # 18
        ]
    },
    "zerodcev2-u3-tiny": {
        "channels": 3,
        "backbone": [
            # [from,  number, module,     args(out_channels, ...)]
            [-1,      1,      Identity,   []],                                                                           # 0  (x)
            [-1,      1,      BSConv2dU,  [16, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d]],  # 1
            [-1,      1,      ReLU,       [True]],                                                                       # 2  (x1)
            [-1,      1,      BSConv2dU,  [16, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d]],  # 3
            [-1,      1,      ReLU,       [True]],                                                                       # 4  (x2)
            [-1,      1,      BSConv2dU,  [16, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d]],  # 5
            [-1,      1,      ReLU,       [True]],                                                                       # 6  (x3)
            [-1,      1,      BSConv2dU,  [16, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d]],  # 7
            [-1,      1,      ReLU,       [True]],                                                                       # 8  (x4)
            [[6, 8],  1,      Concat,     []],                                                                           # 9
            [-1,      1,      BSConv2dU,  [16, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d]],  # 10
            [-1,      1,      ReLU,       [True]],                                                                       # 11 (x5)
            [[4, 11], 1,      Concat,     []],                                                                           # 12
            [-1,      1,      BSConv2dU,  [16, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d]],  # 13
            [-1,      1,      ReLU,       [True]],                                                                       # 14 (x6)
            [[2, 14], 1,      Concat,     []],                                                                           # 15
            [-1,      1,      BSConv2dU,  [3,  3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d]],  # 16 (a)
            [-1,      1,      Tanh,       []],                                                                           # 17
        ],
        "head": [
            [[-1, 0], 1,      PixelwiseHigherOrderLECurve, [8]],                                                         # 18
        ]
    },
    "zerodcev2-u4-tiny": {
        "channels": 3,
        "backbone": [
            # [from,  number, module,     args(out_channels, ...)]
            [-1,      1,      Identity,   []],                                                                           # 0  (x)
            [-1,      1,      ABSConv2dU, [16, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d]],  # 1
            [-1,      1,      ReLU,       [True]],                                                                       # 2  (x1)
            [-1,      1,      ABSConv2dU, [16, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d]],  # 3
            [-1,      1,      ReLU,       [True]],                                                                       # 4  (x2)
            [-1,      1,      ABSConv2dU, [16, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d]],  # 5
            [-1,      1,      ReLU,       [True]],                                                                       # 6  (x3)
            [-1,      1,      ABSConv2dU, [16, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d]],  # 7
            [-1,      1,      ReLU,       [True]],                                                                       # 8  (x4)
            [[6, 8],  1,      Concat,     []],                                                                           # 9
            [-1,      1,      ABSConv2dU, [16, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d]],  # 10
            [-1,      1,      ReLU,       [True]],                                                                       # 11 (x5)
            [[4, 11], 1,      Concat,     []],                                                                           # 12
            [-1,      1,      ABSConv2dU, [16, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d]],  # 13
            [-1,      1,      ReLU,       [True]],                                                                       # 14 (x6)
            [[2, 14], 1,      Concat,     []],                                                                           # 15
            [-1,      1,      ABSConv2dU, [3,  3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d]],  # 16 (a)
            [-1,      1,      Tanh,       []],                                                                           # 17
        ],
        "head": [
            [[-1, 0], 1,      PixelwiseHigherOrderLECurve, [8]],                                                         # 18
        ]
    },
    "zerodcev2-u5-tiny": {
        "channels": 3,
        "backbone": [
            # [from,  number, module,   args(out_channels, ...)]
            [-1,      1,      Identity, []],                                                    # 0  (x)
            [-1,      1,      DCEV2,    [3, 16, partial(ABSConv2dU, act=HalfInstanceNorm2d)]],  # 1
        ],
        "head": [
            [[-1, 0], 1,      PixelwiseHigherOrderLECurve, [8]],                                # 2
        ]
    },
    "zerodcev2-u6-tiny": {
        "channels": 3,
        "backbone": [
            # [from,  number, module,                      args(out_channels, ...)]
            [-1,      1,      Identity,                    []],                                                                           # 0  (x)
            [-1,      1,      ABSConv2dU,                  [32, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d]],  # 1
            [-1,      1,      ReLU,                        [True]],                                                                       # 2  (x1)
            [-1,      1,      ABSConv2dU,                  [16, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d]],  # 3
            [-1,      1,      ReLU,                        [True]],                                                                       # 4  (x2)
            [-1,      1,      ABSConv2dU,                  [16, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d]],  # 5
            [-1,      1,      ReLU,                        [True]],                                                                       # 6  (x3)
            [-1,      1,      ABSConv2dU,                  [16, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d]],  # 7
            [-1,      1,      ReLU,                        [True]],                                                                       # 8  (x4)
            [[6, 8],  1,      Concat,                      []],                                                                           # 9
            [-1,      1,      ABSConv2dU,                  [16, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d]],  # 10
            [-1,      1,      ReLU,                        [True]],                                                                       # 11 (x5)
            [[4, 11], 1,      Concat,                      []],                                                                           # 12
            [-1,      1,      ABSConv2dU,                  [16, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d]],  # 13
            [-1,      1,      ReLU,                        [True]],                                                                       # 14 (x6)
            [[2, 14], 1,      Concat,                      []],                                                                           # 15
            [-1,      1,      ABSConv2dU,                  [16, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d]],  # 16
            [-1,      1,      ReLU,                        [True]],                                                                       # 17 (x7)
            [ 2,      1,      ExtractFeatures,             [0, 16]],                                                                      # 18
            [[2, 17], 1,      Concat,                      []],                                                                           # 19
            [-1,      1,      DepthwiseSeparableConv2d,    [3,  3, 1, 1, 1, 1, 0]],                                                       # 20 (a)
            [-1,      1,      Tanh,                        []],                                                                           # 21
        ],
        "head": [
            [[-1, 0], 1,      PixelwiseHigherOrderLECurve, [8]],                                                                          # 22
        ]
    },
    
    "zerodcev2-u1": {
        "channels": 3,
        "backbone": [
            # [from,  number, module,    args(out_channels, ...)]
            [-1,      1,      Identity,  []],                     # 0  (x)
            [-1,      1,      BSConv2dU, [32, 3, 1, 1]],          # 1
            [-1,      1,      ReLU,      [True]],                 # 2  (x1)
            [-1,      1,      BSConv2dU, [32, 3, 1, 1]],          # 3
            [-1,      1,      ReLU,      [True]],                 # 4  (x2)
            [-1,      1,      BSConv2dU, [32, 3, 1, 1]],          # 5
            [-1,      1,      ReLU,      [True]],                 # 6  (x3)
            [-1,      1,      BSConv2dU, [32, 3, 1, 1]],          # 7
            [-1,      1,      ReLU,      [True]],                 # 8  (x4)
            [[6,  8], 1,      Concat,    []],                     # 9
            [-1,      1,      BSConv2dU, [32, 3, 1, 1]],          # 10
            [-1,      1,      ReLU,      [True]],                 # 11 (x5)
            [[4, 11], 1,      Concat,    []],                     # 12
            [-1,      1,      BSConv2dU, [32, 3, 1, 1]],          # 13
            [-1,      1,      ReLU,      [True]],                 # 14 (x6)
            [[2, 14], 1,      Concat,    []],                     # 15
            [-1,      1,      BSConv2dU, [3,  3, 1, 1]],          # 16 (a)
            [-1,      1,      Tanh,      []],                     # 17
        ],
        "head": [
            [[-1, 0], 1,      PixelwiseHigherOrderLECurve, [8]],  # 18
        ]
    },
    "zerodcev2-u2": {
        "channels": 3,
        "backbone": [
            # [from,  number, module,     args(out_channels, ...)]
            [-1,      1,      Identity,   []],                                                                       # 0  (x)
            [-1,      1,      BSConv2dU,  [32, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, InstanceNorm2d]],  # 1
            [-1,      1,      ReLU,       [True]],                                                                   # 2  (x1)
            [-1,      1,      BSConv2dU,  [32, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, InstanceNorm2d]],  # 3
            [-1,      1,      ReLU,       [True]],                                                                   # 4  (x2)
            [-1,      1,      BSConv2dU,  [32, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, InstanceNorm2d]],  # 5
            [-1,      1,      ReLU,       [True]],                                                                   # 6  (x3)
            [-1,      1,      BSConv2dU,  [32, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, InstanceNorm2d]],  # 7
            [-1,      1,      ReLU,       [True]],                                                                   # 8  (x4)
            [[6, 8],  1,      Concat,     []],                                                                       # 9
            [-1,      1,      BSConv2dU,  [32, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, InstanceNorm2d]],  # 10
            [-1,      1,      ReLU,       [True]],                                                                   # 11 (x5)
            [[4, 11], 1,      Concat,     []],                                                                       # 12
            [-1,      1,      BSConv2dU,  [32, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, InstanceNorm2d]],  # 13
            [-1,      1,      ReLU,       [True]],                                                                   # 14 (x6)
            [[2, 14], 1,      Concat,     []],                                                                       # 15
            [-1,      1,      BSConv2dU,  [3,  3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, InstanceNorm2d]],  # 16 (a)
            [-1,      1,      Tanh,       []],                                                                       # 17
        ],
        "head": [
            [[-1, 0], 1,      PixelwiseHigherOrderLECurve, [8]],                                                     # 18
        ]
    },
    "zerodcev2-u3": {
        "channels": 3,
        "backbone": [
            # [from,  number, module,     args(out_channels, ...)]
            [-1,      1,      Identity,   []],                                                                           # 0  (x)
            [-1,      1,      BSConv2dU,  [32, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d]],  # 1
            [-1,      1,      ReLU,       [True]],                                                                       # 2  (x1)
            [-1,      1,      BSConv2dU,  [32, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d]],  # 3
            [-1,      1,      ReLU,       [True]],                                                                       # 4  (x2)
            [-1,      1,      BSConv2dU,  [32, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d]],  # 5
            [-1,      1,      ReLU,       [True]],                                                                       # 6  (x3)
            [-1,      1,      BSConv2dU,  [32, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d]],  # 7
            [-1,      1,      ReLU,       [True]],                                                                       # 8  (x4)
            [[6, 8],  1,      Concat,     []],                                                                           # 9
            [-1,      1,      BSConv2dU,  [32, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d]],  # 10
            [-1,      1,      ReLU,       [True]],                                                                       # 11 (x5)
            [[4, 11], 1,      Concat,     []],                                                                           # 12
            [-1,      1,      BSConv2dU,  [32, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d]],  # 13
            [-1,      1,      ReLU,       [True]],                                                                       # 14 (x6)
            [[2, 14], 1,      Concat,     []],                                                                           # 15
            [-1,      1,      BSConv2dU,  [3,  3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d]],  # 16 (a)
            [-1,      1,      Tanh,       []],                                                                           # 17
        ],
        "head": [
            [[-1, 0], 1,      PixelwiseHigherOrderLECurve, [8]],                                                         # 18
        ]
    },
    "zerodcev2-u4": {
        "channels": 3,
        "backbone": [
            # [from,  number, module,     args(out_channels, ...)]
            [-1,      1,      Identity,   []],                                                                           # 0  (x)
            [-1,      1,      ABSConv2dU, [32, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d]],  # 1
            [-1,      1,      ReLU,       [True]],                                                                       # 2  (x1)
            [-1,      1,      ABSConv2dU, [32, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d]],  # 3
            [-1,      1,      ReLU,       [True]],                                                                       # 4  (x2)
            [-1,      1,      ABSConv2dU, [32, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d]],  # 5
            [-1,      1,      ReLU,       [True]],                                                                       # 6  (x3)
            [-1,      1,      ABSConv2dU, [32, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d]],  # 7
            [-1,      1,      ReLU,       [True]],                                                                       # 8  (x4)
            [[6, 8],  1,      Concat,     []],                                                                           # 9
            [-1,      1,      ABSConv2dU, [32, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d]],  # 10
            [-1,      1,      ReLU,       [True]],                                                                       # 11 (x5)
            [[4, 11], 1,      Concat,     []],                                                                           # 12
            [-1,      1,      ABSConv2dU, [32, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d]],  # 13
            [-1,      1,      ReLU,       [True]],                                                                       # 14 (x6)
            [[2, 14], 1,      Concat,     []],                                                                           # 15
            [-1,      1,      ABSConv2dU, [3,  3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d]],  # 16 (a)
            [-1,      1,      Tanh,       []],                                                                           # 17
        ],
        "head": [
            [[-1, 0], 1,      PixelwiseHigherOrderLECurve, [8]],                                                         # 18
        ]
    },
    "zerodcev2-u5": {
        "channels": 3,
        "backbone": [
            # [from,  number, module,   args(out_channels, ...)]
            [-1,      1,      Identity, []],                                                    # 0  (x)
            [-1,      1,      DCEV2,    [3, 32, partial(ABSConv2dU, act=HalfInstanceNorm2d)]],  # 1
        ],
        "head": [
            [[-1, 0], 1,      PixelwiseHigherOrderLECurve, [8]],                                # 2
        ]
    },
    "zerodcev2-u6": {
        "channels": 3,
        "backbone": [
            # [from,  number, module,                      args(out_channels, ...)]
            [-1,      1,      Identity,                    []],                                                                           # 0  (x)
            [-1,      1,      ABSConv2dU,                  [32, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d]],  # 1
            [-1,      1,      ReLU,                        [True]],                                                                       # 2  (x1)
            [-1,      1,      ABSConv2dU,                  [32, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d]],  # 3
            [-1,      1,      ReLU,                        [True]],                                                                       # 4  (x2)
            [-1,      1,      ABSConv2dU,                  [32, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d]],  # 5
            [-1,      1,      ReLU,                        [True]],                                                                       # 6  (x3)
            [-1,      1,      ABSConv2dU,                  [32, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d]],  # 7
            [-1,      1,      ReLU,                        [True]],                                                                       # 8  (x4)
            [[6, 8],  1,      Concat,                      []],                                                                           # 9
            [-1,      1,      ABSConv2dU,                  [32, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d]],  # 10
            [-1,      1,      ReLU,                        [True]],                                                                       # 11 (x5)
            [[4, 11], 1,      Concat,                      []],                                                                           # 12
            [-1,      1,      ABSConv2dU,                  [32, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d]],  # 13
            [-1,      1,      ReLU,                        [True]],                                                                       # 14 (x6)
            [[2, 14], 1,      Concat,                      []],                                                                           # 15
            [-1,      1,      ABSConv2dU,                  [32, 3, 1, 1, 1, 1, True, "zeros", None, None, 0.25, 4, HalfInstanceNorm2d]],  # 16
            [-1,      1,      ReLU,                        [True]],                                                                       # 17 (x7)
            [ 2,      1,      ExtractFeatures,             [0, 16]],                                                                      # 18
            [[2, 17], 1,      Concat,                      []],                                                                           # 19
            [-1,      1,      DepthwiseSeparableConv2d,    [3,  3, 1, 1, 1, 1, 0]],                                                       # 20 (a)
            [-1,      1,      Tanh,                        []],                                                                           # 21
        ],
        "head": [
            [[-1, 0], 1,      PixelwiseHigherOrderLECurve, [8]],                                                                          # 22
        ]
    },
    
    "zerodcev2-t1": {
        "channels": 3,
        "backbone": [
            # [from,        number, module,                      args(out_channels, ...)]
            [-1,            1,      Identity,                    []],                                                   # 0  (x)
            # Branch 1: Light
            [-1,            1,      DCEV2,                       [3, 32, partial(BSConv2dS, act=HalfInstanceNorm2d)]],  # 1
            [[1, 0],        1,      PixelwiseHigherOrderLECurve, [8]],                                                  # 2
            [2,             1,      ExtractItem,                 [0]],                                                  # 3
            [2,             1,      ExtractItem,                 [1]],                                                  # 4
            [3,             1,      CropTBLR,                    [6, -6, 6, -6]],                                       # 5
            [4,             1,      CropTBLR,                    [6, -6, 6, -6]],                                       # 6
            # Branch 2: Details
            [4,             1,      VDSR,                        [3]],                                                  # 7
            [-1,            1,      CropTBLR,                    [6, -6, 6, -6]],                                       # 8
            # Branch 3: Structures
            [4,             1,      SRCNN,                       [3, 9, 1, 0, 1, 1, 0, 5, 1, 0]],                       # 9
        ],                                                                      
        "head": [
            [[7, 4],        1,      Sum,                         []],                                                   # 10
            [[7, 9, 3, 10], 1,      Join,                        []],                                                   # 11 (detail, structure, a, output)
        ]
    },
}


@MODELS.register(name="zerodcev2")
class ZeroDCEV2(ImageEnhancementModel):
    """
    
    References:
        https://github.com/Li-Chongyi/Zero-DCE
        
    Args:
        cfg (dict | Path_ | None): Model's layers configuration. It can be an
            external .yaml path or a dictionary. Defaults to None means you
            should define each layer manually in `self.parse_model()` method.
        root (Path_): The root directory of the model. Defaults to RUNS_DIR.
        project (str | None): Project name. Defaults to None.
        name (str | None): Model's name. In case None is given, it will be
            `self.__class__.__name__`. Defaults to None.
        fullname (str | None): Model's fullname in the following format:
            {name}-{data_name}-{postfix}. In case None is given, it will be
            `self.name`. Defaults to None.
        channels (int): Input channel. Defaults to 3.
        num_classes (int | None): Number of classes for classification or
            detection tasks. Defaults to None.
        classlabels (ClassLabels | None): ClassLabels object that contains all
            labels in the dataset. Defaults to None.
        pretrained (Pretrained): Initialize weights from pretrained.
            - If True, use the original pretrained described by the author
              (usually, ImageNet or COCO). By default, it is the first element
              in the `model_zoo` dictionary.
            - If str and is a file/path, then load weights from saved file.
            - In each inherited model, `pretrained` can be a dictionary's
              key to get the corresponding local file or url of the weight.
        phase (ModelPhase_): Model's running phase. Defaults to training.
        loss (Losses_ | None): Loss function for training model.
            Defaults to None.
        metrics (Metrics_ | None): Metric(s) for validating and testing model.
            Defaults to None.
        optimizers (Optimizers_ | None): Optimizer(s) for training model.
            Defaults to None.
        debug (dict | Munch | None): Debug configs. Defaults to None.
        verbose (bool): Verbosity.
    """
    
    model_zoo = {}
    
    def __init__(
        self,
        cfg        : dict | Path_ | None = "zerodcev2-s1",
        root       : Path_               = RUNS_DIR,
        project    : str          | None = None,
        name       : str          | None = "zerodcev2",
        fullname   : str          | None = "zerodcev2-s1",
        channels   : int                 = 3,
        num_classes: int          | None = None,
        classlabels: ClassLabels_ | None = None,
        pretrained : Pretrained			 = False,
        phase      : ModelPhase_         = "training",
        loss   	   : Losses_      | None = CombinedLoss(tv_weight=1600.0),
        metrics	   : Metrics_     | None = None,
        optimizers : Optimizers_  | None = None,
        debug      : dict | Munch | None = None,
        verbose    : bool                = False,
        *args, **kwargs
    ):
        cfg = cfg or "zerodcev2-s1"
        if isinstance(cfg, str) and cfg in cfgs:
            cfg = cfgs[cfg]
        elif isinstance(cfg, (str, Path)) and not is_yaml_file(cfg):
            cfg = CFG_DIR / cfg

        super().__init__(
            cfg         = cfg,
            root        = root,
            project     = project,
            name        = name,
            fullname    = fullname,
            channels    = channels,
            num_classes = num_classes,
            classlabels = classlabels,
            pretrained  = ZeroDCEV2.init_pretrained(pretrained),
            phase       = phase,
            loss        = loss or CombinedLoss(tv_weight=1600.0),
            metrics     = metrics,
            optimizers  = optimizers,
            debug       = debug,
            verbose     = verbose,
            *args, **kwargs
        )
    
    def init_weights(self, m: Module):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            if hasattr(m, "conv"):
                m.conv.weight.data.normal_(0.0, 0.02)
            elif hasattr(m, "dw_conv"):
                m.dw_conv.weight.data.normal_(0.0, 0.02)
            elif hasattr(m, "pw_conv"):
                m.pw_conv.weight.data.normal_(0.0, 0.02)
            else:
                m.weight.data.normal_(0.0, 0.02)
    
    def forward_loss(
        self,
        input : Tensor,
        target: Tensor,
        *args, **kwargs
    ) -> tuple[Tensor, Tensor | None]:
        """
        Forward pass with loss value. Loss function may require more arguments
        beside the ground-truth and prediction values. For calculating the
        metrics, we only need the final predictions and ground-truth.

        Args:
            input (Tensor): Input of shape [B, C, H, W].
            target (Tensor): Ground-truth of shape [B, C, H, W].
            
        Returns:
            Predictions and loss value.
        """
        pred  = self.forward(input=input, *args, **kwargs)
        loss  = self.loss(input, pred) if self.loss else None
        loss += self.regularization_loss(alpha=0.1)
        return pred[-1], loss
    
    def regularization_loss(self, alpha: float = 0.1):
        loss = 0.0
        for sub_module in self.model.modules():
            if hasattr(sub_module, "regularization_loss"):
                loss += sub_module.regularization_loss()
        return alpha * loss

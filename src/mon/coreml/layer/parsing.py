#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements model-parsing functions."""

from __future__ import annotations

__all__ = [
    "parse_model"
]

from torch.nn import Sequential

from mon.coreml.layer import base
from mon.coreml.layer.common import *
from mon.coreml.layer.specific import *


# region Model Parsing

def parse_model(
    d      : dict      | None = None,
    ch     : list[int] | None = None,
    hparams: dict      | None = None,
) -> tuple[Sequential, list[int], list[dict]]:
    """Build the model. We inherit the same idea of model parsing in YOLOv5.
    
    Each layer should have the following attributes:
        - i: index of the layer.
        - f: from, i.e., the current layer receives output from the f-th layer.
             For example: -1 means from a previous layer; -2 means from 2
             previous layers; [99, 101] means from the 99th and 101st layers.
             This attribute is used in forward pass.
        - t: type of the layer using this script:
            t = str(m)[8:-2].replace("__main__.", "")
        - np: number of parameters using the following script:
            np = sum([x.numel() for x in m.parameters()])
    
    Args:
        d: Model definition dictionary. Default to None means building the model
            manually.
        ch: The first layer's input channels. If given, it will be used to
            further calculate the next layer's input channels. Defaults to None
            means defines each layer in_ and out_channels manually.
        hparams: Layer's hyperparameters. They are used to change the values of
            :param:`args`. Usually used in grid search or random search during
            training. Defaults to None.
        
    Returns:
        A Sequential model.
        A list of layer index to save the features during forward pass.
        A list of layer's info for debugging.
    """
    anchors = d.get("anchors",        None)
    nc      = d.get("num_classes",    None)
    gd      = d.get("depth_multiple", 1)
    gw      = d.get("width_multiple", 1)
    
    layers = []      # layers
    save   = []      # savelist
    ch     = ch or [3]
    ch     = [ch] if isinstance(ch, int) else ch
    c2     = ch[-1]  # out_channels
    info   = []      # print data as table
    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):
        # Convert string class name into class
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            if a == "random":
                continue
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass
        
        if isinstance(m, base.HeadLayerParsingMixin):
            args, ch = m.parse_layer(f=f, args=args, nc=nc, ch=ch, hparams=hparams)
        elif isinstance(m, base.LayerParsingMixin):
            args, ch = m.parse_layer(f=f, args=args, ch=ch, hparams=hparams)
        
        # Create layers
        m_    = Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)
        m_.i  = i
        m_.f  = f
        m_.t  = t  = str(m)[8:-2].replace("__main__.", "")      # module type
        m_.np = np = sum([x.numel() for x in m_.parameters()])  # number params
        sa    = [x % i for x in ([f] if isinstance(f, int) else f) if x != -1]
        save.extend(sa)  # append to savelist
        layers.append(m_)
        info.append({
            "index"    : i,
            "from"     : f,
            "n"        : n,
            "params"   : np,
            "module"   : t,
            "arguments": args,
        })
    
    return Sequential(*layers), sorted(save), info


def parse_model_old(
    d : dict      | None = None,
    ch: list[int] | None = None
) -> tuple[Sequential, list[int], list[dict]]:
    """Build the model. We inherit the same idea of model parsing in YOLOv5.
    
    Each layer should have the following attributes:
        - i: index of the layer.
        - f: from, i.e., the current layer receive output from the f-th layer.
             For example: -1 means from previous layer; -2 means from 2 previous
             layers; [99, 101] means from the 99th and 101st layers. This
             attribute is used in forward pass.
        - t: type of the layer using this script:
            t = str(m)[8:-2].replace("__main__.", "")
        - np: number of parameters using the following script:
            np = sum([x.numel() for x in m.parameters()])
    
    Args:
        d: Model definition dictionary. Default to None means building the model
            manually.
        ch: The first layer's input channels. If given, it will be used to
            further calculate the next layer's input channels. Defaults to None
            means defines each layer in_ and out_channels manually.
    
    Returns:
        A Sequential model.
        A list of layer index to save the features during forward pass.
        A list of layer's info for debugging.
    """
    anchors = d.get("anchors",        None)
    nc      = d.get("num_classes",    None)
    gd      = d.get("depth_multiple", 1)
    gw      = d.get("width_multiple", 1)
    
    layers = []      # layers
    save   = []      # savelist
    ch     = ch or [3]
    c2     = ch[-1]  # out_channels
    info   = []      # print data as table
    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):
        # Convert string class name into class
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            if a == "random":
                continue
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass
        
        # Group 01
        if m in [
            ABSConv2dS,
            ABSConv2dS1,
            ABSConv2dS10,
            ABSConv2dS11,
            ABSConv2dS12,
            ABSConv2dS13,
            ABSConv2dS2,
            ABSConv2dS3,
            ABSConv2dS4,
            ABSConv2dS5,
            ABSConv2dS6,
            ABSConv2dS7,
            ABSConv2dS8,
            ABSConv2dS9,
            ABSConv2dU,
            ADCE,
            BSConv2dS,
            BSConv2dU,
            Conv2d,
            Conv2dNormActivation,
            ConvTranspose2d,
            DCE,
            DepthwiseSeparableConv2d,
            DepthwiseSeparableConv2dReLU,
            EnhancementModule,
            FFAPostProcess,
            FFAPreProcess,
            FINetConvBlock,
            FINetGhostConv,
            FINetGhostUpBlock,
            FINetUpBlock,
            GhostConv2d,
            HINetConvBlock,
            HINetUpBlock,
            InceptionBasicConv2d,
            MobileOneConv2d,
            MobileOneStage,
            SRCNN,
            UnconstrainedBlueprintSeparableConv2d,
            UNetBlock,
            VDSR,
        ]:
            if isinstance(f, list | tuple):
                c1, c2 = ch[f[0]], args[0]
            else:
                c1, c2 = ch[f],    args[0]
            args = [c1, c2, *args[1:]]
        # Group 02
        elif m in [
            FFA,
            FFABlock,
            FFAGroup,
            GhostSAM,
            GhostSupervisedAttentionModule,
            PixelAttentionModule,
            SAM,
            SupervisedAttentionModule,
        ]:
            if isinstance(f, list | tuple):
                c1 = c2 = ch[f[0]]
            else:
                c1 = c2 = ch[f]
            args = [c1, *args[0:]]
        # Group 03
        elif m in [
            InvertedResidual,
        ]:
            if isinstance(f, list | tuple):
                c1, c2 = ch[f[0]], args[0]
            else:
                c1, c2 = ch[f],    args[0]
            args = [c1, *args[0:]]
        # Group 04
        elif m in [
            AlexNetClassifier,
            ConvNeXtClassifier,
            GoogleNetClassifier,
            InceptionAux1,
            InceptionAux2,
            InceptionClassifier,
            LeNetClassifier,
            LinearClassifier,
            MobileOneClassifier,
            ShuffleNetV2Classifier,
            SqueezeNetClassifier,
            VGGClassifier,
        ]:
            c1   = args[0]
            c2   = nc
            args = [c1, c2, *args[1:]]
        # Group 05
        elif m in [
            BatchNorm2d,
        ]:
            args = [ch[f]]
        # Group 06
        elif m in [
            DenseBlock,
            DenseTransition,
            Fire,
            Inception,
            InceptionA,
            InceptionB,
            InceptionC,
            InceptionD,
            InceptionE,
        ]:
            c1 = args[0]
            if m in [DenseBlock]:
                out_channels = args[1]
                num_layers   = args[2]
                c2           = c1 + out_channels * num_layers
            elif m in [DenseTransition]:
                c2           = c1 // 2
            if m in [Fire]:
                expand1x1_planes = args[2]
                expand3x3_planes = args[3]
                c2               = expand1x1_planes + expand3x3_planes
            elif m in [Inception]:
                ch1x1     = args[1]
                ch3x3     = args[3]
                ch5x5     = args[5]
                pool_proj = args[6]
                c2        = ch1x1 + ch3x3 + ch5x5 + pool_proj
            elif m in [InceptionA]:
                c2 = m.base_out_channels + args[1]
            elif m in [InceptionB, InceptionD]:
                c2 = m.base_out_channels + c1
            elif m in [InceptionC, InceptionE]:
                c2 = m.base_out_channels
        # Group 07
        elif m in [
            ResNetBlock,
        ]:
            c1 = args[2]
            c2 = args[3]
        # Group 08
        elif m in [
            Join,
            PixelwiseHigherOrderLECurve,
            Shortcut,
            Sum,
        ]:
            c2 = ch[f[-1]]
        # Group 09
        elif m in [Foldcut]:
            c2 = ch[f] // 2
        # Group 10
        elif m in [
            Chuncat,
            Concat,
            InterpolateConcat,
        ]:
            c2 = sum([ch[x] for x in f])
        # Group 11
        elif m in [
            ExtractFeature,
        ]:
            c2 = args[0]
        # Group 12
        elif m in [
            ExtractFeatures,
        ]:
            c2 = args[1] - args[0]
        # Group 13
        else:
            c2 = ch[f]
        
        # Append c2 as c1 for next layers
        if i == 0:
            ch = []
        ch.append(c2)
        
        # Create layers
        m_    = Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)
        m_.i  = i
        m_.f  = f
        m_.t  = t  = str(m)[8:-2].replace("__main__.", "")      # module type
        m_.np = np = sum([x.numel() for x in m_.parameters()])  # number params
        sa    = [x % i for x in ([f] if isinstance(f, int) else f) if x != -1]
        save.extend(sa)  # append to savelist
        layers.append(m_)
        info.append({
            "index"    : i,
            "from"     : f,
            "n"        : n,
            "params"   : np,
            "module"   : t,
            "arguments": args,
        })
    
    return Sequential(*layers), sorted(save), info

# endregion

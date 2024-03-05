#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements model-parsing functions."""

from __future__ import annotations

__all__ = [
    "parse_model"
]

from torch import nn

from mon.nn.layer import *


# region Model Parsing

def parse_model(
    d      : dict      | None = None,
    ch     : list[int] | None = None,
    hparams: dict      | None = None,
) -> tuple[nn.Sequential, list[int], list[dict]]:
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
            further calculate the next layer's input channels. Default: ``None``
            means defines each layer in_ and out_channels manually.
        hparams: Layer's hyperparameters. They are used to change the values of
            :param:`args`. Usually used in grid search or random search during
            training. Default: ``None``.
        
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

    # keys = sorted([k for k in LAYERS.keys()])
    # print(keys)
    
    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):
        # Convert string class name into class
        if m in LAYERS:
            m = LAYERS[m] if isinstance(m, str) else m  # eval strings
        else:
            m = eval(m) if isinstance(m, str) else m  # eval strings
            
        for j, a in enumerate(args):
            if a == "random":
                continue
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass
        
        # console.log(f"Parsing layer: {i}")
        if issubclass(m, base.HeadLayerParsingMixin):
            args, ch = m.parse_layer(f=f, args=args, nc=nc, ch=ch, hparams=hparams)
        else:
            args, ch = m.parse_layer(f=f, args=args, ch=ch, hparams=hparams)
        if i == 0:
            ch = ch[1:]

        # Create layers
        m_    = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)
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
    
    return nn.Sequential(*layers), sorted(save), info

# endregion

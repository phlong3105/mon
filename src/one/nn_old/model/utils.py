#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import os
from typing import Callable

import torch
from one.io import is_weights_file
from torch import nn

from one.core import console

__all__ = [
    "find_modules",
    "get_next_version",
    "named_apply",
    "prune",
    "sparsity",
]


# MARK: - Functional

def find_modules(model, mclass=nn.Conv2d) -> list[int]:
    """Finds layer indices matching module class `mclass`."""
    return [i for i, m in enumerate(model.module_list) if isinstance(m, mclass)]


def get_next_version(root_dir: str) -> int:
    """Get the next experiment version number.
    
    Args:
        root_dir (str):
            Path to the folder that contains all experiment folders.

    Returns:
        version (int):
            Next version number.
    """
    try:
        listdir_info = os.listdir(root_dir)
    except OSError:
        # console.log(f"Missing folder: {root_dir}")
        return 0
    
    existing_versions = []
    for listing in listdir_info:
        if isinstance(listing, str):
            d = listing
        else:
            d = listing["name"]
        bn = os.path.basename(d)
        if bn.startswith("version_"):
            dir_ver = bn.split("_")[1].replace("/", "")
            existing_versions.append(int(dir_ver))
    if len(existing_versions) == 0:
        return 0
    
    return max(existing_versions) + 1


def named_apply(
    fn          : Callable,
    module      : nn.Module,
    name        : str  = "",
    depth_first : bool = True,
    include_root: bool = False,
) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(
            fn=fn, module=child_module, name=child_name,
            depth_first=depth_first, include_root=True
        )
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


def prune(model: nn.Module, amount: float = 0.3):
    """Prune model to requested global sparsity."""
    import torch.nn.utils.prune as prune
    console.log("Pruning model... ", end="")
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            prune.l1_unstructured(m, name="weight", amount=amount)  # prune
            prune.remove(m, "weight")  # make permanent
    console.log(" %.3g global sparsity" % sparsity(model))


def sparsity(model) -> float:
    """Return global model sparsity."""
    a, b = 0.0, 0.0
    for p in model.parameters():
        a += p.numel()
        b += (p == 0).sum()
    return b / a


def strip_optimizer(weight_file: str, new_file: str = ""):
    """Strip optimizer from saved weight file to finalize training.
    Optionally save as `new_file`.
    """
    if not is_weights_file(weight_file):
        raise ValueError(f"`weight_file` must be a torch saved file.")
        
    x = torch.load(weight_file, map_location=torch.device("cpu"))
    x["optimizer"]        = None
    x["training_results"] = None
    x["epoch"]            = -1
    x["model"].half()  # to FP16
    for p in x["model"].parameters():
        p.requires_grad = False
        
    torch.save(x, new_file or weight_file)
    mb = os.path.getsize(new_file or weight_file) / 1E6  # filesize
    console.log(
        "Optimizer stripped from %s,%s %.1fMB"
        % (weight_file, (" saved as %s," % new_file) if new_file else "", mb)
    )

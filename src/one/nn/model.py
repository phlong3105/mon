#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model and training-related components.
"""

from __future__ import annotations

import os
import platform
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.accelerators import CUDAAccelerator
from pytorch_lightning.accelerators import HPUAccelerator
from pytorch_lightning.accelerators import IPUAccelerator
from pytorch_lightning.accelerators import MPSAccelerator
from pytorch_lightning.accelerators import TPUAccelerator
from pytorch_lightning.plugins import DDP2Plugin
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.utilities import _HPU_AVAILABLE
from pytorch_lightning.utilities import _IPU_AVAILABLE
from pytorch_lightning.utilities import _TPU_AVAILABLE
from torch import distributed as dist
from torch import nn
from torch.hub import load_state_dict_from_url

from one.core import assert_ckpt_file
from one.core import assert_weights_file
from one.core import Callable
from one.core import console
from one.core import create_dirs
from one.core import error_console
from one.core import get_latest_file
from one.core import intersect_weight_dicts
from one.core import is_torch_saved_file
from one.core import is_url
from one.core import Path_


# H1: - Checkpoint -------------------------------------------------------------

def extract_weights_from_checkpoint(
    ckpt       : Path_,
    weight_file: Path_ | None = None,
):
    """
    Extract and save model's weights from checkpoint file.
    
    Args:
        ckpt (Path_): Checkpoint file.
        weight_file (Path_ | None): Path save the weights file. Defaults to
            None which saves at the same location as the .ckpt file.
    """
    ckpt = Path(ckpt)
    assert_ckpt_file(ckpt)
    
    state_dict = load_state_dict_from_path(str(ckpt))
    if state_dict is None:
        raise ValueError()
    
    if weight_file is None:
        weight_file = ckpt.parent / f"{ckpt.stem}.pth"
    else:
        weight_file = Path(weight_file)
    create_dirs([weight_file.parent])
    torch.save(state_dict, str(weight_file))


def get_epoch(ckpt: Path_) -> int:
    """
    Get the current epoch from the saved weights file.

    Args:
        ckpt (Path_): Checkpoint path.

    Returns:
        Current epoch.
    """
    ckpt = Path(ckpt)
    assert_ckpt_file(ckpt)
    
    epoch = 0
    if is_torch_saved_file(ckpt):
        ckpt  = torch.load(ckpt)
        epoch = ckpt.get("epoch", 0)
    
    return epoch


def get_global_step(ckpt: Path_) -> int:
    """
    Get the global step from the saved weights file.

    Args:
        ckpt (Path_): Checkpoint path.

    Returns:
        Global step.
    """
    ckpt = Path(ckpt)
    assert_ckpt_file(ckpt)

    global_step = 0
    if is_torch_saved_file(ckpt):
        ckpt        = torch.load(ckpt)
        global_step = ckpt.get("global_step", 0)
    
    return global_step


def get_latest_checkpoint(dirpath: Path_) -> str | None:
    """
    Get the latest weights in the `dir`.

    Args:
        dirpath (Path_): Directory that contains the checkpoints.

    Returns:
        Checkpoint path.
    """
    dirpath  = Path(dirpath)
    ckpt     = get_latest_file(dirpath)
    if ckpt is None:
        error_console.log(f"[red]Cannot find checkpoint file {dirpath}.")
    return ckpt


def load_pretrained(
    module	  	: nn.Module,
    path  		: Path_,
    model_dir   : Path_ | None = None,
    map_location: str   | None = torch.device("cpu"),
    progress	: bool 		   = True,
    check_hash	: bool		   = False,
    filename	: str   | None = None,
    strict		: bool		   = False,
    **_
) -> nn.Module:
    """
    Load pretrained weights. This is a very convenient function to load the
    state dict from saved pretrained weights or checkpoints. Filter out mismatch
    keys and then load the layers' weights.
    
    Args:
        module (nn.Module): Module to load pretrained.
        path (Path_): The weights or checkpoints file to load. If it is a URL,
            it will be downloaded.
        model_dir (Path_ | None): Directory to save the weights or checkpoint
            file. Defaults to None.
        map_location (str | None): A function or a dict specifying how to
            remap storage locations (see torch.load). Defaults to `cpu`.
        progress (bool): Whether to display a progress bar to stderr.
            Defaults to True.
        check_hash (bool): If True, the filename part of the URL should follow
            the naming convention `filename-<sha256>.ext` where `<sha256>` is
            the first eight or more digits of the SHA256 hash of the contents
            of the file. Hash is used to ensure unique names and to verify the
            contents of the file. Defaults to False.
        filename (str | None): Name for the downloaded file. Filename from
            `url` will be used if not set.
        strict (bool): Whether to strictly enforce that the keys in `state_dict`
            match the keys returned by this module's
            `~torch.nn.Module.state_dict` function. Defaults to False.
    """
    state_dict = load_state_dict_from_path(
        path         = path,
        model_dir    = model_dir,
        map_location = map_location,
        progress     = progress,
        check_hash   = check_hash,
        filename     = filename
    )
    module = load_state_dict(
        module     = module,
        state_dict = state_dict,
        strict     = strict
    )
    return module


def load_state_dict(
    module	  : nn.Module,
    state_dict: dict,
    strict    : bool = False,
    **_
) -> nn.Module:
    """
    Load the module state dict. This is an extension of `nn.Module.load_state_dict()`.
    We add an extra snippet to drop missing keys between module's state_dict
    and pretrained state_dict, which will cause an error.

    Args:
        module (nn.Module): Module to load state dict.
        state_dict (dict): A dict containing parameters and persistent buffers.
        strict (bool): Whether to strictly enforce that the keys in `state_dict`
            match the keys returned by this module's `~torch.nn.Module.state_dict`
            function. Defaults to False.

    Returns:
        Module after loading state dict.
    """
    module_dict = module.state_dict()
    module_dict = match_state_dicts(
        model_dict      = module_dict,
        pretrained_dict = state_dict
    )
    module.load_state_dict(module_dict, strict=strict)
    return module


def load_state_dict_from_path(
    path  		: Path_,
    model_dir   : Path_ | None = None,
    map_location: str   | None = torch.device("cpu"),
    progress	: bool 		   = True,
    check_hash	: bool		   = False,
    filename 	: str   | None = None,
    **_
) -> dict | None:
    """
    Load state dict at the given URL. If downloaded file is a zip file, it
    will be automatically decompressed. If the object is already present in
    `model_dir`, it's deserialized and returned.
    
    Args:
        path (Path_): The weights or checkpoints file to load. If it is a URL,
            it will be downloaded.
        model_dir (Path_ | None): Directory in which to save the object.
            Default to None.
        map_location (optional): A function or a dict specifying how to remap
            storage locations (see torch.load). Defaults to `cpu`.
        progress (bool): Whether to display a progress bar to stderr.
            Defaults to True.
        check_hash (bool): If True, the filename part of the URL should follow
            the naming convention `filename-<sha256>.ext` where `<sha256>`
            is the first eight or more digits of the SHA256 hash of the
            contents of the file. Hash is used to ensure unique names and to
            verify the contents of the file. Defaults to False.
        filename (str | None): Name for the downloaded file. Filename from
            `url` will be used if not set.

    Example:
        >>> state_dict = load_state_dict_from_path(
        >>> 	'https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth'
        >>> )
    """
    if path is None:
        raise ValueError()
    if model_dir is not None:
        model_dir = Path(model_dir)
    
    path = Path(path)
    if not is_torch_saved_file(path) and \
        (model_dir is None or not model_dir.is_dir()):
        raise ValueError(f"`model_dir` must be defined. But got: {model_dir}.")
    
    state_dict = None
    if is_torch_saved_file(path):
        # Can be either the weight file or the weights file.
        state_dict = torch.load(str(path), map_location=map_location)
    elif is_url(path):
        state_dict = load_state_dict_from_url(
            url          = str(path),
            model_dir    = str(model_dir),
            map_location = map_location,
            progress     = progress,
            check_hash   = check_hash,
            file_name    = filename
        )
    
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    if "state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    return state_dict


def match_state_dicts(
    model_dict	   : dict,
    pretrained_dict: dict,
    exclude		   : tuple | list = ()
) -> dict:
    """
    Filter out unmatched keys btw the model's `state_dict` and the pretrained's
    `state_dict`. Omitting `exclude` keys.

    Args:
        model_dict (dict): Model's `state_dict`.
        pretrained_dict (dict): Pretrained's `state_dict`.
        exclude (tuple | list): List of excluded keys. Defaults to ().
        
    Returns:
        Filtered model's `state_dict`.
    """
    # 1. Filter out unnecessary keys
    intersect_dict = intersect_weight_dicts(
        pretrained_dict,
        model_dict,
        exclude
    )
    """
       intersect_dict = {
           k: v for k, v in pretrained_dict.items()
           if k in model_dict and
              not any(x in k for x in exclude) and
              v.shape == model_dict[k].shape
       }
       """
    # 2. Overwrite entries in the existing state dict
    model_dict.update(intersect_dict)
    return model_dict


# H1: - Distribution -----------------------------------------------------------

def get_distributed_info() -> tuple[int, int]:
    """
    If distributed is available, return the rank and world size, otherwise
    return 0 and 1
    
    Returns:
        The rank and world size of the current process.
    """
    if dist.is_available():
        initialized = dist.is_initialized()
    else:
        initialized = False
    if initialized:
        rank       = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank       = 0
        world_size = 1
    return rank, world_size


def is_parallel(model: nn.Module) -> bool:
    """
    If the model is a parallel model, then it returns True, otherwise it returns
    False
    
    Args:
        model (nn.Module): The model to check.
    
    Returns:
        A boolean value.
    """
    return type(model) in (
        nn.parallel.DataParallel,
        nn.parallel.DistributedDataParallel
    )


def set_distributed_backend(strategy: str | Callable, cudnn: bool = True):
    """
    If you're running on Windows, set the distributed backend to gloo. If you're
    running on Linux, set the distributed backend to nccl
    
    Args:
        strategy (str | Callable): The distributed strategy to use. One of
            ["ddp", "ddp2"]
        cudnn (bool): Whether to use cuDNN or not. Defaults to True.
    """
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.enabled = cudnn
        console.log(
            f"cuDNN available: [bright_green]True[/bright_green], "
            f"used:" + "[bright_green]True" if cudnn else "[red]False"
        )
    else:
        console.log(f"cuDNN available: [red]False")
    
    if strategy in ["ddp", "ddp2"] or isinstance(strategy, (DDPPlugin, DDP2Plugin)):
        if platform.system() == "Windows":
            os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"
            console.log(
                "Running on a Windows machine, set torch distributed backend "
                "to gloo."
            )
        elif platform.system() == "Linux":
            os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "nccl"
            console.log(
                "Running on a Unix machine, set torch distributed backend "
                "to nccl."
            )


# H1: - Model ------------------------------------------------------------------

def find_modules(model, mclass=nn.Conv2d) -> list[int]:
    """
    Finds layer indices matching module class `mclass`.
    """
    return [i for i, m in enumerate(model.module_list) if isinstance(m, mclass)]


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
    """
    Prune model to requested global sparsity.
    """
    import torch.nn.utils.prune as prune
    console.log("Pruning model... ", end="")
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            prune.l1_unstructured(m, name="weight", amount=amount)  # prune
            prune.remove(m, "weight")  # make permanent
    console.log(" %.3g global sparsity" % sparsity(model))


def sparsity(model: nn.Module) -> float:
    """
    Return global model sparsity.
    """
    a = 0.0
    b = 0.0
    for p in model.parameters():
        a += p.numel()
        b += (p == 0).sum()
    return b / a


def strip_optimizer(weight_file: str, new_file: str = ""):
    """
    Strip optimizer from saved weight file to finalize training.
    Optionally save as `new_file`.
    """
    assert_weights_file(weight_file)
    
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


# H1: - Trainer ----------------------------------------------------------------

class Trainer(pl.Trainer):
    """
    Override `pytorch_lightning.Trainer` with several methods and properties.
    """
    
    @pl.Trainer.current_epoch.setter
    def current_epoch(self, current_epoch: int):
        self.fit_loop.current_epoch = current_epoch
    
    @pl.Trainer.global_step.setter
    def global_step(self, global_step: int):
        self.fit_loop.global_step = global_step
        
    def _log_device_info(self):
        if CUDAAccelerator.is_available():
            gpu_available = True
            gpu_type      = " (cuda)"
        elif MPSAccelerator.is_available():
            gpu_available = True
            gpu_type      = " (mps)"
        else:
            gpu_available = False
            gpu_type      = ""

        gpu_used = isinstance(self.accelerator, (CUDAAccelerator, MPSAccelerator))
        console.log(f"GPU available: {gpu_available}{gpu_type}, used: {gpu_used}")

        num_tpu_cores = self.num_devices if isinstance(self.accelerator, TPUAccelerator) else 0
        console.log(f"TPU available: {_TPU_AVAILABLE}, using: {num_tpu_cores} TPU cores")

        num_ipus = self.num_devices if isinstance(self.accelerator, IPUAccelerator) else 0
        console.log(f"IPU available: {_IPU_AVAILABLE}, using: {num_ipus} IPUs")

        num_hpus = self.num_devices if isinstance(self.accelerator, HPUAccelerator) else 0
        console.log(f"HPU available: {_HPU_AVAILABLE}, using: {num_hpus} HPUs")

        # Integrate MPS Accelerator here, once gpu maps to both
        if CUDAAccelerator.is_available() and not isinstance(self.accelerator, CUDAAccelerator):
            console.log(
                "GPU available but not used. Set `accelerator` and `devices` using"
                f" `Trainer(accelerator='gpu', devices={CUDAAccelerator.auto_device_count()})`.",
            )

        if _TPU_AVAILABLE and not isinstance(self.accelerator, TPUAccelerator):
            console.log(
                "TPU available but not used. Set `accelerator` and `devices` using"
                f" `Trainer(accelerator='tpu', devices={TPUAccelerator.auto_device_count()})`."
            )

        if _IPU_AVAILABLE and not isinstance(self.accelerator, IPUAccelerator):
            console.log(
                "IPU available but not used. Set `accelerator` and `devices` using"
                f" `Trainer(accelerator='ipu', devices={IPUAccelerator.auto_device_count()})`."
            )

        if _HPU_AVAILABLE and not isinstance(self.accelerator, HPUAccelerator):
            console.log(
                "HPU available but not used. Set `accelerator` and `devices` using"
                f" `Trainer(accelerator='hpu', devices={HPUAccelerator.auto_device_count()})`."
            )

        if MPSAccelerator.is_available() and not isinstance(self.accelerator, MPSAccelerator):
            console.log(
                "MPS available but not used. Set `accelerator` and `devices` using"
                f" `Trainer(accelerator='mps', devices={MPSAccelerator.auto_device_count()})`."
            )

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Parse Arguments.

This module defines functions to parse both user-input and config arguments.
"""

from __future__ import annotations

__all__ = [
    "parse_online_input_args",
    "parse_predict_args",
    "parse_predict_input_args",
    "parse_train_args",
    "parse_train_input_args",
]

import argparse
import socket

from platformdirs import user_data_dir

from mon import core


# region Utils

def _str_or_none(value) -> str | None:
    if value == "None":
        return None
    return value


def _int_or_none(value) -> int | None:
    if value == "None":
        return None
    return int(value)


def _float_or_none(value) -> float | None:
    if value == "None":
        return None
    return float(value)

# endregion


# region Train

def parse_train_input_args() -> argparse.Namespace:
    """Parse input arguments for training."""
    parser = argparse.ArgumentParser(description="train")
    parser.add_argument("--config",     type=_str_or_none, default=None, help="Model config.")
    parser.add_argument("--arch",       type=_str_or_none, default=None, help="Model architecture.")
    parser.add_argument("--model",      type=_str_or_none, default=None, help="Model name.")
    parser.add_argument("--root",       type=_str_or_none, default=None, help="Project root.")
    parser.add_argument("--project",    type=_str_or_none, default=None, help="Project name.")
    parser.add_argument("--variant",    type=_str_or_none, default=None, help="Variant name.")
    parser.add_argument("--fullname",   type=_str_or_none, default=None, help="Fullname to save the model's weight.")
    parser.add_argument("--save-dir",   type=_str_or_none, default=None, help="Save results to root/run/train/arch/model/data or root/run/train/arch/project/variant.")
    parser.add_argument("--weights",    type=_str_or_none, default=None, help="Weights paths.")
    parser.add_argument("--device",     type=_str_or_none, default=None, help="Running devices.")
    parser.add_argument("--local-rank", type=_int_or_none, default=-1,   help="DDP parameter, do not modify.")
    parser.add_argument("--launcher",   type=_str_or_none, choices=["none", "pytorch", "slurm"], default="none", help="DDP parameter, do not modify.")
    parser.add_argument("--epochs",     type=_int_or_none, default=None, help="Stop training once this number of epochs is reached.")
    parser.add_argument("--steps",      type=_int_or_none, default=None, help="Stop training once this number of steps is reached.")
    parser.add_argument("--exist-ok",   action="store_true")
    parser.add_argument("--verbose",    action="store_true")
    parser.add_argument("extra_args",   nargs=argparse.REMAINDER, help="Additional arguments")
    args = parser.parse_args()
    return args


def parse_train_args(model_root: str | core.Path | None = None) -> argparse.Namespace:
    """Parse arguments for training."""
    hostname = socket.gethostname().lower()
    
    # Get input args
    input_args = vars(parse_train_input_args())
    config     = input_args.get("config")
    root       = core.Path(input_args.get("root"))
    weights    = input_args.get("weights")
    
    # Get config args
    config = core.parse_config_file(
        project_root = root,
        model_root   = model_root,
        weights_path = weights,
        config       = config,
    )
    args   = core.load_config(config)
    
    # Prioritize input args --> config file args
    arch       = input_args.get("arch")     or args.get("arch")
    model      = input_args.get("model")    or args.get("model")
    data       = input_args.get("data")     or args.get("data")
    project    = input_args.get("project")  or args.get("project")
    variant    = input_args.get("variant")  or args.get("variant")
    fullname   = input_args.get("fullname") or args.get("fullname")
    save_dir   = input_args.get("save_dir") or args.get("save_dir")
    weights    = input_args.get("weights")  or args.get("weights")
    device     = input_args.get("device")   or args.get("device")
    local_rank = input_args.get("local_rank")
    launcher   = input_args.get("launcher")
    epochs     = input_args.get("epochs")   or args.get("epochs")
    steps      = input_args.get("steps")    or args.get("steps")
    exist_ok   = input_args.get("exist_ok") or args.get("exist_ok")
    verbose    = input_args.get("verbose")  or args.get("verbose")
    extra_args = input_args.get("extra_args")

    # Parse arguments
    save_dir = save_dir or core.parse_save_dir(root/"run"/"train", arch, model, data, project, variant)
    save_dir = core.Path(save_dir)
    weights  = core.parse_weights_file(weights)
    device   = core.parse_device(device)
    
    # Update arguments
    args["hostname"]   = hostname
    args["config"]     = config
    args["arch"]       = arch
    args["model"]      = model
    args["data"]       = data
    args["root"]       = root
    args["project"]    = project
    args["variant"]    = variant
    args["fullname"]   = fullname
    args["save_dir"]   = save_dir
    args["weights"]    = weights
    args["device"]     = device
    args["local_rank"] = local_rank
    args["launcher"]   = launcher
    args["epochs"]     = epochs
    args["steps"]      = steps
    args["exist_ok"]   = exist_ok
    args["verbose"]    = verbose
    args |= extra_args
    args  = argparse.Namespace(**args)
    
    if not exist_ok:
        core.delete_dir(paths=core.Path(save_dir))
        
    save_dir.mkdir(parents=True, exist_ok=True)
    if config and config.is_config_file():
        core.copy_file(src=config, dst=save_dir / f"config{config.suffix}")
    
    return args

# endregion


# region Predict

def parse_predict_input_args() -> argparse.Namespace:
    """Parse input arguments for prediction."""
    parser = argparse.ArgumentParser(description="predict")
    parser.add_argument("--config",       type=_str_or_none, default=None, help="Model config.")
    parser.add_argument("--arch",         type=_str_or_none, default=None, help="Model architecture.")
    parser.add_argument("--model",        type=_str_or_none, default=None, help="Model name.")
    parser.add_argument("--data",         type=_str_or_none, default=None, help="Source data directory.")
    parser.add_argument("--root",         type=_str_or_none, default=None, help="Project root.")
    parser.add_argument("--project",      type=_str_or_none, default=None, help="Project name.")
    parser.add_argument("--variant",      type=_str_or_none, default=None, help="Variant name.")
    parser.add_argument("--fullname",     type=_str_or_none, default=None, help="Save results to root/run/predict/arch/model/data or root/run/predict/arch/project/variant.")
    parser.add_argument("--save-dir",     type=_str_or_none, default=None, help="Optional saving directory.")
    parser.add_argument("--weights",      type=_str_or_none, default=None, help="Weights paths.")
    parser.add_argument("--device",       type=_str_or_none, default=None, help="Running devices.")
    parser.add_argument("--imgsz",        type=_int_or_none, default=None, help="Image sizes.")
    parser.add_argument("--resize",       action="store_true")
    parser.add_argument("--benchmark",    action="store_true")
    parser.add_argument("--save-image",   action="store_true")
    parser.add_argument("--save-debug",   action="store_true")
    parser.add_argument("--use-data-dir", action="store_true")
    parser.add_argument("--use-fullpath", action="store_true")
    parser.add_argument("--verbose",      action="store_true")
    parser.add_argument("extra_args",     nargs=argparse.REMAINDER, help="Additional arguments")
    args = parser.parse_args()
    return args


def parse_predict_args(model_root: str | core.Path | None = None) -> argparse.Namespace:
    """Parse arguments for prediction."""
    hostname = socket.gethostname().lower()
    
    # Get input args
    input_args = vars(parse_predict_input_args())
    config     = input_args.get("config")
    root       = core.Path(input_args.get("root"))
    weights    = input_args.get("weights")
    
    # Get config args
    config = core.parse_config_file(
        project_root = root,
        model_root   = model_root,
        weights_path = weights,
        config       = config,
    )
    args   = core.load_config(config)
    
    # Prioritize input args --> config file args
    arch         = input_args.get("arch")         or args.get("arch")
    model        = input_args.get("model")        or args.get("model")
    data         = input_args.get("data")         or args.get("data")
    project      = input_args.get("project")      or args.get("project")
    variant      = input_args.get("variant")      or args.get("variant")
    fullname     = input_args.get("fullname")     or args.get("fullname")
    save_dir     = input_args.get("save_dir")     or args.get("save_dir")
    weights      = input_args.get("weights")      or args.get("weights")
    device       = input_args.get("device")       or args.get("device")
    imgsz        = input_args.get("imgsz")        or args.get("imgsz")
    resize       = input_args.get("resize")       or args.get("resize")
    benchmark    = input_args.get("benchmark")    or args.get("benchmark")
    save_image   = input_args.get("save_image")   or args.get("save_image")
    save_debug   = input_args.get("save_debug")   or args.get("save_debug")
    use_data_dir = input_args.get("use_data_dir") or args.get("use_data_dir")
    use_fullpath = input_args.get("use_fullpath") or args.get("use_fullpath")
    verbose      = input_args.get("verbose")      or args.get("verbose")
    extra_args   = input_args.get("extra_args")
    
    # Parse arguments
    save_dir = save_dir or core.parse_save_dir(root/"run"/"predict", arch, model, None, project, variant)
    save_dir = core.Path(save_dir)
    weights  = core.parse_weights_file(weights)
    device   = core.parse_device(device)
    imgsz    = core.parse_hw(imgsz)
    
    # Update arguments
    args["hostname"]     = hostname
    args["config"]       = config
    args["arch"]         = arch
    args["model"]        = model
    args["data"]         = data
    args["root"]         = root
    args["project"]      = project
    args["variant"]      = variant
    args["fullname"]     = fullname
    args["save_dir"]     = save_dir
    args["weights"]      = weights
    args["device"]       = device
    args["imgsz"]        = imgsz
    args["resize"]       = resize
    args["benchmark"]    = benchmark
    args["save_image"]   = save_image
    args["save_debug"]   = save_debug
    args["use_data_dir"] = use_data_dir
    args["use_fullpath"] = use_fullpath
    args["verbose"]      = verbose
    args |= extra_args
    args  = argparse.Namespace(**args)
    
    save_dir.mkdir(parents=True, exist_ok=True)
    if config and config.is_config_file():
        core.copy_file(src=config, dst=save_dir / f"config{config.suffix}")
    
    return args

# endregion


# region Online

def parse_online_input_args() -> argparse.Namespace:
    """Parse input arguments for online learning."""
    parser = argparse.ArgumentParser(description="online")
    parser.add_argument("--config",       type=_str_or_none, default=None, help="Model config.")
    parser.add_argument("--arch",         type=_str_or_none, default=None, help="Model architecture.")
    parser.add_argument("--model",        type=_str_or_none, default=None, help="Model name.")
    parser.add_argument("--root",         type=_str_or_none, default=None, help="Project root.")
    parser.add_argument("--project",      type=_str_or_none, default=None, help="Project name.")
    parser.add_argument("--variant",      type=_str_or_none, default=None, help="Variant name.")
    parser.add_argument("--fullname",     type=_str_or_none, default=None, help="Fullname to save the model's weight.")
    parser.add_argument("--save-dir",     type=_str_or_none, default=None, help="Save results to root/run/train/arch/model/data or root/run/train/arch/project/variant.")
    parser.add_argument("--weights",      type=_str_or_none, default=None, help="Weights paths.")
    parser.add_argument("--device",       type=_str_or_none, default=None, help="Running devices.")
    parser.add_argument("--local-rank",   type=_int_or_none, default=-1,   help="DDP parameter, do not modify.")
    parser.add_argument("--launcher",     type=_str_or_none, choices=["none", "pytorch", "slurm"], default="none", help="DDP parameter, do not modify.")
    parser.add_argument("--epochs",       type=_int_or_none, default=-1,   help="Stop training once this number of epochs is reached.")
    parser.add_argument("--steps",        type=_int_or_none, default=-1,   help="Stop training once this number of steps is reached.")
    parser.add_argument("--imgsz",        type=_int_or_none, default=None, help="Image sizes.")
    parser.add_argument("--resize",       action="store_true")
    parser.add_argument("--benchmark",    action="store_true")
    parser.add_argument("--save-image",   action="store_true")
    parser.add_argument("--save-debug",   action="store_true")
    parser.add_argument("--use-data-dir", action="store_true")
    parser.add_argument("--use-fullpath", action="store_true")
    parser.add_argument("--verbose",      action="store_true")
    parser.add_argument("--exist-ok",     action="store_true")
    parser.add_argument("--verbose",      action="store_true")
    parser.add_argument("extra_args",     nargs=argparse.REMAINDER, help="Additional arguments")
    args = parser.parse_args()
    return args

# endregion

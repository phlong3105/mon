#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

__all__ = [
    "parse_predict_args",
    "parse_train_args",
]

import argparse
import socket

import mon


# region Train

def _parse_train_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="predict")
    parser.add_argument("--config",     type=str, default=None, help="Model config.")
    parser.add_argument("--arch",       type=str, default=None, help="Model architecture.")
    parser.add_argument("--model",      type=str, default=None, help="Model name.")
    parser.add_argument("--root",       type=str, default=None, help="Project root.")
    parser.add_argument("--project",    type=str, default=None, help="Project name.")
    parser.add_argument("--variant",    type=str, default=None, help="Variant name.")
    parser.add_argument("--fullname",   type=str, default=None, help="Fullname to save the model's weight.")
    parser.add_argument("--save-dir",   type=str, default=None, help="Save results to root/run/train/arch/model/data or root/run/train/arch/project/variant.")
    parser.add_argument("--weights",    type=str, default=None, help="Weights paths.")
    parser.add_argument("--device",     type=str, default=None, help="Running devices.")
    parser.add_argument("--local-rank", type=int, default=-1,   help="DDP parameter, do not modify.")
    parser.add_argument("--launcher",   type=str, choices=["none", "pytorch", "slurm"], default="none", help="DDP parameter, do not modify.")
    parser.add_argument("--epochs",     type=int, default=None, help="Stop training once this number of epochs is reached.")
    parser.add_argument("--steps",      type=int, default=None, help="Stop training once this number of steps is reached.")
    parser.add_argument("--exist-ok",   action="store_true")
    parser.add_argument("--verbose",    action="store_true")
    parser.add_argument("extra_args",   nargs=argparse.REMAINDER, help="Additional arguments")
    args = parser.parse_args()
    return args


def parse_train_args() -> argparse.Namespace:
    hostname = socket.gethostname().lower()
    
    # Get input args
    input_args = vars(_parse_predict_args())
    config     = input_args.get("config")
    root       = mon.Path(input_args.get("root"))
    
    # Get config args
    config = mon.parse_config_file(project_root=root / "config", config=config)
    args   = mon.load_config(config)
    
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
    save_dir = save_dir or mon.parse_save_dir(root/"run"/"train", arch, model, data, project, variant)
    save_dir = mon.Path(save_dir)
    weights  = mon.to_list(weights)
    device   = mon.parse_device(device)
    
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
        mon.delete_dir(paths=mon.Path(save_dir))
        
    save_dir.mkdir(parents=True, exist_ok=True)
    mon.copy_file(src=config, dst=save_dir / "config.py")
    
    return args
    

'''
def parse_train_args() -> argparse.Namespace:
    
    @click.command(name="train", context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
    @click.option("--config",   type=str, default=None, help="Model config.")
    @click.option("--arch",     type=str, default=None, help="Model architecture.")
    @click.option("--model",    type=str, default=None, help="Model name.")
    @click.option("--root",     type=str, default=None, help="Project root.")
    @click.option("--project",  type=str, default=None, help="Project name.")
    @click.option("--variant",  type=str, default=None, help="Variant name.")
    @click.option("--fullname", type=str, default=None, help="Fullname to save the model's weight.")
    @click.option("--save-dir", type=str, default=None, help="Save results to root/run/train/arch/model/data or root/run/train/arch/project/variant.")
    @click.option("--weights",  type=str, default=None, help="Weights paths.")
    @click.option("--device",   type=str, default=None, help="Running devices.")
    @click.option("--epochs",   type=int, default=None, help="Stop training once this number of epochs is reached.")
    @click.option("--steps",    type=int, default=None, help="Stop training once this number of steps is reached.")
    @click.option("--exist-ok", is_flag=True)
    @click.option("--verbose",  is_flag=True)
    @click.pass_context
    def command(
        ctx,
        config  : str,
        arch    : str,
        model   : str,
        root    : str,
        project : str,
        variant : str,
        fullname: str,
        save_dir: str,
        weights : str,
        device  : str,
        epochs  : int,
        steps   : int,
        exist_ok: bool,
        verbose : bool,
    ) -> argparse.Namespace:
        hostname = socket.gethostname().lower()
        
        # Get extra args
        extra_args = {
            k.lstrip("--"): ctx.args[i + 1]
            if not (i + 1 >= len(ctx.args) or ctx.args[i + 1].startswith("--")) else True
            for i, k in enumerate(ctx.args) if k.startswith("--")
        }
        
        # Get config args
        config = mon.parse_config_file(project_root=mon.Path(root) / "config", config=config)
        args   = mon.load_config(config)
        
        # Prioritize input args --> config file args
        model    = model    or args.get("model")
        data     =             args.get("data")
        project  = project  or args.get("project")
        variant  = variant  or args.get("variant")
        fullname = fullname or args.get("fullname")
        save_dir = save_dir or args.get("save_dir")
        weights  = weights  or args.get("weights")
        device   = device   or args.get("device")
        epochs   = epochs   or args.get("epochs")
        steps    = steps    or args.get("steps")
        exist_ok = exist_ok or args.get("exist_ok")
        verbose  = verbose  or args.get("verbose")
        
        # Parse arguments
        root     = mon.Path(root)
        save_dir = save_dir or mon.parse_save_dir(root/"run"/"train", arch, model, data, project, variant)
        save_dir = mon.Path(save_dir)
        weights  = mon.to_list(weights)
        device   = mon.parse_device(device)
        
        # Update arguments
        args["hostname"] = hostname
        args["config"]   = config
        args["arch"]     = arch
        args["model"]    = model
        args["root"]     = root
        args["project"]  = project
        args["variant"]  = variant
        args["fullname"] = fullname
        args["save_dir"] = save_dir
        args["weights"]  = weights
        args["device"]   = device
        args["epochs"]   = epochs
        args["steps"]    = steps
        args["exist_ok"] = exist_ok
        args["verbose"]  = verbose
        args |= extra_args
        args  = argparse.Namespace(**args)
        
        if not exist_ok:
            mon.delete_dir(paths=mon.Path(args.save_dir))
        
        return args
    
    return command()
'''

# endregion


# region Predict

def _parse_predict_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="predict")
    parser.add_argument("--config",     type=str, default=None, help="Model config.")
    parser.add_argument("--arch",       type=str, default=None, help="Model architecture.")
    parser.add_argument("--model",      type=str, default=None, help="Model name.")
    parser.add_argument("--data",       type=str, default=None, help="Source data directory.")
    parser.add_argument("--root",       type=str, default=None, help="Project root.")
    parser.add_argument("--project",    type=str, default=None, help="Project name.")
    parser.add_argument("--variant",    type=str, default=None, help="Variant name.")
    parser.add_argument("--fullname",   type=str, default=None, help="Save results to root/run/predict/arch/model/data or root/run/predict/arch/project/variant.")
    parser.add_argument("--save-dir",   type=str, default=None, help="Optional saving directory.")
    parser.add_argument("--weights",    type=str, default=None, help="Weights paths.")
    parser.add_argument("--device",     type=str, default=None, help="Running devices.")
    parser.add_argument("--imgsz",      type=int, default=None, help="Image sizes.")
    parser.add_argument("--resize",     action="store_true")
    parser.add_argument("--benchmark",  action="store_true")
    parser.add_argument("--save-image", action="store_true")
    parser.add_argument("--save-debug", action="store_true")
    parser.add_argument("--verbose",    action="store_true")
    parser.add_argument("extra_args",   nargs=argparse.REMAINDER, help="Additional arguments")
    args = parser.parse_args()
    return args


def parse_predict_args() -> argparse.Namespace:
    hostname = socket.gethostname().lower()
    
    # Get input args
    input_args = vars(_parse_predict_args())
    config     = input_args.get("config")
    root       = mon.Path(input_args.get("root"))
    
    # Get config args
    config = mon.parse_config_file(project_root=root / "config", config=config)
    args   = mon.load_config(config)
    
    # Prioritize input args --> config file args
    arch       = input_args.get("arch")       or args.get("arch")
    model      = input_args.get("model")      or args.get("model")
    data       = input_args.get("data")       or args.get("data")
    project    = input_args.get("project")    or args.get("project")
    variant    = input_args.get("variant")    or args.get("variant")
    fullname   = input_args.get("fullname")   or args.get("fullname")
    save_dir   = input_args.get("save_dir")   or args.get("save_dir")
    weights    = input_args.get("weights")    or args.get("weights")
    device     = input_args.get("device")     or args.get("device")
    imgsz      = input_args.get("imgsz")      or args.get("imgsz")
    resize     = input_args.get("resize")     or args.get("resize")
    benchmark  = input_args.get("benchmark")  or args.get("benchmark")
    save_image = input_args.get("save_image") or args.get("save_image")
    save_debug = input_args.get("save_debug") or args.get("save_debug")
    verbose    = input_args.get("verbose")    or args.get("verbose")
    extra_args = input_args.get("extra_args")
    
    # Parse arguments
    save_dir = save_dir or mon.parse_save_dir(root/"run"/"predict", arch, model, None, project, variant)
    save_dir = mon.Path(save_dir)
    weights  = mon.to_list(weights)
    device   = mon.parse_device(device)
    imgsz    = mon.parse_hw(imgsz)
    
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
    args["imgsz"]      = imgsz
    args["resize"]     = resize
    args["benchmark"]  = benchmark
    args["save_image"] = save_image
    args["save_debug"] = save_debug
    args["verbose"]    = verbose
    args |= extra_args
    args  = argparse.Namespace(**args)
    
    save_dir.mkdir(parents=True, exist_ok=True)
    mon.copy_file(src=config, dst=save_dir / "config.py")
    
    return args


'''
def parse_predict_args() -> argparse.Namespace:
    
    @click.command(name="predict", context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
    @click.option("--config",     type=str, default=None, help="Model config.")
    @click.option("--arch",       type=str, default=None, help="Model architecture.")
    @click.option("--model",      type=str, default=None, help="Model name.")
    @click.option("--data",       type=str, default=None, help="Source data directory.")
    @click.option("--root",       type=str, default=None, help="Project root.")
    @click.option("--project",    type=str, default=None, help="Project name.")
    @click.option("--variant",    type=str, default=None, help="Variant name.")
    @click.option("--fullname",   type=str, default=None, help="Save results to root/run/predict/arch/model/data or root/run/predict/arch/project/variant.")
    @click.option("--save-dir",   type=str, default=None, help="Optional saving directory.")
    @click.option("--weights",    type=str, default=None, help="Weights paths.")
    @click.option("--device",     type=str, default=None, help="Running devices.")
    @click.option("--imgsz",      type=int, default=None, help="Image sizes.")
    @click.option("--resize",     is_flag=True)
    @click.option("--benchmark",  is_flag=True)
    @click.option("--save-image", is_flag=True)
    @click.option("--save-debug", is_flag=True)
    @click.option("--verbose",    is_flag=True)
    @click.pass_context
    def command(
        ctx,
        config    : str,
        arch      : str,
        model     : str,
        data      : str,
        root      : str,
        project   : str,
        variant   : str,
        fullname  : str,
        save_dir  : str,
        weights   : str,
        device    : int | list[int] | str,
        imgsz     : int,
        resize    : bool,
        benchmark : bool,
        save_image: bool,
        save_debug: bool,
        verbose   : bool,
    ) -> argparse.Namespace:
        hostname = socket.gethostname().lower()
        
        # Get extra args
        extra_args = {
            k.lstrip("--"): ctx.args[i + 1]
            if not (i + 1 >= len(ctx.args) or ctx.args[i + 1].startswith("--")) else True
            for i, k in enumerate(ctx.args) if k.startswith("--")
        }
        
        # Get config args
        config = mon.parse_config_file(project_root=mon.Path(root) / "config", config=config)
        args   = mon.load_config(config)
        
        # Prioritize input args --> config file args
        project    = project    or args.get("project")
        variant    = variant    or args.get("variant")
        fullname   = fullname   or args.get("fullname")
        save_dir   = save_dir   or args.get("save_dir")
        weights    = weights    or args.get("weights")
        device     = device     or args.get("device")
        imgsz      = imgsz      or args.get("imgsz")
        resize     = resize     or args.get("resize")
        benchmark  = benchmark  or args.get("benchmark")
        save_image = save_image or args.get("save_image")
        save_debug = save_debug or args.get("save_debug")
        verbose    = verbose    or args.get("verbose")
        
        # Parse arguments
        root     = mon.Path(root)
        save_dir = save_dir or mon.parse_save_dir(root/"run"/"predict", arch, model, None, project, variant)
        save_dir = mon.Path(save_dir)
        weights  = mon.to_list(weights)
        device   = mon.parse_device(device)
        imgsz    = mon.parse_hw(imgsz)
        
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
        args["imgsz"]      = imgsz
        args["resize"]     = resize
        args["benchmark"]  = benchmark
        args["save_image"] = save_image
        args["save_debug"] = save_debug
        args["verbose"]    = verbose
        args |= extra_args
        args  = argparse.Namespace(**args)
        return args
    
    return command()
'''

# endregion

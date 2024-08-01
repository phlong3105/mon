#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import socket

import click

import mon
from mon import core

console      = core.console
current_file = core.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# region Predict


# endregion


# region Main

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
def main(
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
) -> str:
    hostname = socket.gethostname().lower()
    
    # Get config args
    config   = mon.parse_config_file(project_root=root, config=config)
    args     = mon.load_config(config)
    
    # Prioritize input args --> config file args
    model      = core.Path(model or args.get("model"))
    model      = model if model.exists() else current_dir / "config"  / model.name
    model      = model.config_file()
    data_      = core.Path(args.get("data"))
    data_      = data_ if data_.exists() else current_dir / "data" / data_.name
    data_      = data_.config_file()
    data       = data       or args.get("source")
    root       = root       or args.get("root")
    root       = core.Path(root)
    project    = project    or args.get("project")
    variant    = variant    or args.get("variant")
    fullname   = fullname   or args.get("name")
    save_dir   = save_dir   or mon.parse_save_dir(root/"run"/"predict", arch, model, None, project, variant)
    save_dir   = core.Path(save_dir)
    weights    = weights    or args.get("weights")
    device     = device     or args.get("device")
    imgsz      = imgsz      or args.get("imgsz")
    resize     = resize     or args.get("resize")
    benchmark  = benchmark  or args.get("benchmark")
    save_image = save_image or args.get("save_image")
    save_debug = save_debug or args.get("save_debug")
    verbose    = verbose    or args.get("verbose")
    
    # Update arguments
    args["hostname"]   = hostname
    args["config"]     = config
    args["arch"]       = arch
    args["model"]      = str(model)
    args["data"]       = str(data_)
    args["source"]     = data
    args["root"]       = root
    args["project"]    = project
    args["variant"]    = variant
    args["name"]       = fullname
    args["fullname"]   = fullname
    args["save_dir"]   = save_dir
    args["weights"]    = core.to_list(weights)
    args["device"]     = device
    args["imgsz"]      = core.to_list(imgsz)
    args["resize"]     = resize
    args["benchmark"]  = benchmark
    args["save_image"] = save_image
    args["save_debug"] = save_debug
    args["verbose"]    = verbose
    
    opt = argparse.Namespace(**args)
    
    return str(opt.save_dir)
    

if __name__ == "__main__":
    main()

# endregion

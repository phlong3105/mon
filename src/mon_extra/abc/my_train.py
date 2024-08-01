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


# region Train


# endregion


# region Main

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
def main(
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
) -> str:
    hostname = socket.gethostname().lower()
    
    # Get config args
    config   = mon.parse_config_file(project_root=root, config=config)
    args     = mon.load_config(config)
    
    # Prioritize input args --> config file args
    model    = core.Path(model or args.get("model"))
    # model    = model if model.exists() else current_dir / "config"  / model.name
    # model    = model.config_file()
    data     = core.Path(args.get("data"))
    root     = root      or args.get("root")
    root     = core.Path(root)
    project  = project   or args.get("project")
    variant  = variant   or args.get("variant")
    fullname = fullname  or args.get("fullname")
    save_dir = save_dir  or mon.parse_save_dir(root/"run"/"train", arch, model, data, project, variant)
    save_dir = core.Path(save_dir)
    weights  = weights   or args.get("weights")
    device   = device    or args.get("device")
    epochs   = epochs    or args.get("epochs")
    exist_ok = exist_ok  or args.get("exist_ok")
    verbose  = verbose   or args.get("verbose")
    
    # Update arguments
    args["hostname"]   = hostname
    args["config"]     = config
    args["arch"]       = arch
    args["model"]      = str(model)
    args["data"]       = str(data)
    args["root"]       = root
    args["project"]    = project
    args["variant"]    = variant
    args["name"]       = fullname
    args["fullname"]   = fullname
    args["save_dir"]   = save_dir
    args["weights"]    = weights
    args["device"]     = device
    args["epochs"]     = epochs
    args["steps"]      = steps
    args["exist_ok"]   = exist_ok
    args["verbose"]    = verbose
    
    opt = argparse.Namespace(**args)
    
    if not exist_ok:
        core.delete_dir(paths=core.Path(opt.save_dir))
    core.Path(opt.save_dir).mkdir(parents=True, exist_ok=True)
    
    # Call train()
    
    return str(opt.save_dir)
        

if __name__ == "__main__":
    main()

# endregion

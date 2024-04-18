#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import socket

import click

from mon import core

console       = core.console
_current_file = core.Path(__file__).absolute()
_current_dir  = _current_file.parents[0]


# region Train


# endregion


# region Main

@click.command(name="train", context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.option("--root",       type=str, default=None, help="Project root.")
@click.option("--config",     type=str, default=None, help="Model config.")
@click.option("--weights",    type=str, default=None, help="Weights paths.")
@click.option("--model",      type=str, default=None, help="Model name.")
@click.option("--fullname",   type=str, default=None, help="Save results to root/run/train/fullname.")
@click.option("--save-dir",   type=str, default=None, help="Optional saving directory.")
@click.option("--device",     type=str, default=None, help="Running devices.")
@click.option("--epochs",     type=int, default=None, help="Stop training once this number of epochs is reached.")
@click.option("--steps",      type=int, default=None, help="Stop training once this number of steps is reached.")
@click.option("--exist-ok",   is_flag=True)
@click.option("--verbose",    is_flag=True)
def main(
    root      : str,
    config    : str,
    weights   : str,
    model     : str,
    fullname  : str,
    save_dir  : str,
    device    : str,
    epochs    : int,
    steps     : int,
    exist_ok  : bool,
    verbose   : bool,
) -> str:
    hostname = socket.gethostname().lower()
    
    # Get config args
    config = core.parse_config_file(project_root=_current_dir / "config", config=config)
    args   = core.load_config(config)
    
    # Prioritize input args --> config file args
    root     = root or args["root"]
    root     = core.Path(root)
    weights  = weights or args["weights"]
    model    = core.Path(model or args["model"])
    # model    = model if model.exists() else _current_dir / "config"  / model.name
    # model    = model.config_file()
    data     = core.Path(args["data"])
    project  = root.name or args["project"]
    fullname = fullname  or args["name"]
    save_dir = save_dir  or root / "run" / "train" / fullname
    save_dir = core.Path(save_dir)
    weights  = weights   or args["weights"]
    device   = device    or args["device"]
    epochs   = epochs    or args["epochs"]
    exist_ok = exist_ok  or args["exist_ok"]
    verbose  = verbose   or args["verbose"]
    
    # Update arguments
    args["root"]       = root
    args["config"]     = config
    args["weights"]    = weights
    args["model"]      = str(model)
    args["data"]       = str(data)
    args["root"]       = root
    args["project"]    = project
    args["name"]       = fullname
    args["save_dir"]   = save_dir
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

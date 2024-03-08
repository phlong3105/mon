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


# region Predict


# endregion


# region Main

@click.command(name="predict", context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.option("--root",       type=str, default=None, help="Project root.")
@click.option("--config",     type=str, default=None, help="Model config.")
@click.option("--weights",    type=str, default=None, help="Weights paths.")
@click.option("--model",      type=str, default=None, help="Model name.")
@click.option("--data",       type=str, default=None, help="Source data directory.")
@click.option("--fullname",   type=str, default=None, help="Save results to root/run/predict/fullname.")
@click.option("--save-dir",   type=str, default=None, help="Optional saving directory.")
@click.option("--device",     type=str, default=None, help="Running devices.")
@click.option("--imgsz",      type=int, default=None, help="Image sizes.")
@click.option("--resize",     is_flag=True)
@click.option("--benchmark",  is_flag=True)
@click.option("--save-image", is_flag=True)
@click.option("--verbose",    is_flag=True)
def main(
    root      : str,
    config    : str,
    weights   : str,
    model     : str,
    data      : str,
    fullname  : str,
    save_dir  : str,
    device    : str,
    imgsz     : int,
    resize    : bool,
    benchmark : bool,
    save_image: bool,
    verbose   : bool,
) -> str:
    hostname = socket.gethostname().lower()
    
    # Get config args
    config = core.parse_config_file(project_root=_current_dir / "config", config=config)
    args   = core.load_config(config)
    
    # Prioritize input args --> config file args
    root     = root      or args["root"]
    root     = core.Path(root)
    weights  = weights   or args["weights"]
    model    = core.Path(model or args["model"])
    model    = model if model.exists() else _current_dir / "config"  / model.name
    model    = model.config_file()
    data_    = core.Path(args["data"])
    data_    = data_ if data_.exists() else _current_dir / "data" / data_.name
    data_    = data_.config_file()
    data     = data      or args["source"]
    project  = root.name or args["project"]
    fullname = fullname  or args["name"]
    save_dir = save_dir  or root / "run" / "predict" / model
    save_dir = core.Path(save_dir)
    device   = device    or args["device"]
    imgsz    = imgsz     or args["imgsz"]
    verbose  = verbose   or args["verbose"]
    
    # Update arguments
    args["root"]     = root
    args["config"]   = config
    args["weights"]  = core.to_list(weights)
    args["model"]    = str(model)
    args["data"]     = str(data_)
    args["source"]   = data
    args["project"]  = project
    args["name"]     = fullname
    args["save_dir"] = save_dir
    args["device"]   = device
    args["imgsz"]    = core.to_list(imgsz)
    args["verbose"]  = verbose
    
    opt = argparse.Namespace(**args)
    
    return str(opt.save_dir)
    

if __name__ == "__main__":
    main()

# endregion

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the prediction scripts for YOLOv8."""

from __future__ import annotations

import socket

import click

import ultralytics
from mon import core, DATA_DIR
from ultralytics import YOLO

console       = core.console
_current_file = core.Path(__file__).absolute()
_current_dir  = _current_file.parents[0]

ultralytics.utils.DATASETS_DIR = DATA_DIR


# region Predict

def predict(args: dict):
    model    = YOLO(args["model"])
    _project = args.pop("project")
    _name    = args.pop("name")
    project  = f"{_project}/{_name}"
    sources  = args.pop("source")
    sources  = [sources] if not isinstance(sources, list) else sources
    for source in sources:
        path = core.Path(source)
        name = path.parent.name if path.name == "images" else path.name
        _    = model(source=source, project=f"{project}", name=name, **args)

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
    config = core.parse_config_file(project_root=_current_dir.parent / "config", config=config)
    args   = core.load_config(config)
    
    # Prioritize input args --> config file args
    root       = root     or args["root"]
    root       = core.Path(root)
    weights    = weights  or args["model"]
    # weights    = weights[0] if isinstance(weights, list | tuple) and len(weights) == 1 else weights
    data       = data     or args["source"]
    fullname   = fullname or args["name"]
    save_dir   = save_dir or root / "run" / "train" / model
    save_dir   = core.Path(save_dir)
    device     = device   or args["device"]
    imgsz      = imgsz    or args["imgsz"]
    resize     = resize
    benchmark  = benchmark
    save_image = save_image
    verbose    = verbose
    
    # Update arguments
    args["root"]     = root
    args["mode"]    = "predict"
    args["model"]   = weights
    args["project"] = str(save_dir.parent)
    args["name"]    = str(save_dir.name)
    args["imgsz"]   = imgsz
    args["device"]  = device
    args["verbose"] = verbose
    args["source"]  = data
    
    predict(args=args)
    return str(save_dir)
    

if __name__ == "__main__":
    main()

# endregion

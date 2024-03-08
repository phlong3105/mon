#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the prediction scripts for YOLOv8."""

from __future__ import annotations

import socket

import click

import ultralytics
from mon import core, DATA_DIR
from ultralytics import YOLO

ultralytics.utils.DATASETS_DIR = DATA_DIR

console       = core.console
_current_file = core.Path(__file__).absolute()
_current_dir  = _current_file.parents[0]


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
@click.option("--config",     type=str, default=None, help="Model config.")
@click.option("--model",      type=str, default=None, help="Model name.")
@click.option("--data",       type=str, default=None, help="Source data directory.")
@click.option("--root",       type=str, default=None, help="Save results to root/project/fullname.")
@click.option("--project",    type=str, default=None, help="Save results to root/project/fullname.")
@click.option("--fullname",   type=str, default=None, help="Save results to root/project/fullname.")
@click.option("--save-dir",   type=str, default=None, help="Optional saving directory.")
@click.option("--weights",    type=str, default=None, multiple=True, help="Weights paths.")
@click.option("--device",     type=str, default=None, help="Running devices.")
@click.option("--imgsz",      type=int, default=None, multiple=True, help="Image sizes.")
@click.option("--resize",     is_flag=True)
@click.option("--benchmark",  is_flag=True)
@click.option("--save-image", is_flag=True)
@click.option("--verbose",    is_flag=True)
def main(
    config    : str,
    model     : str,
    data      : str,
    root      : str,
    project   : str,
    fullname  : str,
    save_dir  : str,
    weights   : str,
    device    : str,
    imgsz     : int,
    resize    : bool,
    benchmark : bool,
    save_image: bool,
    verbose   : bool,
) -> str:
    hostname = socket.gethostname().lower()
    
    # Get config args
    config_file = core.parse_config_file(project=project, config=config, name=model)
    config_file = config_file if config_file.exists() else _current_dir / "cfg" / core.Path(config_file).name
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    args = core.load_config_from_file(config_file)
    
    # Prioritize input args --> config file args
    data_      = core.Path(args["data"])
    data_      = data_ if data_.exists() else _current_dir / "ultralytics/cfg/datasets" / data_.name
    data_      = core.Path(data_).config_file()
    data       = data     or args["source"]
    project    = project  or args["project"]
    fullname   = fullname or args["name"]
    save_dir   = save_dir if save_dir not in [None, "None", ""] else core.Path(root) / project / fullname
    save_dir   = core.Path(save_dir)
    weights    = weights  or args["model"]
    device     = device   or args["device"]
    device     = core.parse_device(device)
    imgsz      = imgsz    or args["imgsz"]
    resize     = resize
    benchmark  = benchmark
    save_image = save_image
    verbose    = verbose
    
    # Update arguments
    args["mode"]    = "predict"
    args["model"]   = core.to_list(weights)
    args["data"]    = str(data_)
    args["source"]  = data
    args["project"] = str(save_dir.parent)
    args["name"]    = str(save_dir.name)
    args["imgsz"]   = imgsz
    args["device"]  = device
    args["verbose"] = verbose
    
    predict(args=args)
    
    return str(save_dir)


if __name__ == "__main__":
    main()

# endregion

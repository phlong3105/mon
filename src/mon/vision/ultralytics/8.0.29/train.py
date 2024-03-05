#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the training scripts for YOLOv8."""

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


# region Train

def train(args: dict):
    model = YOLO(args["model"])
    _     = model.train(**args)

# endregion


# region Main

@click.command(name="train", context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.option("--config",   type=str, default=None, help="Model config.")
@click.option("--model",    type=str, default=None, help="Model name.")
@click.option("--data",     type=str, default=None, help="Source data directory.")
@click.option("--root",     type=str, default=None, help="Save results to root/project/fullname.")
@click.option("--project",  type=str, default=None, help="Save results to root/project/fullname.")
@click.option("--fullname", type=str, default=None, help="Save results to root/project/fullname.")
@click.option("--save-dir", type=str, default=None, help="Optional saving directory.")
@click.option("--weights",  type=str, default=None, help="Weights paths.")
@click.option("--device",   type=str, default=None, multiple=True, help="Running devices.")
@click.option("--epochs",   type=int, default=None, help="Stop training once this number of epochs is reached.")
@click.option("--steps",    type=int, default=None, help="Stop training once this number of steps is reached.")
@click.option("--exist-ok", is_flag=True)
@click.option("--verbose",  is_flag=True)
def main(
    config  : str,
    model   : str,
    data    : str,
    root    : str,
    project : str,
    fullname: str,
    save_dir: str,
    weights : str,
    device  : int | list[int] | str,
    epochs  : int,
    steps   : int,
    exist_ok: bool,
    verbose : bool,
) -> str:
    hostname = socket.gethostname().lower()
    
    # Get config args
    config_file = core.parse_config_file(project=project, config=config, name=model)
    config_file = config_file if config_file.exists() else _current_dir / "cfg" / core.Path(config_file).name
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    args = core.load_config_from_file(config_file)
    
    # Prioritize input args --> config file args
    data     = core.Path(data or args["data"])
    data     = data  if data.exists() else _current_dir / "ultralytics/cfg/datasets" / data.name
    data     = core.Path(data).config_file()
    project  = project  or args["project"]
    fullname = fullname or args["name"]
    save_dir = save_dir if save_dir not in [None, "None", ""] else core.Path(root) / project / fullname
    save_dir = core.Path(save_dir)
    weights  = weights  or args["model"]
    device   = device   or args["device"]
    epochs   = epochs   or args["epochs"]
    exist_ok = exist_ok or args["exist_ok"]
    verbose  = verbose  or args["verbose"]
    
    # Update arguments
    args["mode"]     = "train"
    args["model"]    = core.to_list(weights)
    args["data"]     = str(data)
    args["project"]  = str(save_dir.parent)
    args["name"]     = str(save_dir.name)
    args["epochs"]   = epochs
    args["device"]   = device
    args["exist_ok"] = exist_ok
    args["verbose"]  = verbose
    
    if not exist_ok:
        core.delete_dir(paths=core.Path(save_dir))
    
    train(args=args)
    return str(save_dir)


if __name__ == "__main__":
    main()

# endregion

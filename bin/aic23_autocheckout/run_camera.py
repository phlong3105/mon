#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This script runs camera."""

from __future__ import annotations

import click

import mon
from supr import DATA_DIR, io
from supr.config import aic23_autocheckout as autocheckout
from supr.globals import CAMERAS

console = mon.console


# region Function

@click.command()
@click.option(
    "--config", default="testA",
    type=click.Choice([
        "testA_1", "testA_2", "testA_3", "testA_4", "testA_5", "testA",
        "testB_1", "testB_2", "testB_3", "testB_4", "testB_5", "testB",
    ], case_sensitive=True),
    help="Camera configuration."
)
def run_camera(config: mon.Path | str):
    config_files = dir(autocheckout)
    if config in ["all", "*"]:
        config_files = [f for f in config_files if "test" in f]
    else:
        config_files = [f for f in config_files if config in f]
    
    for cf in config_files:
        args   = mon.get_module_vars(getattr(autocheckout, cf))
        camera = args.pop("camera")
        camera = CAMERAS.get(camera)
        camera = camera(**args)
        camera.run()
    
    subset = "testA" if "testA" in config else "testB"
    io.AIC23AutoCheckoutWriter.merge_results(
        output_dir  = DATA_DIR/"aic23-autocheckout"/subset/"result",
        output_name = "track4.txt",
        subset      = subset
    )
    
# endregion


# region Main

if __name__ == "__main__":
    run_camera()

# endregion

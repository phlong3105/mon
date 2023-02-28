#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This script runs camera."""

from __future__ import annotations

import click

import mon
from supr.config import aic23_autocheckout as autocheckout
from supr.globals import CAMERAS

console = mon.console


# region Function

@click.command()
@click.option(
    "--config", default="testA_3",
    type=click.Choice([
        "testA", "testA_1", "testA_2", "testA_3", "testA_4", "testA_5", "all",
        "testB", "testB_1", "testB_2", "testB_3", "testB_4", "testA_5", "all",
    ], case_sensitive=True),
    help="Camera configuration."
)
def run_camera(config: mon.Path | str):
    config_files = dir(autocheckout)
    if config == "all":
        config_files = [f for f in config_files if "test" in f]
    else:
        config_files = [f for f in config_files if config in f]
    
    for cf in config_files:
        config = mon.get_module_vars(getattr(autocheckout, cf))
        camera = config.pop("camera")
        camera = CAMERAS.get(camera)
        camera = camera(**config)
        camera.run()
        
# endregion


# region Main

if __name__ == "__main__":
    run_camera()

# endregion

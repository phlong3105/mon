#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This script runs camera."""

from __future__ import annotations

from importlib import import_module

import click
import cv2

import mon
import supr
from supr.config import aic23_checkout

console = mon.console


# region Function

@click.command()
@click.option(
    "--config", default="testA_1",
    type=click.Choice([
        "testA", "testA_1", "testA_2", "testA_3", "testA_4",
        "testB", "testB_1", "testB_2", "testB_3", "testB_4",
    ], case_sensitive=True),
    help="Camera configuration."
)
def run_camera(config: mon.Path | str):
    config_files = dir(aic23_checkout)
    config_files = [f for f in config_files if config in f]
    
    for cf in config_files:
        config = mon.get_module_vars(getattr(aic23_checkout, cf))
        camera = supr.CheckoutCamera(**config)
        camera.run()
        
# endregion


# region Main

if __name__ == "__main__":
    run_camera()

# endregion

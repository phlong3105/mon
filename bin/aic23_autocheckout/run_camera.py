#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This script runs camera."""

from __future__ import annotations

import click
from timeit import default_timer as timer

import mon
from supr import DATA_DIR, io
from supr.config import aic23_autocheckout as autocheckout
from supr.globals import CAMERAS

console = mon.console


# region Function

@click.command()
@click.option(
    "--config", default="testA_1",
    type=click.Choice([
        "testA_1", "testA_2", "testA_3", "testA_4", "testA_5", "testA",
        "testB_1", "testB_2", "testB_3", "testB_4", "testB_5", "testB",
    ], case_sensitive=True),
    help="Camera configuration."
)
@click.option("--batch-size", default=8, type=int, help="Batch size.")
@click.option("--queue-size", default=1, type=int, help="Queue size for multi-threading.")
def run_camera(
    config    : mon.Path | str,
    batch_size: int,
    queue_size: int,
):
    # Start timer
    process_start_time = timer()
    camera_init_time   = 0
    
    config_files = dir(autocheckout)
    if config in ["all", "*"]:
        config_files = [f for f in config_files if "test" in f]
    else:
        config_files = [f for f in config_files if config in f]
      
    for cf in config_files:
        # Measure camera initialize time
        camera_start_time = timer()
        args = mon.get_module_vars(getattr(autocheckout, cf))
        args["queue_size"]                 = queue_size
        args["image_loader"]["batch_size"] = batch_size
        camera = args.pop("camera") if queue_size <= 1 else "auto_checkout_camera_multithread"
        camera = CAMERAS.get(camera)
        camera = camera(**args)
        camera_init_time += timer() - camera_start_time
        # Process
        camera.run()
    
    # End timer
    process_time = timer() - process_start_time
    console.log(f"Total processing time: {process_time} seconds.")
    console.log(f"Total camera init time: {camera_init_time} seconds.")
    console.log(f"Actual processing time: {process_time - camera_init_time} seconds.")
    
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

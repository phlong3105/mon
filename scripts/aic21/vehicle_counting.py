#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import argparse
import os
from timeit import default_timer as timer

from aic import AICVehicleCountingCamera
from one import console
from one import load_config


# MARK: - Main Function

def main(args):
    # NOTE: Start timer
    process_start_time = timer()
    camera_start_time  = timer()
    
    # NOTE: Parse camera config
    data_dir    = os.path.abspath("../data")
    config_path = os.path.join(data_dir, args.dataset, "configs", args.config)
    camera_cfg  = load_config(config_path)
    # Update value from args
    camera_cfg.DATA_DIR        = data_dir
    camera_cfg.dataset         = args.dataset
    camera_cfg.subset          = args.subset
    camera_cfg.verbose         = args.verbose
    camera_cfg.save_image      = args.save_image
    camera_cfg.save_video      = args.save_video
    camera_cfg.save_results    = args.save_results
    camera_cfg.data.batch_size = args.batch_size
    
    # NOTE: Define camera
    camera           = AICVehicleCountingCamera(**camera_cfg)
    camera_init_time = timer() - camera_start_time
    
    # NOTE: Process
    camera.run()
    
    # NOTE: End timer
    total_process_time = timer() - process_start_time
    console.log(f"Total processing time: {total_process_time} seconds.")
    console.log(f"Camera init time: {camera_init_time} seconds.")
    console.log(f"Actual processing time: {total_process_time - camera_init_time} seconds.")


def parse_args():
    parser = argparse.ArgumentParser(description="Config parser")
    parser.add_argument("--dataset", default="aic21vehiclecounting", help="Dataset to run on.")
    parser.add_argument("--subset",  default="dataset_a",            help="Subset name. One of: [`dataset_a`, `dataset_b`].")
    parser.add_argument(
        "--config",  default="cam_1.yaml",
        help="Config file for each camera. Final path to the config file is: tss/data/<dataset>/configs/<config>/"
    )
    parser.add_argument("--batch_size",   default=32,    type=int, help="Max batch size")
    parser.add_argument("--verbose",      default=True,  help="Should visualize the images.")
    parser.add_argument("--save_image",   default=False, help="Should save results to images.")
    parser.add_argument("--save_video",   default=True,  help="Should save results to a video.")
    parser.add_argument("--save_results", default=True,  help="Should save results to file.")
    args = parser.parse_args()
    return args


# MARK: - Main

if __name__ == "__main__":
    main(parse_args())

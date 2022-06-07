#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
configs_yolov4p5_448: 714.0099547459977 seconds.
configs_yolov4p5_896: 798.4888908839785 seconds.
configs_yolov4p7_896: 823.755982607021 seconds.
configs_yolov5s6_640: 979.2011659840355 seconds.
"""

from __future__ import annotations

import argparse
import os
from timeit import default_timer as timer

from aic import AIC22RetailCheckoutCamera
from aic import AIC22RetailCheckoutWriter
from one import console
from one import load_config

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

configs_a = [
    "testA_1.yaml",
    "testA_2.yaml",
    "testA_3.yaml",
    "testA_4.yaml",
    "testA_5.yaml"
]
configs_b = [
    "testB_1.yaml",
    "testB_2.yaml",
    "testB_3.yaml",
    "testB_4.yaml",
    "testB_5.yaml"
]


# MARK: - Main Function

def main(args):
    data_dir = os.path.abspath("../data")
    
    # NOTE: Start timer
    process_start_time = timer()
    camera_start_time  = timer()
    camera_init_time   = timer() - camera_start_time

    if args.subset not in ["test_a", "test_b"]:
        raise ValueError(f"`subset` must be one of ['test_a', 'test_b']. "
                         f"But got: {args.subset}.")
    if args.subset == "test_a":
        configs = configs_a
    else:
        configs = configs_b
    
    for cfg in configs:
        # NOTE: Parse camera config
        args.config = cfg
        config_path = os.path.join(data_dir, args.dataset, args.configs, args.config)
        camera_cfg  = load_config(config_path)
        # Update value from args
        camera_cfg.data_dir        = data_dir
        camera_cfg.dataset         = args.dataset
        camera_cfg.subset          = args.subset
        camera_cfg.verbose         = args.verbose
        camera_cfg.save_image      = args.save_image
        camera_cfg.save_video      = args.save_video
        camera_cfg.save_results    = args.save_results
        camera_cfg.data.batch_size = args.batch_size
        
        # NOTE: Define camera
        camera = AIC22RetailCheckoutCamera(**camera_cfg)
        
        # NOTE: Process
        camera.run()
    
    # NOTE: End timer
    total_process_time = timer() - process_start_time
    console.log(f"Total processing time: {total_process_time} seconds.")
    console.log(f"Camera init time: {camera_init_time} seconds.")
    console.log(f"Actual processing time: {total_process_time - camera_init_time} seconds.")

    AIC22RetailCheckoutWriter.compress_all_results(
        output_dir=os.path.join(data_dir, args.dataset, "output"),
        subset=args.subset
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Config parser")
    parser.add_argument("--dataset",      default="aic22retail", help="Dataset to run on.")
    parser.add_argument("--subset",       default="test_a",      help="Subset name. One of: [`test_a`, `test_b`].")
    parser.add_argument("--configs",      default="configs_yolov4p7_1536",)
    parser.add_argument("--batch_size",   default=1,     type=int, help="Max batch size")
    parser.add_argument("--verbose",      default=True, help="Should visualize the images.")
    parser.add_argument("--save_image",   default=False, help="Should save results to images.")
    parser.add_argument("--save_video",   default=False, help="Should save results to a video.")
    parser.add_argument("--save_results", default=True,  help="Should save results to file.")
    args = parser.parse_args()
    return args


# MARK: - Main

if __name__ == "__main__":
    main(parse_args())

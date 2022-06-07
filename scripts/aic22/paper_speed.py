#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import argparse
import csv
import os
from timeit import default_timer as timer

from aic import AIC22RetailCheckoutCamera
from aic import AIC22RetailCheckoutWriter
from one import console
from one import load_config

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

configs = [
    "testA_1.yaml",
    "testA_2.yaml",
    "testA_3.yaml",
    "testA_4.yaml",
    "testA_5.yaml"
]


# MARK: - Main Function

def main(args):
    data_dir = os.path.abspath("../data")

    with open('speed.csv', 'w') as csvfile:
        fieldnames = ['batch_size', 'queue_size', "run_time"]
        writer     = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        batch_sizes = [1, 8, 16, 32, 64]
        queue_sizes = [1]
        for b in batch_sizes:
            for q in queue_sizes:

                # NOTE: Start timer
                process_start_time = timer()
                camera_start_time  = timer()
                camera_init_time   = timer() - camera_start_time
                
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
                    camera_cfg.data.batch_size = b  # args.batch_size
                    camera_cfg.queue_size      = q
                   
                    # NOTE: Define camera
                    # camera = AIC22RetailCheckoutCameraAsync(**camera_cfg)
                    camera = AIC22RetailCheckoutCamera(**camera_cfg)
                    
                    # NOTE: Process
                    camera.run()
                
                # NOTE: End timer
                total_process_time = timer() - process_start_time
                console.log(f"batch_size {b}, queue_size {q}:")
                console.log(f"Total processing time: {total_process_time} seconds.")
                console.log(f"Camera init time: {camera_init_time} seconds.")
                console.log(f"Actual processing time: {total_process_time - camera_init_time} seconds.")
                console.log("")
                
                AIC22RetailCheckoutWriter.compress_all_results(output_dir=os.path.join(data_dir, args.dataset, "output"))

                writer.writerow({'batch_size': f'{b}', 'queue_size': f'{q}', "run_time": f"{total_process_time - camera_init_time}"})
                

def parse_args():
    parser = argparse.ArgumentParser(description="Config parser")
    parser.add_argument("--dataset", default="aic22retail", help="Dataset to run on.")
    parser.add_argument("--subset",  default="test_a",      help="Subset name. One of: [`test_a`, `test_b`].")
    parser.add_argument("--configs", default="configs_yolov4p7_896", )
    parser.add_argument(
        "--config", default="testA_5.yaml",
        help="Config file for each camera. Final path to the config file is: ../data/<dataset>/configs/<config>/"
    )
    parser.add_argument("--batch_size",   default=8,    type=int, help="Max batch size")
    parser.add_argument("--queue_size",   default=16,    type=int, help="Max queue size")
    parser.add_argument("--verbose",      default=False, help="Should visualize the images.")
    parser.add_argument("--save_image",   default=False, help="Should save results to images.")
    parser.add_argument("--save_video",   default=False, help="Should save results to a video.")
    parser.add_argument("--save_results", default=False, help="Should save results to file.")
    args = parser.parse_args()
    return args


# MARK: - Main

if __name__ == "__main__":
    main(parse_args())

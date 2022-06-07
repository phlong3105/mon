#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Training scripts
"""

from __future__ import annotations

import argparse
import os

from chalearn import pretrained_dir
from chalearn.ltd.scaled_yolov4 import run_train


# MARK: - Main

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre_cfg",      default="",                                            type=str, help="Predefined configs.")
    parser.add_argument("--weights",      default=os.path.join(pretrained_dir, "scaled_yolov4", "yolov4_p7_coco.pt"), type=str, help="initial weights path")
    parser.add_argument("--cfg",          default="",                                            type=str, help="model.yaml path")
    parser.add_argument("--data",         default="chalearnltdmonth.yaml",                       type=str, help="data.yaml path")
    parser.add_argument("--hyp",          default="",                                            type=str, help="hyperparameters path, i.e. data/hyp.scratch.yaml")
    parser.add_argument("--epochs",       default=300,                                           type=int)
    parser.add_argument("--batch-size",   default=16,                                            type=int, help="total batch size for all GPUs")
    parser.add_argument("--img-size",     default=[640, 640],                                    nargs="+", type=int, help="train,test sizes")
    parser.add_argument("--rect",         default=False,                                         action="store_true", help="rectangular training")
    parser.add_argument("--resume",       default=False,                                         nargs="?", const="get_last", help="resume from given path/last.pt, or most recent run if blank")
    parser.add_argument("--nosave",       default=False,                                         action="store_true", help="only save final checkpoint")
    parser.add_argument("--notest",       default=False,                                         action="store_true", help="only test final epoch")
    parser.add_argument("--noautoanchor", default=False,                                         action="store_true", help="disable autoanchor check")
    parser.add_argument("--evolve",       default=False,                                         action="store_true", help="evolve hyperparameters")
    parser.add_argument("--bucket",       default="",                                            type=str,  help="gsutil bucket")
    parser.add_argument("--cache-images", default=False,                                         action="store_true", help="cache images for faster training")
    parser.add_argument("--name",         default="",                                            help="renames results.txt to results_name.txt if supplied")
    parser.add_argument("--device",       default="",                                            help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--multi-scale",  default=False,                                         action="store_true", help="vary img-size +/- 50%%")
    parser.add_argument("--single-cls",   default=False,                                         action="store_true", help="train as single-class dataset")
    parser.add_argument("--adam",         default=False,                                         action="store_true", help="use torch.optim.Adam() optimizer")
    parser.add_argument("--sync-bn",      default=False,                                         action="store_true", help="use SyncBatchNorm, only available in DDP mode")
    parser.add_argument("--local_rank",   default=-1,                                            type=int,  help="DDP parameter, do not modify")
    parser.add_argument("--logdir",       default=os.path.join(pretrained_dir, "scaled_yolov4"), type=str,  help="logging directory")
    parser.add_argument("--verbose",      default="False",                                       type=bool, help="Verbosity")
    args = parser.parse_args()
    print(args)
    return args


if __name__ == "__main__":
    args = parse_args()
    run_train(args)

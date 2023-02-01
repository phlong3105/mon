#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the training scripts for YOLOv8."""

from __future__ import annotations

import argparse

import munch

import mon
from ultralytics import YOLO

_CURRENT_DIR = mon.Path(__file__).absolute().parent


# region Function

def train(args: munch.Munch):
    # Load a model
    model = args.pop("model")
    model = YOLO(model)  # load a pretrained model (recommended for training)
    
    # Train the model
    _ = model.train(**args)

# endregion


# region Main

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        default = "detect",
        help    = "Inference task, i.e. detect, segment, or classify."
    )
    parser.add_argument(
        "--mode",
        default = "train",
        help    = "YOLO mode, i.e. train, val, predict, or export."
    )
    parser.add_argument(
        "--resume",
        action  = "store_true",
        default = False,
        help    = "Resume training from last checkpoint or custom checkpoint "
                  "if passed as resume=path/to/best.pt."
    )
    parser.add_argument(
        "--model",
        default = "weight/yolov8n.pt",
        help    = "Path to model file, i.e. yolov8n.pt, yolov8n.yaml."
    )
    parser.add_argument(
        "--data",
        default = "data/visdrone-a2i2-of.yaml",
        help    = "Path to data file, i.e. i.e. coco128.yaml."
    )
    parser.add_argument(
        "--epochs",
        type    = int,
        default = 100,
        help    = "Number of epochs to train for."
    )
    parser.add_argument(
        "--patience",
        default = 50,
        help    = "Epochs to wait for no observable improvement for early "
                  "stopping of training."
    )
    parser.add_argument(
        "--batch",
        type    = int,
        default = -1,
        help    = "Number of images per batch (-1 for AutoBatch)."
    )
    parser.add_argument(
        "--imgsz",
        nargs   = '+',
        type    = int,
        default = 1280,
        help    = "Size of input images as integer or w,h."
    )
    parser.add_argument(
        "--save",
        action = "store_true",
        help   = "Save train checkpoints and predict results."
    )
    parser.add_argument(
        "--cache",
        action = "store_true",
        help   = "True/ram, disk or False. Use cache for data loading."
    )
    parser.add_argument(
        "--device",
        default = "cpu",
        help    = "Device to run on, i.e. cuda device=0 or device=0,1,2,3 or device=cpu."
    )
    parser.add_argument(
        "--workers",
        type    = int,
        default = 8,
        help    = "Number of worker threads for data loading (per RANK if DDP)."
    )
    parser.add_argument(
        "--project",
        default = "run/train",
        help    = "Project name."
    )
    parser.add_argument(
        "--name",
        default = "exp",
        help    = "Experiment name."
    )
    parser.add_argument(
        "--exist-ok",
        action = "store_true",
        help   = "Whether to overwrite existing experiment."
    )
    parser.add_argument(
        "--pretrained",
        action = "store_true",
        help   = "Whether to use a pretrained model."
    )
    parser.add_argument(
        "--optimizer",
        default = "SGD",
        help    = "Optimizer to use, choices=['SGD', 'Adam', 'AdamW', 'RMSProp']."
    )
    parser.add_argument(
        "--verbose",
        default = False,
        help    = "Whether to print verbose output."
    )
    parser.add_argument(
        "--seed",
        default = 0,
        help    = "Random seed for reproducibility."
    )
    parser.add_argument(
        "--deterministic",
        default = True,
        help    = "Whether to enable deterministic mode."
    )
    parser.add_argument(
        "--single-cls",
        default = False,
        help    = "Train multi-class data as single-class."
    )
    parser.add_argument(
        "--image-weights",
        default = False,
        help    = "Use weighted image selection for training."
    )
    parser.add_argument(
        "--rect",
        default = False,
        help    = "Support rectangular training."
    )
    parser.add_argument(
        "--cos-lr",
        default = False,
        help    = "Use cosine learning rate scheduler."
    )
    parser.add_argument(
        "--close-mosaic",
        type    = int,
        default = 10,
        help    = "Disable mosaic augmentation for final 10 epochs."
    )
    parser.add_argument(
        "--lr0",
        type    = float,
        default = 0.01,
        help    = "Initial learning rate (i.e. SGD=1E-2, Adam=1E-3)."
    )
    parser.add_argument(
        "--lrf",
        type    = float,
        default = 0.01,
        help    = "Final learning rate (lr0 * lrf)."
    )
    parser.add_argument(
        "--momentum",
        type    = float,
        default = 0.937,
        help    = "SGD momentum/Adam beta1."
    )
    parser.add_argument(
        "--weight-decay",
        type    = float,
        default = 0.0005,
        help    = "Optimizer weight decay 5e-4."
    )
    parser.add_argument(
        "--warmup-epochs",
        type    = float,
        default = 3.0,
        help    = "Warmup epochs (fractions ok)."
    )
    parser.add_argument(
        "--warmup-momentum",
        type    = float,
        default = 0.8,
        help    = "Warmup initial momentum."
    )
    parser.add_argument(
        "--warmup-bias-lr",
        type    = float,
        default = 0.1,
        help    = "Warmup initial bias lr."
    )
    parser.add_argument(
        "--bbox",
        type    = float,
        default = 7.5,
        help    = "Box loss gain."
    )
    parser.add_argument(
        "--cls",
        type    = float,
        default = 0.5,
        help    = "Cls loss gain (scale with pixels)."
    )
    parser.add_argument(
        "--dfl",
        type    = float,
        default = 1.5,
        help    = "dfl loss gain."
    )
    parser.add_argument(
        "--fl-gamma",
        type    = float,
        default = 0.0,
        help    = "Focal loss gamma (efficientDet default gamma=1.5)."
    )
    parser.add_argument(
        "--label-smoothing",
        type    = float,
        default = 0.0,
        help    = "Label smoothing (fraction)."
    )
    parser.add_argument(
        "--nbs",
        type    = int,
        default = 64,
        help    = "Nominal batch size."
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = munch.Munch.fromDict(vars(parse_args()))
    args.mode = "train"
    train(args=args)
    
# endregion

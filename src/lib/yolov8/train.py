#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the training scripts for YOLOv8."""

from __future__ import annotations

import argparse

import click

import mon
from ultralytics import YOLO

_current_dir = mon.Path(__file__).absolute().parent


# region Function

@click.command()
@click.option("--task",            default="detect", help="Inference task, i.e. detect, segment, or classify.")
@click.option("--resume",          is_flag=True, help="Resume training from last checkpoint or custom checkpoint if passed as resume=path/to/best.pt.")
@click.option("--model",           default="weight/yolov8n.pt", type=click.Path(exists=False), help="Path to model file, i.e. yolov8n.pt, yolov8n.yaml.")
@click.option("--data",            default="data/visdrone-a2i2-of.yaml", type=click.Path(exists=True), help="Path to data file, i.e. i.e. coco128.yaml.")
@click.option("--project",         default="run/train", type=click.Path(exists=False), help="Project name.")
@click.option("--name",            default="exp", type=str, help="Experiment name.")
@click.option("--epochs",          default=100, type=int, help="Number of epochs to train for.")
@click.option("--batch",           default=-1, type=int, help="Number of images per batch (-1 for AutoBatch).")
@click.option("--imgsz",           default=1280, type=int, nargs='2', help="Size of input images as integer or w,h.")
@click.option("--patience",        default=50, type=int, help="Epochs to wait for no observable improvement for early stopping of training.")
@click.option("--workers",         default=8, type=int, help="Number of worker threads for data loading (per RANK if DDP).")
@click.option("--device",          default="cpu", help="Device to run on, i.e. cuda device=0 or device=0,1,2,3 or device=cpu.")
@click.option("--optimizer",       default="SGD", help="Optimizer to use, choices=['SGD', 'Adam', 'AdamW', 'RMSProp'].")
@click.option("--seed",            default=0, help="Random seed for reproducibility.")
@click.option("--nbs",             default=64, type=int, help="Nominal batch size.")
@click.option("--close-mosaic",    default=10, type=int, help="Disable mosaic augmentation for final 10 epochs.")
@click.option("--lr0",             default=0.01, type=float, help="Initial learning rate (i.e. SGD=1E-2, Adam=1E-3).")
@click.option("--lrf",             default=0.01, type=float, help="Final learning rate (lr0 * lrf).")
@click.option("--momentum",        default=0.937, type=float, help="SGD momentum/Adam beta1.")
@click.option("--weight-decay",    default=0.0005, type=float, help="Optimizer weight decay 5e-4.")
@click.option("--warmup-epochs",   default=3.0, type=float, help="Warmup epochs (fractions ok).")
@click.option("--warmup-momentum", default=0.8, type=float, help="Warmup initial momentum.")
@click.option("--warmup-bias-lr",  default=0.1, type=float, help="Warmup initial bias lr.")
@click.option("--box",             default=7.5, type=float, help="Box loss gain.")
@click.option("--cls",             default=0.5, type=float, help="Cls loss gain (scale with pixels).")
@click.option("--dfl",             default=1.5, type=float, help="dfl loss gain.")
@click.option("--fl-gamma",        default=0.0, type=float, help="Focal loss gamma (efficientDet default gamma=1.5).")
@click.option("--label-smoothing", default=0.0, type=float, help="Label smoothing (fraction).")
@click.option("--deterministic",   is_flag=True, help="Whether to enable deterministic mode.")
@click.option("--single-cls",      is_flag=True, help="Train multi-class data as single-class.")
@click.option("--image-weights",   is_flag=True, help="Use weighted image selection for training.")
@click.option("--rect",            is_flag=True, help="Support rectangular training.")
@click.option("--cos-lr",          is_flag=True, help="Use cosine learning rate scheduler.")
@click.option("--save",            is_flag=True, help="Save train checkpoints and predict results.")
@click.option("--cache",           is_flag=True, help="True/ram, disk or False. Use cache for data loading.")
@click.option("--exist-ok",        is_flag=True, help="Whether to overwrite existing experiment.")
@click.option("--pretrained",      is_flag=True, help="Whether to use a pretrained model.")
@click.option("--verbose",         is_flag=True, help="Whether to print verbose output.")
def train(
    task, resume, model, data, project, name, epochs, batch, imgsz, patience,
    workers, device, optimizer, seed, nbs, close_mosaic, lr0, lrf, momentum,
    weight_decay, warmup_epochs, warmup_momentum, warmup_bias_lr, box, cls, dfl,
    fl_gamma, label_smoothing, deterministic, single_cls, image_weights, rect,
    cos_lr, save, cache, exist_ok, pretrained, verbose,
):
    # Load a model
    model = YOLO(model)  # load a pretrained model (recommended for training)
    
    # Train the model
    args = {
        "task"           : task,
        "mode"           : "train",
        "resume"         : resume,
        "data"           : data,
        "project"        : project,
        "name"           : name,
        "epochs"         : epochs,
        "batch"          : batch,
        "imgsz"          : imgsz,
        "patience"       : patience,
        "workers"        : workers,
        "device"         : device,
        "optimizer"      : optimizer,
        "seed"           : seed,
        "nbs"            : nbs,
        "close_mosaic"   : close_mosaic,
        "lr0"            : lr0,
        "lrf"            : lrf,
        "momentum"       : momentum,
        "weight_decay"   : weight_decay,
        "warmup_epochs"  : warmup_epochs,
        "warmup_momentum": warmup_momentum,
        "warmup_bias_lr" : warmup_bias_lr,
        "box"            : box,
        "cls"            : cls,
        "dfl"            : dfl,
        "fl_gamma"       : fl_gamma,
        "label_smoothing": label_smoothing,
        "deterministic"  : deterministic,
        "single_cls"     : single_cls,
        "image_weights"  : image_weights,
        "rect"           : rect,
        "cos_lr"         : cos_lr,
        "save"           : save,
        "cache"          : cache,
        "exist_ok"       : exist_ok,
        "pretrained"     : pretrained,
        "verbose"        : verbose,
    }
    _    = model.train(**args)


# endregion


# region Main

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",            default="detect", help="Inference task, i.e. detect, segment, or classify.")
    parser.add_argument("--resume",          action="store_true", help="Resume training from last checkpoint or custom checkpoint if passed as resume=path/to/best.pt.")
    parser.add_argument("--model",           default="weight/yolov8n.pt", help="Path to model file, i.e. yolov8n.pt, yolov8n.yaml.")
    parser.add_argument("--data",            default="data/visdrone-a2i2-of.yaml", help="Path to data file, i.e. i.e. coco128.yaml.")
    parser.add_argument("--project",         type=str, default="run/train", help="Project name.")
    parser.add_argument("--name",            type=str, default="exp", help="Experiment name.")
    parser.add_argument("--epochs",          type=int, default=100, help="Number of epochs to train for.")
    parser.add_argument("--batch",           type=int, default=-1, help="Number of images per batch (-1 for AutoBatch).")
    parser.add_argument("--imgsz",           type=int, default=1280, nargs='+', help="Size of input images as integer or w,h.")
    parser.add_argument("--patience",        type=int, default=50, help="Epochs to wait for no observable improvement for early stopping of training.")
    parser.add_argument("--workers",         type=int, default=8, help="Number of worker threads for data loading (per RANK if DDP).")
    parser.add_argument("--device",          default="cpu", help="Device to run on, i.e. cuda device=0 or device=0,1,2,3 or device=cpu.")
    parser.add_argument("--optimizer",       default="SGD", help="Optimizer to use, choices=['SGD', 'Adam', 'AdamW', 'RMSProp'].")
    parser.add_argument("--seed",            default=0, help="Random seed for reproducibility.")
    parser.add_argument("--nbs",             type=int, default=64, help="Nominal batch size.")
    parser.add_argument("--close-mosaic",    type=int, default=10, help="Disable mosaic augmentation for final 10 epochs.")
    parser.add_argument("--lr0",             type=float, default=0.01, help="Initial learning rate (i.e. SGD=1E-2, Adam=1E-3).")
    parser.add_argument("--lrf",             type=float, default=0.01, help="Final learning rate (lr0 * lrf).")
    parser.add_argument("--momentum",        type=float, default=0.937, help="SGD momentum/Adam beta1.")
    parser.add_argument("--weight-decay",    type=float, default=0.0005, help="Optimizer weight decay 5e-4.")
    parser.add_argument("--warmup-epochs",   type=float, default=3.0, help="Warmup epochs (fractions ok).")
    parser.add_argument("--warmup-momentum", type=float, default=0.8, help="Warmup initial momentum.")
    parser.add_argument("--warmup-bias-lr",  type=float, default=0.1, help="Warmup initial bias lr.")
    parser.add_argument("--box",             type=float, default=7.5, help="Box loss gain.")
    parser.add_argument("--cls",             type=float, default=0.5, help="Cls loss gain (scale with pixels).")
    parser.add_argument("--dfl",             type=float, default=1.5, help="dfl loss gain.")
    parser.add_argument("--fl-gamma",        type=float, default=0.0, help="Focal loss gamma (efficientDet default gamma=1.5).")
    parser.add_argument("--label-smoothing", type=float, default=0.0, help="Label smoothing (fraction).")
    parser.add_argument("--deterministic",   action="store_true", help="Whether to enable deterministic mode.")
    parser.add_argument("--single-cls",      action="store_true", help="Train multi-class data as single-class.")
    parser.add_argument("--image-weights",   action="store_true", help="Use weighted image selection for training.")
    parser.add_argument("--rect",            action="store_true", help="Support rectangular training.")
    parser.add_argument("--cos-lr",          action="store_true", help="Use cosine learning rate scheduler.")
    parser.add_argument("--save",            action="store_true", help="Save train checkpoints and predict results.")
    parser.add_argument("--cache",           action="store_true", help="True/ram, disk or False. Use cache for data loading.")
    parser.add_argument("--exist-ok",        action="store_true", help="Whether to overwrite existing experiment.")
    parser.add_argument("--pretrained",      action="store_true", help="Whether to use a pretrained model.")
    parser.add_argument("--verbose",         action="store_true", help="Whether to print verbose output.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    train()

# endregion

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the prediction scripts for YOLOv8."""

from __future__ import annotations

import argparse

import mon
from ultralytics import YOLO

_current_dir = mon.Path(__file__).absolute().parent


# region Function

def predict(args: dict):
    # Load a model
    model = args.pop("model")
    model = YOLO(model)  # load a pretrained model (recommended for training)
    
    # Predict with the model
    _ = model(**args)

# endregion


# region Main

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",           default="detect", help="Inference task, i.e. detect, segment, or classify.")
    parser.add_argument("--model",          default="weight/yolov8n.pt", help="Path to model file, i.e. yolov8n.pt, yolov8n.yaml.")
    parser.add_argument("--data",           default="data/visdrone-a2i2-of.yaml", help="Path to data file, i.e. i.e. coco128.yaml.")
    
    parser.add_argument("--project",        default="run/train", help="Project name.")
    parser.add_argument("--name",           default="exp", help="Experiment name.")
    parser.add_argument("--source",         default="data/", help="Source directory for images or videos.")
    parser.add_argument("--imgsz",          type=int,   default=1280, nargs='+', help="Size of input images as integer or w,h.")
    parser.add_argument("--conf",           type=float, default=0.25, help="Object confidence threshold for detection.")
    parser.add_argument("--iou",            type=float, default=0.7, help="Intersection over union (IoU) threshold for NMS.")
    parser.add_argument("--max-det",        type=int,   default=300, help="Maximum number of detections per image.")
    parser.add_argument("--augment",        action="store_true", help="Apply image augmentation to prediction sources.")
    parser.add_argument("--agnostic-nms",   action="store_true", help="class-agnostic NMS.")
    parser.add_argument("--device",         default="cpu", help="Device to run on, i.e. cuda device=0/1/2/3 or device=cpu.")
    parser.add_argument("--exist-ok",       action="store_true", help="Whether to overwrite existing experiment.")
    parser.add_argument("--show",           action="store_true", help="Show results if possible.")
    parser.add_argument("--visualize",      action="store_true", help="Visualize model features.")
    parser.add_argument("--save-txt",       action="store_true", help="Save results as .txt file.")
    parser.add_argument("--save-conf",      action="store_true", help="Save results with confidence scores.")
    parser.add_argument("--save-crop",      action="store_true", help="Save cropped images with results.")
    parser.add_argument("--hide-labels",    action="store_true", help="Hide labels.")
    parser.add_argument("--hide-conf",      action="store_true", help="Hide confidence scores.")
    parser.add_argument("--vid-stride",     action="store_true", help="Video frame-rate stride.")
    parser.add_argument("--line-thickness", type=int, default=4, help="Bounding box thickness (pixels).")
    parser.add_argument("--retina-masks",   action="store_true", help="Use high-resolution segmentation masks.")
    parser.add_argument("--classes",        nargs='+', help="Filter results by class, i.e. class=0, or class=[0,2,3].")
    parser.add_argument("--box",            action="store_true", help="Show boxes in segmentation predictions.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args         = vars(parse_args())
    args["mode"] = "predict"
    predict(args=args)

# endregion

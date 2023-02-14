#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the prediction scripts for YOLOv8."""

from __future__ import annotations

import click

import mon
from ultralytics import YOLO

_current_dir = mon.Path(__file__).absolute().parent


# region Function

@click.command()
@click.option("--task",           default="detect", help="Inference task, i.e. detect, segment, or classify.")
@click.option("--model",          default="weight/yolov8n.pt", type=click.Path(exists=True), help="Path to model file, i.e. yolov8n.pt, yolov8n.yaml.")
@click.option("--data",           default="data/visdrone-a2i2-of.yaml", type=click.Path(exists=True), help="Path to data file, i.e. i.e. coco128.yaml.")
@click.option("--project",        default="run/train", type=click.Path(exists=False), help="Project name.")
@click.option("--name",           default="exp", help="Experiment name.")
@click.option("--source",         default=mon.DATA_DIR, type=click.Path(exists=True), help="Source directory for images or videos.")
@click.option("--imgsz",          default=1280, type=int,  help="Size of input images as integer or w,h.")
@click.option("--conf",           default=0.25, type=float, help="Object confidence threshold for detection.")
@click.option("--iou",            default=0.7,  type=float, help="Intersection over union (IoU) threshold for NMS.")
@click.option("--max-det",        default=300,  type=int,   help="Maximum number of detections per image.")
@click.option("--augment",        is_flag=True, help="Apply image augmentation to prediction sources.")
@click.option("--agnostic-nms",   is_flag=True, help="class-agnostic NMS.")
@click.option("--device",         default="cpu", help="Device to run on, i.e. cuda device=0/1/2/3 or device=cpu.")
@click.option("--exist-ok",       is_flag=True, help="Whether to overwrite existing experiment.")
@click.option("--show",           is_flag=True, help="Show results if possible.")
@click.option("--visualize",      is_flag=True, help="Visualize model features.")
@click.option("--save",           is_flag=True, help="Save train checkpoints and predict results.")
@click.option("--save-txt",       is_flag=True, help="Save results as .txt file.")
@click.option("--save-conf",      is_flag=True, help="Save results with confidence scores.")
@click.option("--save-crop",      is_flag=True, help="Save cropped images with results.")
@click.option("--hide-labels",    is_flag=True, help="Hide labels.")
@click.option("--hide-conf",      is_flag=True, help="Hide confidence scores.")
@click.option("--vid-stride",     is_flag=True, help="Video frame-rate stride.")
@click.option("--overlap-mask",   is_flag=True)
@click.option("--line-thickness", default=4, type=int, help="Bounding box thickness (pixels).")
@click.option("--retina-masks",   is_flag=True, help="Use high-resolution segmentation masks.")
@click.option("--classes",        type=int, help="Filter results by class, i.e. class=0, or class=[0,2,3].")
@click.option("--box",            is_flag=True, help="Show boxes in segmentation predictions.")
def predict(
    task, model, data, project, name, source, imgsz, conf, iou, max_det,
    augment, agnostic_nms, device, exist_ok, show, visualize, save, save_txt,
    save_conf, save_crop, hide_labels, hide_conf, vid_stride, overlap_mask,
    line_thickness, retina_masks, classes, box,
):
    # Load a model
    model = YOLO(model)  # load a pretrained model (recommended for training)
    
    # Predict with the model
    args = {
        "task"          : task,
        "mode"          : "predict",
        "data"          : data,
        "project"       : project,
        "name"          : name,
        "source"        : source,
        "imgsz"         : imgsz,
        "conf"          : conf,
        "iou"           : iou,
        "max_det"       : max_det,
        "augment"       : augment,
        "agnostic_nms"  : agnostic_nms,
        "device"        : device,
        "exist_ok"      : exist_ok,
        "show"          : show,
        "visualize"     : visualize,
        "save"          : save,
        "save_txt"      : save_txt,
        "save_conf"     : save_conf,
        "save_crop"     : save_crop,
        "hide_labels"   : hide_labels,
        "hide_conf"     : hide_conf,
        "vid_stride"    : vid_stride,
        "overlap_mask"  : overlap_mask,
        "line_thickness": line_thickness,
        "retina_masks"  : retina_masks,
        "classes"       : classes,
        "box"           : box,
    }
    _    = model(**args)

# endregion


# region Main

if __name__ == "__main__":
    predict()

# endregion

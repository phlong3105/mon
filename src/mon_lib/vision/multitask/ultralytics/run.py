#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements the training scripts for YOLOv8."""

from __future__ import annotations

import argparse

from mon import DATA_DIR
from ultralytics import YOLO


def run(args: argparse.Namespace):
    model = YOLO(args.model)
    _     = model.train(**vars(args))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",            type=str,            default="detect",       help="YOLO task, i.e., detect, segment, classify, pose.")
    parser.add_argument("--mode",            type=str,            default="train",        help="YOLO mode, i.e., train, val, predict, export, track, benchmark.")
    # Train settings
    parser.add_argument("--model",           type=str,            default="yolov8n.yaml", help="Path to model file, i.e., yolov8n.pt, yolov8n.yaml.")
    parser.add_argument("--data",            type=str,            default="coco128.yaml", help="Path to data file, i.e. i.e. coco128.yaml.")
    parser.add_argument("--epochs",          type=int,            default=100,            help="Number of epochs to train for.")
    parser.add_argument("--patience",        type=int,            default=50,             help="Epochs to wait for no observable improvement for early stopping of training.")
    parser.add_argument("--batch",           type=int,            default=-1,             help="Number of images per batch (-1 for AutoBatch).")
    parser.add_argument("--imgsz",           type=int,            default=1280,           help="Input images size as int for train and val modes, or list[w,h] for predict and export modes.")
    parser.add_argument("--save",            action="store_true",                         help="Save train checkpoints and predict results.")
    parser.add_argument("--save_period",     type=int,            default=-1,             help="Save checkpoint every x epochs (disabled if < 1).")
    parser.add_argument("--cache",           action="store_true",                         help="True/ram, disk or False. Use cache for data loading.")
    parser.add_argument("--device",          type=str,            default="cpu",          help="Device to run on, i.e. cuda device=0 or device=0,1,2,3 or device=cpu.")
    parser.add_argument("--workers",         type=int,            default=8,              help="Number of worker threads for data loading (per RANK if DDP).")
    parser.add_argument("--project",         type=str,            default="run",          help="Project name.")
    parser.add_argument("--name",            type=str,            default=None,           help="Experiment name, results saved to 'project/name' directory.")
    parser.add_argument("--exist-ok",        action="store_true",                         help="Whether to overwrite existing experiment.")
    parser.add_argument("--pretrained",      action="store_true",                         help="Whether to use a pretrained model.")
    parser.add_argument("--optimizer",       type=str,            default="auto",         help="Optimizer to use, choices=[SGD, Adam, Adamax, AdamW, NAdam, RAdam, RMSProp, auto].")
    parser.add_argument("--seed",            type=int,            default=0,              help="Random seed for reproducibility.")
    parser.add_argument("--deterministic",   action="store_true",                         help="Whether to enable deterministic mode.")
    parser.add_argument("--single-cls",      action="store_true",                         help="Train multi-class data as single-class.")
    parser.add_argument("--rect",            action="store_true",                         help="Support rectangular training.")
    parser.add_argument("--cos-lr",          action="store_true",                         help="Use cosine learning rate scheduler.")
    parser.add_argument("--close-mosaic",    type=int,            default=10,             help="Disable mosaic augmentation for final 10 epochs.")
    parser.add_argument("--amp",             action="store_true",                         help="Automatic Mixed Precision (AMP) training, choices=[True, False], True runs AMP check.")
    parser.add_argument("--fraction",        type=float,          default=1.0,            help="Dataset fraction to train on (default is 1.0, all images in train set).")
    parser.add_argument("--profile",         action="store_true",                         help="Profile ONNX and TensorRT speeds during training for loggers.")
    parser.add_argument("--freeze",                               default=None,           help="Freeze first n layers, or freeze list of layer indices during training.")
    parser.add_argument("--multi_scale",     action="store_true",                         help="Whether to use multi-scale during training.")
    # Segmentation
    parser.add_argument("--overlap_mask",    action="store_true",                         help="Masks should overlap during training (segment train only).")
    parser.add_argument("--mask_ratio",      type=int,            default=4,              help="Mask downsample ratio (segment train only).")
    # Classification
    parser.add_argument("--dropout",         type=float,          default=0.0,            help="Use dropout regularization (classify train only).")
    # Val/Test settings
    parser.add_argument("--val",             action="store_true",                         help="Validate/test during training.")
    parser.add_argument("--split",           type=str,            default="val",          help="Dataset split to use for validation, i.e. 'val', 'test' or 'train'.")
    parser.add_argument("--save_json",       action="store_true",                         help="Save results to JSON file.")
    parser.add_argument("--save_hybrid",     action="store_true",                         help="Save hybrid version of labels (labels + additional predictions).")
    parser.add_argument("--conf",            type=float,          default=0.25,           help="Object confidence threshold for detection (default 0.25 predict, 0.001 val).")
    parser.add_argument("--iou",             type=float,          default=0.70,           help="Intersection over union (IoU) threshold for NMS.")
    parser.add_argument("--max_det",         type=int,            default=300,            help="Maximum number of detections per image.")
    parser.add_argument("--half",            action="store_true",                         help="Use half precision (FP16).")
    parser.add_argument("--dnn",             action="store_true",                         help="Use OpenCV DNN for ONNX inference.")
    parser.add_argument("--plots",           action="store_true",                         help="Save plots and images during train/val.")
    # Predict settings
    parser.add_argument("--source",          type=str,            default=DATA_DIR,       help="Source directory for images or videos.")
    parser.add_argument("--vid_stride",      type=int,            default=1,              help="Video frame-rate stride.")
    parser.add_argument("--stream_buffer",   action="store_true",                         help="Buffer all streaming frames (True) or return the most recent frame (False).")
    parser.add_argument("--visualize",       action="store_true",                         help="Visualize model features.")
    parser.add_argument("--augment",         action="store_true",                         help="Apply image augmentation to prediction sources.")
    parser.add_argument("--agnostic_nms",    action="store_true",                         help="Class-agnostic NMS.")
    parser.add_argument("--classes",                                                      help="Filter results by class, i.e. classes=0, or classes=[0,2,3].")
    parser.add_argument("--retina_masks",    action="store_true",                         help="Use high-resolution segmentation masks.")
    parser.add_argument("--embed",                                                        help="Return feature vectors/embeddings from given layers.")
    # Visualize settings
    parser.add_argument("--show",            action="store_true",                         help="Show predicted images and videos if environment allows.")
    parser.add_argument("--save_frames",     action="store_true",                         help="Save predicted individual video frames.")
    parser.add_argument("--save_txt",        action="store_true",                         help="Save results as .txt file.")
    parser.add_argument("--save_conf",       action="store_true",                         help="Save results with confidence scores.")
    parser.add_argument("--save_crop",       action="store_true",                         help="Save cropped images with results.")
    parser.add_argument("--show_labels",     action="store_true",                         help="Show prediction labels, i.e. 'person'.")
    parser.add_argument("--show_conf",       action="store_true",                         help="Show prediction confidence, i.e. '0.99'.")
    parser.add_argument("--show_boxes",      action="store_true",                         help="Show prediction boxes.")
    parser.add_argument("--line_width",      type=int,                                    help="Line width of the bounding boxes. Scaled to image size if None.")
    # Export settings
    parser.add_argument("--format",          type=str,            default="torchscript",  help="Format to export to, choices at https://docs.ultralytics.com/modes/export/#export-formats.")
    parser.add_argument("--keras",           action="store_true",                         help="Use Keras.")
    parser.add_argument("--optimize",        action="store_true",                         help="TorchScript: optimize for mobile.")
    parser.add_argument("--int8",            action="store_true",                         help="CoreML/TF INT8 quantization.")
    parser.add_argument("--dynamic",         action="store_true",                         help="ONNX/TF/TensorRT: dynamic axes.")
    parser.add_argument("--simplify",        action="store_true",                         help="ONNX: simplify model.")
    parser.add_argument("--opset",           type=int,                                    help="ONNX: opset version.")
    parser.add_argument("--workspace",       type=int,            default=4,              help="TensorRT: workspace size (GB).")
    parser.add_argument("--nms",             action="store_true",                         help="CoreML: add NMS.")
    # Hyperparameters
    parser.add_argument("--lr0",             type=float,          default=0.01,           help="Initial learning rate (i.e. SGD=1E-2, Adam=1E-3).")
    parser.add_argument("--lrf",             type=float,          default=0.01,           help="Final learning rate (lr0 * lrf).")
    parser.add_argument("--momentum",        type=float,          default=0.937,          help="SGD momentum/Adam beta1.")
    parser.add_argument("--weight_decay",    type=float,          default=0.0005,         help="Optimizer weight decay 5e-4.")
    parser.add_argument("--warmup_epochs",   type=float,          default=3.0,            help="Warmup epochs (fractions ok).")
    parser.add_argument("--warmup_momentum", type=float,          default=0.8,            help="Warmup initial momentum.")
    parser.add_argument("--warmup_bias_lr",  type=float,          default=0.1,            help="Warmup initial bias lr.")
    parser.add_argument("--box",             type=float,          default=7.5,            help="Box loss gain.")
    parser.add_argument("--cls",             type=float,          default=0.5,            help="CLS loss gain (scale with pixels).")
    parser.add_argument("--dfl",             type=float,          default=1.5,            help="DFL loss gain.")
    parser.add_argument("--pose",            type=float,          default=12.0,           help="Pose loss gain.")
    parser.add_argument("--kobj",            type=float,          default=1.0,            help="Keypoint obj loss gain.")
    parser.add_argument("--label_smoothing", type=float,          default=0.0,            help="Label smoothing (fraction).")
    parser.add_argument("--nbs",             type=float,          default=64,             help="Nominal batch size.")
    parser.add_argument("--hsv_h",           type=float,          default=0.015,          help="Image HSV-Hue augmentation (fraction).")
    parser.add_argument("--hsv_s",           type=float,          default=0.7,            help="Image HSV-Saturation augmentation (fraction).")
    parser.add_argument("--hsv_v",           type=float,          default=0.4,            help="Image HSV-Value augmentation (fraction).")
    parser.add_argument("--degrees",         type=float,          default=0.0,            help="Image rotation (+/- deg).")
    parser.add_argument("--translate",       type=float,          default=0.1,            help="Image translation (+/- fraction).")
    parser.add_argument("--scale",           type=float,          default=0.5,            help="Image scale (+/- gain).")
    parser.add_argument("--shear",           type=float,          default=0.0,            help="Image shear (+/- deg).")
    parser.add_argument("--perspective",     type=float,          default=0.0,            help="Image perspective (+/- fraction), range 0-0.001.")
    parser.add_argument("--flipud",          type=float,          default=0.0,            help="Image flip up-down (probability).")
    parser.add_argument("--fliplr",          type=float,          default=0.5,            help="Image flip left-right (probability).")
    parser.add_argument("--mosaic",          type=float,          default=1.0,            help="Image mosaic (probability).")
    parser.add_argument("--mixup",           type=float,          default=0.0,            help="Image mixup (probability).")
    parser.add_argument("--copy_paste",      type=float,          default=0.0,            help="Segment copy-paste (probability).")
    parser.add_argument("--auto_augment",    type=str,            default="randaugment",  help="Auto augmentation policy for classification (randaugment, autoaugment, augmix).")
    parser.add_argument("--erasing",         type=float,          default=0.4,            help="Probability of random erasing during classification training (0-1).")
    parser.add_argument("--crop_fraction",   type=float,          default=1.0,            help="Image crop fraction for classification evaluation/inference (0-1).")
    parser.add_argument("--tracker",         type=str,            default="botsort.yaml", help="Tracker type, choices=[botsort.yaml, bytetrack.yaml].")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    run(args)

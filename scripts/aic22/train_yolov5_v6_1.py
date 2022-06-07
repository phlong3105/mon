#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import argparse
import os
from argparse import Namespace

import torch

import one.vision.object_detection.yolov5_v6_1.detect as detect
import one.vision.object_detection.yolov5_v6_1.train as train
import one.vision.object_detection.yolov5_v6_1.val as test
from aic import pretrained_dir

yolov5_pretrained_dir = os.path.join(pretrained_dir, "yolov5_v6_1")
yolov5_root           = os.path.dirname(os.path.abspath(train.__file__))


# MARK: - Train

train_configs = {
    "yolov5n6_v6_1_aic22retail117_480" : {
        "weights"        : os.path.join(yolov5_pretrained_dir, "yolov5n6_v6_1_coco.pt"),
        "cfg"            : os.path.join(yolov5_root, "models", "yolov5n6.yaml"),
        "data"           : os.path.join(yolov5_root, "data",   "aic22retail117.yaml"),
        "hyp"            : os.path.join(yolov5_root, "data",   "hyps/hyp.scratch-low.yaml"),
        "epochs"         : 50,
        "batch_size"     : 128,
        "imgsz"          : 480,
        "rect"           : False,
        "resume"         : False,
        "evolve"         : False,
        "bucket"         : "",
        "cache"          : False,
        "image_weights"  : False,
        "device"         : "0",
        "multi_scale"    : False,
        "single_cls"     : False,
        "optimizer"      : "SGD",
        "sync_bn"        : False,
        "workers"        : 8,
        "project"        : f"{pretrained_dir}/yolov5_v6_1",
        "name"           : "yolov5n6_v6_1_aic22retail117_480",
        "exist_ok"       : False,
        "quad"           : False,
        "cos_lr"         : False,
        "label_smoothing": 0.0,
        "patience"       : 100,
        "freeze"         : [0],
        "save_period"    : -1,
        "local_rank"     : -1,
        "nosave"         : False,
        "noval"          : False,
        "noautoanchor"   : False,
        "entity"         : None,
        "upload_dataset" : False,
        "bbox_interval"  : -1,
        "artifact_alias" : "latest",
    },
    "yolov5s6_v6_1_aic22retail117_640" : {
        "weights"        : os.path.join(yolov5_pretrained_dir, "yolov5s6_v6_1_coco.pt"),
        "cfg"            : os.path.join(yolov5_root, "models", "yolov5s6.yaml"),
        "data"           : os.path.join(yolov5_root, "data",   "aic22retail117.yaml"),
        "hyp"            : os.path.join(yolov5_root, "data",   "hyps/hyp.scratch-low.yaml"),
        "epochs"         : 50,
        "batch_size"     : 64,
        "imgsz"          : 640,
        "rect"           : False,
        "resume"         : False,
        "evolve"         : False,
        "bucket"         : "",
        "cache"          : False,
        "image_weights"  : False,
        "device"         : "0",
        "multi_scale"    : False,
        "single_cls"     : False,
        "optimizer"      : "SGD",
        "sync_bn"        : False,
        "workers"        : 8,
        "project"        : f"{pretrained_dir}/yolov5_v6_1",
        "name"           : "yolov5s6_v6_1_aic22retail117_640",
        "exist_ok"       : False,
        "quad"           : False,
        "cos_lr"         : False,
        "label_smoothing": 0.0,
        "patience"       : 100,
        "freeze"         : [0],
        "save_period"    : -1,
        "local_rank"     : -1,
        "nosave"         : False,
        "noval"          : False,
        "noautoanchor"   : False,
        "entity"         : None,
        "upload_dataset" : False,
        "bbox_interval"  : -1,
        "artifact_alias" : "latest",
    },
    "yolov5x6_v6_1_aic22retail117_1536": {
        "weights"        : os.path.join(yolov5_pretrained_dir, "yolov5x6_v6_1_coco.pt"),
        "cfg"            : os.path.join(yolov5_root, "models", "yolov5x6.yaml"),
        "data"           : os.path.join(yolov5_root, "data",   "aic22retail117.yaml"),
        "hyp"            : os.path.join(yolov5_root, "data",   "hyps/hyp.scratch-low.yaml"),
        "epochs"         : 20,
        "batch_size"     : 4,
        "imgsz"          : 1536,
        "rect"           : False,
        "resume"         : False,
        "evolve"         : False,
        "bucket"         : "",
        "cache"          : False,
        "image_weights"  : False,
        "device"         : "0",
        "multi_scale"    : False,
        "single_cls"     : False,
        "optimizer"      : "SGD",
        "sync_bn"        : False,
        "workers"        : 8,
        "project"        : f"{pretrained_dir}/yolov5_v6_1",
        "name"           : "yolov5x6_v6_1_aic22retail117_1536",
        "exist_ok"       : False,
        "quad"           : False,
        "cos_lr"         : False,
        "label_smoothing": 0.0,
        "patience"       : 100,
        "freeze"         : [0],
        "save_period"    : -1,
        "local_rank"     : -1,
        "nosave"         : False,
        "noval"          : False,
        "noautoanchor"   : False,
        "entity"         : None,
        "upload_dataset" : False,
        "bbox_interval"  : -1,
        "artifact_alias" : "latest",
    },
}


def run_train(cfg: str = "yolov5x6_v6_1_aic22retail117_1536"):
    if cfg in train_configs:
        opt = train_configs[cfg]
        opt = Namespace(**opt)
        print(opt)
        train.main(opt)


def run_save_weights(path):
    ckpt    = torch.load(path, map_location="cpu")
    weights = ckpt["model"].float().state_dict()
    torch.save(weights, path.replace(".pt", "_weights.pt"))
    

# MARK: - Test

test_configs = {

}


def run_test(cfg: str = "yolov5x6_v6_1_aic22retail117_1536"):
    if cfg in train_configs:
        opt = test_configs[cfg]
        test.main(opt)
        

# MARK: - Detect

detect_configs = {

}


def run_detect(cfg: str = "yolov5x6_v6_1_aic22retail117_1536"):
    if cfg in train_configs:
        opt = train_configs[cfg]
        detect.main(opt)


# MARK: - Main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="save_weights",                     type=str)
    parser.add_argument("--cfg", default="yolov5s6_v6_1_aic22retail117_640", type=str)
    opt = parser.parse_args()
    
    if opt.run == "train":
        run_train(cfg=opt.cfg)
    elif opt.run == "test":
        run_test(cfg=opt.cfg)
    elif opt.run == "detect":
        run_detect(cfg=opt.cfg)
    elif opt.run == "save_weights":
        run_save_weights(path=os.path.join(pretrained_dir, "yolov5_v6_1", "best.pt"))

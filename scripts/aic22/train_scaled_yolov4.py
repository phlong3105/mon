#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import argparse
import os
from argparse import Namespace

import torch

import one.vision.detection.scaled_yolov4.detect as detect
import one.vision.detection.scaled_yolov4.test as test
import one.vision.detection.scaled_yolov4.train as train
from aic import data_dir
from aic import pretrained_dir

yolov4_pretrained_dir = os.path.join(pretrained_dir, "scaled_yolov4")
yolov4_root           = os.path.dirname(os.path.abspath(train.__file__))


# MARK: - Train

train_configs = {
    "yolov4-csp_aic22retail117_448": {
        "weights"        : os.path.join(yolov4_pretrained_dir, "yolov4-csp_coco.weights"),
        "cfg"            : os.path.join(yolov4_root, "models", "yolov4-csp.yaml"),
        "data"           : os.path.join(yolov4_root, "data",   "aic22retail117.yaml"),
        "hyp"            : os.path.join(yolov4_root, "data",   "hyp.scratch.yaml"),
        "epochs"         : 50,
        "batch_size"     : 16,
        "img_size"       : [448, 448],
        "rect"           : False,
        "resume"         : False,
        "nosave"         : False,
        "notest"         : False,
        "noautoanchor"   : False,
        "evolve"         : False,
        "bucket"         : "",
        "cache_images"   : False,
        "name"           : "yolov4-csp_aic22retail117_448",
        "device"         : "0",
        "multi_scale"    : False,
        "single_cls"     : False,
        "adam"           : False,
        "sync_bn"        : True,
        "local_rank"     : -1,
        "logdir"         : f"{pretrained_dir}/scaled_yolov4",
    },
    "yolov4-csp_aic22retail117_896": {
        "weights"        : os.path.join(yolov4_pretrained_dir, "yolov4-csp_coco.weights"),
        "cfg"            : os.path.join(yolov4_root, "models", "yolov4-csp.yaml"),
        "data"           : os.path.join(yolov4_root, "data",   "aic22retail117.yaml"),
        "hyp"            : os.path.join(yolov4_root, "data",   "hyp.scratch.yaml"),
        "epochs"         : 50,
        "batch_size"     : 16,
        "img_size"       : [896, 896],
        "rect"           : False,
        "resume"         : False,
        "nosave"         : False,
        "notest"         : False,
        "noautoanchor"   : False,
        "evolve"         : False,
        "bucket"         : "",
        "cache_images"   : False,
        "name"           : "yolov4-csp_aic22retail117_896",
        "device"         : "0",
        "multi_scale"    : False,
        "single_cls"     : False,
        "adam"           : False,
        "sync_bn"        : True,
        "local_rank"     : -1,
        "logdir"         : f"{pretrained_dir}/scaled_yolov4",
    },
    "yolov4-p5_aic22retail116_448" : {
        "weights"        : os.path.join(yolov4_pretrained_dir, "yolov4-p5_coco.pt"),
        "cfg"            : os.path.join(yolov4_root, "models", "yolov4-p5.yaml"),
        "data"           : os.path.join(yolov4_root, "data",   "aic22retail116.yaml"),
        "hyp"            : os.path.join(yolov4_root, "data",   "hyp.scratch.yaml"),
        "epochs"         : 50,
        "batch_size"     : 8,
        "img_size"       : [448, 448],
        "rect"           : False,
        "resume"         : False,
        "nosave"         : False,
        "notest"         : False,
        "noautoanchor"   : False,
        "evolve"         : False,
        "bucket"         : "",
        "cache_images"   : False,
        "name"           : "yolov4-p5_aic22retail116_448",
        "device"         : "0",
        "multi_scale"    : False,
        "single_cls"     : False,
        "adam"           : False,
        "sync_bn"        : True,
        "local_rank"     : -1,
        "logdir"         : f"{pretrained_dir}/scaled_yolov4",
    },
    "yolov4-p5_aic22retail117_448" : {
        "weights"        : os.path.join(yolov4_pretrained_dir, "yolov4-p5_coco.pt"),
        "cfg"            : os.path.join(yolov4_root, "models", "yolov4-p5.yaml"),
        "data"           : os.path.join(yolov4_root, "data",   "aic22retail117.yaml"),
        "hyp"            : os.path.join(yolov4_root, "data",   "hyp.scratch.yaml"),
        "epochs"         : 50,
        "batch_size"     : 8,
        "img_size"       : [448, 448],
        "rect"           : False,
        "resume"         : False,
        "nosave"         : False,
        "notest"         : False,
        "noautoanchor"   : False,
        "evolve"         : False,
        "bucket"         : "",
        "cache_images"   : False,
        "name"           : "yolov4-p5_aic22retail117_448",
        "device"         : "0",
        "multi_scale"    : False,
        "single_cls"     : False,
        "adam"           : False,
        "sync_bn"        : True,
        "local_rank"     : -1,
        "logdir"         : f"{pretrained_dir}/scaled_yolov4",
    },
    "yolov4-p5_aic22retail117_896" : {
        "weights"        : os.path.join(yolov4_pretrained_dir, "yolov4-p5_coco.pt"),
        "cfg"            : os.path.join(yolov4_root, "models", "yolov4-p5.yaml"),
        "data"           : os.path.join(yolov4_root, "data",   "aic22retail117.yaml"),
        "hyp"            : os.path.join(yolov4_root, "data",   "hyp.scratch.yaml"),
        "epochs"         : 50,
        "batch_size"     : 4,
        "img_size"       : [896, 896],
        "rect"           : False,
        "resume"         : False,
        "nosave"         : False,
        "notest"         : False,
        "noautoanchor"   : False,
        "evolve"         : False,
        "bucket"         : "",
        "cache_images"   : False,
        "name"           : "yolov4-p5_aic22retail117_896",
        "device"         : "0",
        "multi_scale"    : False,
        "single_cls"     : False,
        "adam"           : False,
        "sync_bn"        : True,
        "local_rank"     : -1,
        "logdir"         : f"{pretrained_dir}/scaled_yolov4",
    },
    "yolov4-p7_aic22retail116_1536": {
        "weights"        : os.path.join(yolov4_pretrained_dir, "yolov4-p7_coco.pt"),
        "cfg"            : os.path.join(yolov4_root, "models", "yolov4-p7.yaml"),
        "data"           : os.path.join(yolov4_root, "data",   "aic22retail116.yaml"),
        "hyp"            : os.path.join(yolov4_root, "data",   "hyp.scratch.yaml"),
        "epochs"         : 20,
        "batch_size"     : 4,
        "img_size"       : [1536, 1536],
        "rect"           : False,
        "resume"         : False,
        "nosave"         : False,
        "notest"         : False,
        "noautoanchor"   : False,
        "evolve"         : False,
        "bucket"         : "",
        "cache_images"   : False,
        "name"           : "yolov4-p7_aic22retail116_1536",
        "device"         : "0",
        "multi_scale"    : False,
        "single_cls"     : False,
        "adam"           : False,
        "sync_bn"        : True,
        "local_rank"     : -1,
        "logdir"         : f"{pretrained_dir}/scaled_yolov4",
    },
    "yolov4-p7_aic22retail117_1536": {
        "weights"        : os.path.join(yolov4_pretrained_dir, "yolov4-p7_coco.pt"),
        "cfg"            : os.path.join(yolov4_root, "models", "yolov4-p7.yaml"),
        "data"           : os.path.join(yolov4_root, "data",   "aic22retail117.yaml"),
        "hyp"            : os.path.join(yolov4_root, "data",   "hyp.scratch.yaml"),
        "epochs"         : 20,
        "batch_size"     : 4,
        "img_size"       : [1536, 1536],
        "rect"           : False,
        "resume"         : False,
        "nosave"         : False,
        "notest"         : False,
        "noautoanchor"   : False,
        "evolve"         : False,
        "bucket"         : "",
        "cache_images"   : False,
        "name"           : "yolov4-p7_aic22retail117_1536",
        "device"         : "0",
        "multi_scale"    : False,
        "single_cls"     : False,
        "adam"           : False,
        "sync_bn"        : True,
        "local_rank"     : -1,
        "logdir"         : f"{pretrained_dir}/scaled_yolov4",
    },
}


def run_train(cfg: str = "yolov4-p7_aic22retail117_1536"):
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


def run_test(cfg: str = "yolov4-p7_aic22retail117_1536"):
    if cfg in train_configs:
        opt = test_configs[cfg]
        test.main(opt)
        

# MARK: - Detect

detect_configs = {
    "yolov4-p7_aic22retail117_1536": {
        "weights"     : os.path.join(pretrained_dir, "scaled_yolov4", "yolov4-p7_aic22retail117_1536.pt"),
        "source"      : os.path.join(data_dir, "test_a", "testA_2.mp4"),
        "output"      : "inference/output",
        "img_size"    : 1536,
        "conf_thres"  : 0.3,
        "iou_thres"   : 0.5,
        "device"      : "0",
        "view_img"    : True,
        "save_txt"    : False,
        "classes"     : None,
        "agnostic_nms": False,
        "augment"     : False,
        "update"      : False,
        "verbose"     : True,
    },
}


def run_detect(cfg: str = "yolov4-p7_aic22retail117_1536"):
    if cfg in detect_configs:
        opt = detect_configs[cfg]
        opt = Namespace(**opt)
        print(opt)
        detect.main(opt)


# MARK: - Main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="train",                        type=str)
    parser.add_argument("--cfg", default="yolov4-p5_aic22retail116_448", type=str)
    opt = parser.parse_args()
    
    if opt.run == "train":
        run_train(cfg=opt.cfg)
    elif opt.run == "test":
        run_test(cfg=opt.cfg)
    elif opt.run == "detect":
        run_detect(cfg=opt.cfg)
    elif opt.run == "save_weights":
        run_save_weights(path=os.path.join(pretrained_dir, "scaled_yolov4", "best.pt"))

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Custom Scaled-YOLOV4's training, testing, and inference pipelines for the
ChaLearn LTD dataset.
"""

from __future__ import annotations

import csv
import glob
import os
import pickle
import platform
import time
from argparse import Namespace
from pathlib import Path
from typing import Optional
from typing import Union

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from tqdm import tqdm

import one.vision.object_detection.scaled_yolov4.test as test
import one.vision.object_detection.scaled_yolov4.train as train
from chalearn import data_dir
from chalearn import pretrained_dir
from one import console
from one import create_dirs
from one import load_file
from one import progress_bar
from one.vision.object_detection.scaled_yolov4 import apply_classifier
from one.vision.object_detection.scaled_yolov4 import attempt_load
from one.vision.object_detection.scaled_yolov4 import check_img_size
from one.vision.object_detection.scaled_yolov4 import load_classifier
from one.vision.object_detection.scaled_yolov4 import LoadImages
from one.vision.object_detection.scaled_yolov4 import LoadStreams
from one.vision.object_detection.scaled_yolov4 import non_max_suppression
from one.vision.object_detection.scaled_yolov4 import plot_one_box
from one.vision.object_detection.scaled_yolov4 import scale_coords
from one.vision.object_detection.scaled_yolov4 import select_device
from one.vision.object_detection.scaled_yolov4 import strip_optimizer
from one.vision.object_detection.scaled_yolov4 import time_synchronized
from one.vision.object_detection.scaled_yolov4 import xyxy2xywh

__all__ = [
    "run_detect",
    "run_train",
    "run_save_weights",
    "run_test",
    "create_pkl_from_txts",
    "merge_pkls",
    "fix_cls_id",
    "Detect",
]


# MARK: - Globals

yolov4_pretrained_dir = os.path.join(pretrained_dir, "scaled_yolov4")
yolov4_root_dir       = os.path.dirname(os.path.abspath(train.__file__))
current_dir           = os.path.dirname(os.path.abspath(__file__))


months_map = {
    "jan": 0,
    "mar": 1,
    "apr": 2,
    "may": 3,
    "jun": 4,
    "jul": 5,
    "aug": 6,
    "sep": 7
}

months_map_invert = {
    0: "jan",
    1: "mar",
    2: "apr",
    3: "may",
    4: "jun",
    5: "jul",
    6: "aug",
    7: "sep"
}

train_configs = {
    "yolov4-p7_chalearnltdmonth_896" : {
        # "weights"     : os.path.join(yolov4_pretrained_dir, "yolov4-p7_coco.pt"),
        "weights"     : os.path.join(pretrained_dir, "scaled_yolov4", "exp0_yolov4-p7_chalearnltdmonth_896_2", "weights", "best.pt"),
        "cfg"         : os.path.join(yolov4_root_dir, "models", "yolov4-p7.yaml"),
        "data"        : os.path.join(current_dir, "chalearnltdmonth.yaml"),
        "hyp"         : os.path.join(yolov4_root_dir, "data", "hyp.scratch.yaml"),
        "epochs"      : 100,
        "batch_size"  : 8,
        "img_size"    : [896, 896],
        "rect"        : True,
        "resume"      : False,
        "nosave"      : False,
        "notest"      : False,
        "noautoanchor": False,
        "evolve"      : False,
        "bucket"      : "",
        "cache_images": False,
        "name"        : "yolov4-p7_chalearnltdmonth_896_3",
        "device"      : "0,1",
        "multi_scale" : False,
        "single_cls"  : False,
        "adam"        : False,
        "sync_bn"     : True,
        "local_rank"  : -1,
        "logdir"      : os.path.join(pretrained_dir, "scaled_yolov4"),
        "verbose"     : False,
    },
    "yolov4-p7_chalearnltdmonth_1280": {
        "weights"     : os.path.join(yolov4_pretrained_dir, "yolov4-p7_coco.pt"),
        # "weights"     : os.path.join(pretrained_dir,  "scaled_yolov4", "exp0_yolov4-p7_chalearnltdmonth_1280", "weights", "best.pt"),
        "cfg"         : os.path.join(yolov4_root_dir, "models", "yolov4-p7.yaml"),
        "data"        : os.path.join(current_dir,     "chalearnltdmonth.yaml"),
        "hyp"         : os.path.join(yolov4_root_dir, "data", "hyp.scratch.yaml"),
        "epochs"      : 100,
        "batch_size"  : 8,
        "img_size"    : [1280, 1280],
        "rect"        : False,
        "resume"      : False,
        "nosave"      : False,
        "notest"      : False,
        "noautoanchor": False,
        "evolve"      : False,
        "bucket"      : "",
        "cache_images": False,
        "name"        : "yolov4-p7_chalearnltdmonth_1280",
        "device"      : "0, 1",
        "multi_scale" : False,
        "single_cls"  : False,
        "adam"        : False,
        "sync_bn"     : True,
        "local_rank"  : -1,
        "logdir"      : os.path.join(pretrained_dir, "scaled_yolov4"),
        "verbose"     : False,
    },
    "yolov4-p7_chalearnltdmonth_1536": {
        # "weights"     : os.path.join(yolov4_pretrained_dir, "yolov4-p7_coco.pt"),
        "weights"     : os.path.join(pretrained_dir,  "scaled_yolov4", "exp0_yolov4-p7_chalearnltdmonth_1536_2", "weights", "best.pt"),
        "cfg"         : os.path.join(yolov4_root_dir, "models", "yolov4-p7.yaml"),
        "data"        : os.path.join(current_dir,     "chalearnltdmonth.yaml"),
        "hyp"         : os.path.join(yolov4_root_dir, "data", "hyp.scratch.yaml"),
        "epochs"      : 100,
        "batch_size"  : 8,
        "img_size"    : [1536, 1536],
        "rect"        : False,
        "resume"      : False,
        "nosave"      : False,
        "notest"      : False,
        "noautoanchor": False,
        "evolve"      : False,
        "bucket"      : "",
        "cache_images": False,
        "name"        : "yolov4-p7_chalearnltdmonth_1536_3",
        "device"      : "0, 1",
        "multi_scale" : False,
        "single_cls"  : False,
        "adam"        : False,
        "sync_bn"     : True,
        "local_rank"  : -1,
        "logdir"      : os.path.join(pretrained_dir, "scaled_yolov4"),
        "verbose"     : False,
    },
    "yolov4-p7_chalearnltdmonth_1920": {
        # "weights"     : os.path.join(yolov4_pretrained_dir, "yolov4-p7_coco.pt"),
        "weights"     : os.path.join(pretrained_dir, "scaled_yolov4", "exp0_yolov4-p7_chalearnltdmonth_1920", "weights", "best.pt"),
        "cfg"         : os.path.join(yolov4_root_dir, "models", "yolov4-p7.yaml"),
        "data"        : os.path.join(current_dir, "chalearnltdmonth.yaml"),
        "hyp"         : os.path.join(yolov4_root_dir, "data", "hyp.scratch.yaml"),
        "epochs"      : 100,
        "batch_size"  : 4,
        "img_size"    : [1920, 1920],
        "rect"        : False,
        "resume"      : False,
        "nosave"      : False,
        "notest"      : False,
        "noautoanchor": False,
        "evolve"      : False,
        "bucket"      : "",
        "cache_images": False,
        "name"        : "yolov4-p7_chalearnltdmonth_1920_2",
        "device"      : "0, 1",
        "multi_scale" : False,
        "single_cls"  : False,
        "adam"        : False,
        "sync_bn"     : True,
        "local_rank"  : -1,
        "logdir"      : os.path.join(pretrained_dir, "scaled_yolov4"),
        "verbose"     : False,
    },
}

test_configs = {}

detect_configs = {
    "yolov4-p7_chalearnltdmonth_1920": {
        "weights"     : [
            os.path.join(pretrained_dir, "scaled_yolov4", "exp0_yolov4-p7_chalearnltdmonth_896_2",  "weights", "best.pt"),
            os.path.join(pretrained_dir, "scaled_yolov4", "exp0_yolov4-p7_chalearnltdmonth_896",    "weights", "best.pt"),
            os.path.join(pretrained_dir, "scaled_yolov4", "exp0_yolov4-p7_chalearnltdmonth_1536_2", "weights", "best.pt"),
            os.path.join(pretrained_dir, "scaled_yolov4", "exp0_yolov4-p7_chalearnltdmonth_1536",   "weights", "best.pt"),
            os.path.join(pretrained_dir, "scaled_yolov4", "exp0_yolov4-p7_chalearnltdmonth_1920",   "weights", "best.pt")
        ],
        "source"      : os.path.join(data_dir, "chalearn", "ltd", "val"),
        "output"      : os.path.join("inference", "output"),
        "img_size"    : 1920,
        "batch_size"  : 2,
        "conf_thres"  : 0.1,
        "iou_thres"   : 0.5,
        "device"      : "0",
        "view_img"    : False,
        "save_img"    : False,
        "save_txt"    : True,
        "classes"     : None,
        "agnostic_nms": True,
        "augment"     : True,
        "update"      : False,
        "verbose"     : False,
        "recreate"    : False,
    },
    "yolov4-p7_chalearnltdmonth_1536": {
        "weights"     : [
            os.path.join(pretrained_dir, "scaled_yolov4", "exp0_yolov4-p7_chalearnltdmonth_896",    "weights", "best.pt"),
            os.path.join(pretrained_dir, "scaled_yolov4", "exp0_yolov4-p7_chalearnltdmonth_896_2",  "weights", "best.pt"),
            os.path.join(pretrained_dir, "scaled_yolov4", "exp0_yolov4-p7_chalearnltdmonth_1536_2", "weights", "best.pt"),
            os.path.join(pretrained_dir, "scaled_yolov4", "exp0_yolov4-p7_chalearnltdmonth_1536",   "weights", "best.pt"),
            os.path.join(pretrained_dir, "scaled_yolov4", "exp0_yolov4-p7_chalearnltdmonth_1920",   "weights", "best.pt")
        ],
        "source"      : os.path.join(data_dir, "chalearn", "ltd", "val"),
        "output"      : os.path.join("inference", "output"),
        "img_size"    : 1536,
        "batch_size"  : 2,
        "conf_thres"  : 0.01,
        "iou_thres"   : 0.5,
        "device"      : "0",
        "view_img"    : False,
        "save_img"    : False,
        "save_txt"    : True,
        "classes"     : None,
        "agnostic_nms": True,
        "augment"     : True,
        "update"      : False,
        "verbose"     : False,
        "recreate"    : False,
    },
    "yolov4-p7_chalearnltdmonth_896" : {
        "weights"     : [
            os.path.join(pretrained_dir, "scaled_yolov4", "exp0_yolov4-p7_chalearnltdmonth_896_2",  "weights", "best.pt"),
            os.path.join(pretrained_dir, "scaled_yolov4", "exp0_yolov4-p7_chalearnltdmonth_896",    "weights", "best.pt"),
            os.path.join(pretrained_dir, "scaled_yolov4", "exp0_yolov4-p7_chalearnltdmonth_1536_2", "weights", "best.pt"),
            os.path.join(pretrained_dir, "scaled_yolov4", "exp0_yolov4-p7_chalearnltdmonth_1536",   "weights", "best.pt"),
            os.path.join(pretrained_dir, "scaled_yolov4", "exp0_yolov4-p7_chalearnltdmonth_1920",   "weights", "best.pt")
        ],
        "source"      : os.path.join(data_dir, "chalearn", "ltd", "val"),
        "output"      : os.path.join("inference", "output"),
        "img_size"    : 896,
        "batch_size"  : 2,
        "conf_thres"  : 0.01,
        "iou_thres"   : 0.5,
        "device"      : "0",
        "view_img"    : False,
        "save_img"    : False,
        "save_txt"    : True,
        "classes"     : None,
        "agnostic_nms": True,
        "augment"     : True,
        "update"      : False,
        "verbose"     : False,
        "recreate"    : False,
    },
    "yolov4-p7_chalearnltdmonth_384" : {
        "weights"     : os.path.join(pretrained_dir, "scaled_yolov4", "exp0_yolov4-p7_chalearnltdmonth_1536", "weights", "best.pt"),
        "source"      : os.path.join(data_dir, "chalearn", "ltd", "val"),
        "output"      : os.path.join("inference", "output"),
        "img_size"    : 384,
        "batch_size"  : 2,
        "conf_thres"  : 0.35,
        "iou_thres"   : 0.5,
        "device"      : "0",
        "view_img"    : False,
        "save_img"    : False,
        "save_txt"    : False,
        "classes"     : None,
        "agnostic_nms": True,
        "augment"     : True,
        "update"      : False,
        "verbose"     : False,
        "recreate"    : False,
    }
}


# MARK: - Functional

def run_train(args: Namespace):
    torch.backends.cudnn.benchmark = True
    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(False)
    torch.autograd.profiler.emit_nvtx(False)
   
    if args.pre_cfg in train_configs:
        pre_args = train_configs[args.pre_cfg]
        new_args = vars(args) | pre_args
        args     = Namespace(**new_args)
    if not os.path.isfile(args.data):
        args.data = os.path.join(current_dir, str(Path(args.data).name))
    
    train.main(args)


def run_save_weights(path: str):
    ckpt    = torch.load(path, map_location="cpu")
    weights = ckpt["model"].float().state_dict()
    torch.save(weights, path.replace(".pt", "_weights.pt"))


def run_test(args: Namespace):
    if args.pre_cfg in test_configs:
        pre_args = test_configs[args.pre_cfg]
        new_args = vars(args) | pre_args
        args     = Namespace(**new_args)
    test.main(args)
    

def run_detect(args: Namespace):
    if args.pre_cfg in detect_configs:
        pre_args = detect_configs[args.pre_cfg]
        new_args = vars(args) | pre_args
        args     = Namespace(**new_args)
        
    with torch.no_grad():
        if args.update:  # Update all models (to fix SourceChangeWarning)
            for args.weights in [""]:
                Detect(**vars(args)).run()
                strip_optimizer(args.weights)
        else:
            # if not args.run_async:
            Detect(**vars(args)).run()
            # else:
                # DetectAsync(**vars(args)).run()


def create_pkl_from_txts(
    output_path: str   = os.path.join("inference", "output"),
    conf_thres : float = 0.5,
):
    predictions: dict = load_file(
        path=os.path.join(
            data_dir, "chalearn", "ltd", "toolkit",
            "sample_val_predictions.pkl"
        )
    )
    for m, m_dict in predictions.items():
        for d, d_dict in m_dict.items():
            d_dict["boxes"]  = []
            d_dict["labels"] = []
            
    with progress_bar() as pbar:
        for file in pbar.track(
            description="Merging...",
            sequence=glob.glob(os.path.join(output_path, "**", "*.txt"), recursive=True)
        ):
            frame_number = str(Path(file).stem)
            frame_number = frame_number.replace("image_", "")
            clip         = os.path.basename(Path(file).parents[0])
            date         = os.path.basename(Path(file).parents[1])
            month        = os.path.basename(Path(file).parents[3]).capitalize()
            day          = f"{date}_{clip}_{frame_number}"
            if os.path.exists(file):
                with open(file, "r") as d:
                    reader = csv.reader(d, delimiter=" ")
                    for row in reader:
                        cls  = int(row[0])
                        x1   = float(row[1])
                        y1   = float(row[2])
                        x2   = float(row[3])
                        y2   = float(row[4])
                        conf = float(row[5])
                        if conf >= conf_thres:
                            predictions[month][day]["boxes"].append([x1, y1, x2, y2])
                            predictions[month][day]["labels"].append(int(cls))
                            
    # Write final pickle file
    f = open(os.path.join(output_path, f"predictions.pkl"), "wb")
    pickle.dump(predictions, f, protocol=4)
    f.close()


def merge_pkls(output_path: str = os.path.join("inference", "output")):
    predictions = load_file(
        path=os.path.join(
            data_dir, "chalearn", "ltd", "toolkit",
            "sample_val_predictions.pkl"
        )
    )
    with progress_bar() as pbar:
        for file in pbar.track(
            description="Merging...",
            sequence=glob.glob(os.path.join(output_path, "*.pkl"))
        ):
            stem               = str(Path(file).stem)
            idx                = int(stem.replace("predictions", ""))
            month              = months_map_invert[idx].capitalize()
            data               = load_file(file)
            predictions[month] = data[month]
    
    # Write final pickle file
    f = open(os.path.join(output_path, f"predictions.pkl"), "wb")
    pickle.dump(predictions, f, protocol=4)
    f.close()


def fix_cls_id(output_path: str = os.path.join("inference", "output")):
    predictions = load_file(os.path.join(output_path, f"predictions.pkl"))
    with progress_bar() as pbar:
        for _, m in pbar.track(
            description="Processing...", sequence=predictions.items()
        ):
            for _, d in m.items():
                labels      = d["labels"]
                d["labels"] = [l - 1 for l in labels]
            
    # Write final pickle file
    f = open(os.path.join(output_path, f"predictions_fixed.pkl"), "wb")
    pickle.dump(predictions, f, protocol=4)
    f.close()
    

# MARK: - Module

class Detect:
    """Detection pipeline."""
    
    # MARK: Magic Functions
    
    def __init__(
        self,
        months      : Union[str, list[str]] = "*",
        index       : Optional[int]         = None,
        weights     : Union[str, list[str]] = os.path.join(yolov4_pretrained_dir, "yolov4-p7_coco.pt"),
        source      : str                   = os.path.join("inference", "output"),
        output      : str                   = os.path.join("inference", "output"),
        img_size    : int                   = 1536,
        batch_size  : int                   = 1,
        conf_thres  : float                 = 0.4,
        iou_thres   : float                 = 0.5,
        device      : str                   = "",
        view_img    : bool                  = True,
        save_img    : bool                  = False,
        save_txt    : bool                  = False,
        classes     : list[int]             = (),
        agnostic_nms: bool                  = False,
        augment     : bool                  = False,
        update      : bool                  = False,
        verbose     : bool                  = True,
        clear_output: bool                  = False,
        *args, **kwargs,
    ):
        self.months       = months
        self.index        = index
        self.weights      = weights
        self.source       = source
        self.output       = output
        self.img_size     = img_size
        self.batch_size   = batch_size
        self.conf_thres   = conf_thres
        self.iou_thres    = iou_thres
        self.device       = device
        self.view_img     = view_img
        self.save_img     = save_img
        self.save_txt     = save_txt
        self.classes      = classes
        self.agnostic_nms = agnostic_nms
        self.augment      = augment
        self.update       = update
        self.verbose      = verbose
        self.clear_output = clear_output
        self.save_img     = save_img
        self.predictions  = None
        self.init_predictions()
    
    # MARK: Configure
    
    def init_predictions(self):
        # Load the predictions template
        self.predictions = load_file(path=os.path.join(
            data_dir, "chalearn", "ltd", "toolkit",
            "sample_val_predictions.pkl"
        ))
        # Clear all predefined values
        for m, m_dict in self.predictions.items():
            for d, d_dict in m_dict.items():
                d_dict["boxes"]  = []
                d_dict["labels"] = []
        
    # MARK: Process
    
    def run(self):
        months = self.months
        index  = self.index
        webcam = self.source == "0" or self.source.startswith("rtsp") or \
                 self.source.startswith("http") or self.source.endswith(".txt")
        
        # Month subset to infer
        if isinstance(months, str):
            index  = months_map.get(months, 0) if index is None else index
            months = [months]
        elif isinstance(months, (list, tuple)) and len(months) == 1:
            index = months_map.get(months[0], 0) if index is None else index
        if "*" in months:
            months = glob.glob(os.path.join(self.source, "*"))
            index  = 0
        else:
            months = [os.path.join(self.source, m) for m in months]
        
        # Initialize
        ng     = torch.cuda.device_count()
        device = select_device(f"{index % ng}")  # select_device(opt.device)
        create_dirs(paths=[self.output], recreate=self.clear_output)  # Make new output folder
        half   = device.type != "cpu"  # Half precision only supported on CUDA
        
        # Load model
        model = attempt_load(self.weights, map_location=device)  # load FP32 model
        imgsz = check_img_size(self.img_size, s=model.stride.max())  # check img_size
        if half:
            model.half()  # to FP16
    
        # Second-stage classifier
        classify = False
        if classify:
            modelc = load_classifier(name="resnet101", n=2)  # initialize
            modelc.load_state_dict(torch.load("weights/resnet101.pt", map_location=device)["model"])  # load weights
            modelc.to(device).eval()
        
        # Main loop
        for month in months:
            m      = str(Path(month).stem).capitalize()
            prefix = os.path.join(month, "images")

            # Set Dataloader
            pattern              = os.path.join(prefix, "*", "*", "*.jpg")
            vid_path, vid_writer = None, None
            if webcam:
                view_img         = True
                cudnn.benchmark  = True  # set True to speed up constant image size inference
                dataset          = LoadStreams(pattern, img_size=imgsz)
            else:
                dataset          = LoadImages(pattern, img_size=imgsz)
            
            # Get names and colors
            names  = model.module.names if hasattr(model, "module") else model.names
            colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
            
            # Run inference
            t0   = time.time()
            img  = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
            _    = model(img.half() if half else img) if device.type != "cpu" else None  # run once
            
            for path, img, im0s, vid_cap in tqdm(dataset, position=index):
                    img  = torch.from_numpy(img).to(device)
                    img  = img.half() if half else img.float()  # uint8 to fp16/32
                    img /= 255.0  # 0 - 255 to 0.0 - 1.0
                    if img.ndimension() == 3:
                        img = img.unsqueeze(0)
                    
                    # Inference
                    t1   = time_synchronized()
                    pred = model(img, augment=self.augment)[0]
                    
                    # Apply NMS
                    pred = non_max_suppression(
                        prediction = pred,
                        conf_thres = self.conf_thres,
                        iou_thres  = self.iou_thres,
                        classes    = self.classes,
                        agnostic   = self.agnostic_nms
                    )
                    t2   = time_synchronized()
                    
                    # Apply Classifier
                    if classify:
                        pred = apply_classifier(pred, modelc, img, im0s)
                    
                    # Process detections
                    for idx, det in enumerate(pred):  # detections per image
                        if webcam:  # batch_size >= 1
                            p, s, im0 = path[idx], "%g: " % idx, im0s[idx].copy()
                        else:
                            p, s, im0 = path, "", im0s
                        
                        p1         = p.replace(self.source + "/", "")
                        save_path  = os.path.join(self.output, p1)
                        txt_path   = str(os.path.join(self.output, p1).split(".")[0])
                        txt_path  += "_%g" % dataset.frame if dataset.mode == "video" else ""
                        s         += "%gx%g " % img.shape[2:]  # print string
                        gn        = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                        if det is not None and len(det):
                            # Rescale boxes from img_size to im0 size
                            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                            
                            # Print results
                            for c in det[:, -1].unique():
                                n  = (det[:, -1] == c).sum()  # detections per class
                                s += "%g %ss, " % (n, names[int(c)])  # add to string
                            
                            # Save
                            p2       = p.replace(prefix + "/", "")
                            d, c, *_ = p2.split("/")
                            i        = str(Path(p).stem)
                            i        = i.replace("image_", "")
                            dci      = f"{d}_{c}_{i}"
                            
                            # Write results
                            for *xyxy, conf, cls in det:
                                self.predictions[m][dci]["boxes"].append([int(x) for x in xyxy])
                                self.predictions[m][dci]["labels"].append(int(cls))

                                if self.save_txt:  # Write to file
                                    create_dirs(paths=[str(Path(txt_path).parent)])
                                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                                    with open(txt_path + ".txt", "a") as f:
                                        f.write(("%g " * 5 + "%f" + "\n") % (int(cls), *xyxy, conf))  # label format
                    
                                if self.save_img or self.view_img:  # Add bbox to image
                                    label = "%s" % (names[int(cls)])
                                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2)
                    
                        # Print time (inference + NMS)
                        mem = "%.3gG" % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                        if self.verbose:
                            console.log("%sDone. (%s, %.3fs)" % (s, mem, t2 - t1))
                    
                        # Stream results
                        if self.view_img:
                            cv2.imshow(p, im0)
                            if cv2.waitKey(1) == ord("q"):  # q to quit
                                raise StopIteration
                    
                        # Save results (image with detections)
                        if self.save_img:
                            if dataset.mode == "images":
                                cv2.imwrite(save_path, im0)
                            else:
                                if vid_path != save_path:  # new video
                                    vid_path = save_path
                                    if isinstance(vid_writer, cv2.VideoWriter):
                                        vid_writer.release()  # release previous video writer
                    
                                    fourcc     = "mp4v"  # output video codec
                                    fps        = vid_cap.get(cv2.CAP_PROP_FPS)
                                    w          = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                    h          = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                                vid_writer.write(im0)
                                
            if self.save_txt or self.save_img:
                if self.verbose:
                    console.log("Results saved to %s" % Path(self.output))
                if platform == "darwin" and not self.update:  # MacOS
                    os.system("open " + save_path)
            
            if self.verbose:
                console.log("Done. (%.3fs)" % (time.time() - t0))
        
        # Write final pickle file
        index = "" if index is None else index
        f     = open(os.path.join(self.output, f"predictions{index}.pkl"), "wb")
        pickle.dump(self.predictions, f, protocol=4)
        f.close()

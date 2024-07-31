#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import pathlib
import socket
import sys
import time

import click
import cv2
import torch
import torch.backends.cudnn as cudnn
import yaml
from numpy import random

_root = pathlib.Path(__file__).resolve().parents[0]  # root directory
if str(_root) not in sys.path:
    sys.path.append(str(_root))  # add ROOT to PATH

from models.experimental import attempt_load
import mon
from utils.datasets import LoadImages, LoadStreams
from utils.general import (
    apply_classifier, check_img_size, non_max_suppression, scale_coords, set_logging,
    strip_optimizer, xyxy2xywh,
)
from utils.plots import plot_one_box
from utils.torch_utils import load_classifier, select_device, time_synchronized

console       = mon.console
current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# region Predict

def load_classes(path):
    # Loads *.names file at "path"
    with open(path, "r") as f:
        names = f.read().split("\n")
    return list(filter(None, names))  # filter removes empty strings (such as last line)


def predict(opt, save_img: bool = False):
    source   = opt.source
    save_dir = mon.Path(opt.save_dir)
    weights  = opt.weights
    weights  = weights[0] if isinstance(weights, list | tuple) and len(weights) == 1 else weights
    view_img = opt.view_img
    save_txt = opt.save_txt
    imgsz    = opt.imgsz
    names    = opt.names
    webcam   = source == "0" or source.startswith("rtsp") or source.startswith("http") or source.endswith(".txt")
    
    # Directories
    (save_dir / "images" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    
    # Initialize
    set_logging()
    device = select_device(opt.device)
    half   = device.type != "cpu"  # half precision only supported on CUDA
    
    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = imgsz[0] if isinstance(imgsz, list | tuple) else imgsz
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name="resnet101", n=2)  # initialize
        modelc.load_state_dict(torch.load("weights/resnet101.pt", map_location=device)["model"]).to(device).eval()
    
    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img        = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset         = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset  = LoadImages(source, img_size=imgsz, auto_size=64)
    
    # Get names and colors
    if hasattr(model, "module"):
        _module = model.module
        if hasattr(_module, "names"):
            names = _module.names
        elif hasattr(_module, "nc"):
            names = [i for i in range(_module.nc)]
        else:
            names = None
    elif hasattr(model, "names"):
        names = model.names
    elif hasattr(model, "nc"):
        names = [i for i in range(model.nc)]
    else:
        # names = None
        names = [i for i in range(5)]
    
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    with open(opt.data, encoding="utf-8") as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)  # data dict
        colors    = data_dict.get("colors", colors)

    # Run inference
    t0  = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _   = model(img.half() if half else img) if device.type != "cpu" else None  # run once
    for path, img, im0s, vid_cap in dataset:
        img  = torch.from_numpy(img).to(device)
        img  = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1   = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(
            prediction = pred,
            conf_thres = opt.conf,
            iou_thres  = opt.iou,
            max_det    = opt.max_det,
            classes    = opt.classes,
            agnostic   = opt.agnostic_nms,
        )
        t2   = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], "%g: " % i, im0s[i].copy()
            else:
                p, s, im0 = path, "", im0s

            save_path  = str(save_dir / "images" / f"{mon.Path(p).stem}.jpg")
            txt_path   = str(save_dir / "labels" / mon.Path(p).stem) + ("_%g" % dataset.frame if dataset.mode == "video" else "")
            s         += "%gx%g " % img.shape[2:]  # print string
            gn         = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n  = (det[:, -1] == c).sum()  # detections per class
                    s += "%g %ss, " % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in det:
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        # xywh     = (xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist()  # xywh
                        # xywh[0] -= (xywh[2] / 2)  # xy center to top-left corner
                        # xywh[1] -= (xywh[3] / 2)  # xy center to top-left corner
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + ".txt", "a") as f:
                            f.write(("%g " * len(line)).rstrip() % line + "\n")

                    if save_img or view_img:  # Add bbox to image
                        label = "%s %.2f" % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            # Print time (inference + NMS)
            print("%sDone. (%.3fs)" % (s, t2 - t1))

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord("q"):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == "images":
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = "mp4v"  # output video codec
                        fps    = vid_cap.get(cv2.CAP_PROP_FPS)
                        w      = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h      = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)
    
    if save_txt or save_img:
        print("Results saved to %s" % save_dir)
    
    print("Done. (%.3fs)" % (time.time() - t0))

# endregion


# region Main


def main() -> str:
    # Parse args
    args        = mon.parse_predict_args(model_root=_current_dir)
    model       = mon.Path(args.model)
    model       = model if model.exists() else _current_dir / "config" / model.name
    model       = str(model.config_file())
    data_       = mon.Path(args.data)
    data_       = data_ if data_.exists() else _current_dir / "data" / data_.name
    data_       = str(data_.config_file())
    args.model  = model
    args.source = args.data
    args.data   = data_
    
    with torch.no_grad():
        if args.update:  # update all models (to fix SourceChangeWarning)
            for args.weights in [
                "yolor_p6.pt", "yolor_w6.pt", "yolor_e6.pt", "yolor_d6.pt",
                "yolor-p6.pt", "yolor-w6.pt", "yolor-e6.pt", "yolor-d6.pt"
            ]:
                predict(args)
                strip_optimizer(args.weights)
        else:
            predict(args)


if __name__ == "__main__":
    main()

# endregion

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

import argparse
import os
import time
from pathlib import Path
from typing import Union

import cv2
import torch
import torch.backends.cudnn as cudnn
from munch import Munch
from numpy import random

from one import pretrained_dir
from one.vision.detection.yolov7.models.experimental import attempt_load
from one.vision.detection.yolov7.utils.datasets import LoadImages
from one.vision.detection.yolov7.utils.datasets import LoadStreams
from one.vision.detection.yolov7.utils.general import apply_classifier
from one.vision.detection.yolov7.utils.general import check_img_size
from one.vision.detection.yolov7.utils.general import check_imshow
from one.vision.detection.yolov7.utils.general import increment_path
from one.vision.detection.yolov7.utils.general import non_max_suppression
from one.vision.detection.yolov7.utils.general import scale_coords
from one.vision.detection.yolov7.utils.general import set_logging
from one.vision.detection.yolov7.utils.general import strip_optimizer
from one.vision.detection.yolov7.utils.general import xyxy2xywh
from one.vision.detection.yolov7.utils.plots import plot_one_box
from one.vision.detection.yolov7.utils.torch_utils import load_classifier
from one.vision.detection.yolov7.utils.torch_utils import select_device
from one.vision.detection.yolov7.utils.torch_utils import time_synchronized
from one.vision.detection.yolov7.utils.torch_utils import TracedModel


# MARK: - Functional

def detect(
    opt     : Union[dict, Munch, argparse.Namespace],
    save_img: bool = False
):
    source   = opt.source
    weights  = opt.weights
    view_img = opt.view_img
    save_txt = opt.save_txt
    imgsz    = opt.img_size
    trace    = not opt.no_trace
    save_img = not opt.nosave and not source.endswith(".txt")  # save inference images
    webcam   = source.isnumeric() or source.endswith(".txt") or source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half   = device.type != "cpu"  # half precision only supported on CUDA

    # Load model
    model  = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz  = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name="resnet101", n=2)  # initialize
        modelc.load_state_dict(torch.load("weights/resnet101.pt", map_location=device)["model"]).to(device).eval()

    # Set Dataloader
    vid_path   = None
    vid_writer = None
    if webcam:
        view_img        = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset         = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names  = model.module.names if hasattr(model, "module") else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != "cpu":
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
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
            conf_thres = opt.conf_thres,
            iou_thres  = opt.iou_thres,
            classes    = opt.classes,
            agnostic   = opt.agnostic_nms
        )
        t2   = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], "%g: " % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, "", im0s, getattr(dataset, "frame", 0)

            p         = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path  = str(save_dir / "labels" / p.stem) + ("" if dataset.mode == "image" else f"_{frame}")  # img.txt
            s        += "%gx%g " % img.shape[2:]  # print string
            gn        = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n  = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + ".txt", "a") as f:
                            f.write(("%g " * len(line)).rstrip() % line + "\n")

                    if save_img or view_img:  # Add bbox to image
                        label = f"{names[int(cls)]} {conf:.2f}"
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            # Print time (inference + NMS)
            # print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == "image":
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # "video" or "stream"
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w   = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h   = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += ".mp4"
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ""
        # print(f"Results saved to {save_dir}{s}")

    print(f"Done. ({time.time() - t0:.3f}s)")


def parse_opt():
    yolov7_dir = os.path.join(pretrained_dir, "yolov7")
    parser     = argparse.ArgumentParser()
    parser.add_argument("--weights",      default=os.path.join(yolov7_dir, "yolov7-coco.pt"), nargs="+", type=str, help="model.pt path(s)")
    parser.add_argument("--source",       default="inference/images", type=str,            help="Source")  # file/folder, 0 for webcam
    parser.add_argument("--img-size",     default=640,                type=int,            help="Inference size (pixels)")
    parser.add_argument("--conf-thres",   default=0.25,               type=float,          help="Object confidence threshold")
    parser.add_argument("--iou-thres",    default=0.45,               type=float,          help="IOU threshold for NMS")
    parser.add_argument("--device",       default="",                                      help="Cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--view-img",                                 action="store_true", help="Display results")
    parser.add_argument("--save-txt",                                 action="store_true", help="Save results to *.txt")
    parser.add_argument("--save-conf",                                action="store_true", help="Save confidences in --save-txt labels")
    parser.add_argument("--nosave",                                   action="store_true", help="Do not save images/videos")
    parser.add_argument("--classes",                                  nargs="+", type=int, help="Filter by class: --class 0, or --class 0 2 3")
    parser.add_argument("--agnostic-nms",                             action="store_true", help="Class-agnostic NMS")
    parser.add_argument("--augment",                                  action="store_true", help="Augmented inference")
    parser.add_argument("--update",                                   action="store_true", help="Update all models")
    parser.add_argument("--project",      default="runs/detect",                           help="Save results to project/name")
    parser.add_argument("--name",         default="exp",                                   help="Save results to project/name")
    parser.add_argument("--exist-ok",                                 action="store_true", help="Existing project/name ok, do not increment")
    parser.add_argument("--no-trace",                                 action="store_true", help="Don`t trace model")
    parser.add_argument("--verbose",      default=True,               action="store_true", help="Verbosity")
    opt = parser.parse_args()
    print(opt)
    return opt


def main(opt: Union[dict, Munch, argparse.Namespace]):
    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ["yolov7.pt"]:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()


# MARK: - Main

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
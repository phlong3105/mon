#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import sys
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from one.utils import pretrained_dir
from one.vision.detection.scaled_yolov4.models.experimental import attempt_load
from one.vision.detection.scaled_yolov4.utils.datasets import LoadImages
from one.vision.detection.scaled_yolov4.utils.datasets import LoadStreams
from one.vision.detection.scaled_yolov4.utils.general import apply_classifier
from one.vision.detection.scaled_yolov4.utils.general import check_img_size
from one.vision.detection.scaled_yolov4.utils.general import non_max_suppression
from one.vision.detection.scaled_yolov4.utils.general import plot_one_box
from one.vision.detection.scaled_yolov4.utils.general import scale_coords
from one.vision.detection.scaled_yolov4.utils.general import strip_optimizer
from one.vision.detection.scaled_yolov4.utils.general import xyxy2xywh
from one.vision.detection.scaled_yolov4.utils.torch_utils import load_classifier
from one.vision.detection.scaled_yolov4.utils.torch_utils import select_device
from one.vision.detection.scaled_yolov4.utils.torch_utils import time_synchronized

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


# MARK: - Functional

def detect(opt, save_img=False):
    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source == "0" or source.startswith("rtsp") or source.startswith("http") or source.endswith(".txt")

    # Initialize
    device = select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != "cpu"  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name="resnet101", n=2)  # initialize
        modelc.load_state_dict(torch.load("weights/resnet101.pt", map_location=device)["model"])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img        = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset         = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset  = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names  = model.module.names if hasattr(model, "module") else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

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
            pred,
            opt.conf_thres,
            opt.iou_thres,
            classes  = opt.classes,
            agnostic = opt.agnostic_nms
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

            save_path = str(Path(out) / Path(p).name)
            txt_path  = str(Path(out) / Path(p).stem) + ("_%g" % dataset.frame if dataset.mode == "video" else "")
            s  += "%gx%g " % img.shape[2:]  # print string
            gn  = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
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
                        with open(txt_path + ".txt", "a") as f:
                            if cls == 0:
                                name = "vehicle"
                                f.write(("%s " + "%.2f " * 5 + "\n") % (name, *xyxy, conf))  # label format
                            else:
                                f.write(("%g " * 5 + "\n") % (cls, *xywh))  # label format

                    if save_img or view_img:  # Add bbox to image
                        label = "%s" % (names[int(cls)])
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=2)

            # Print time (inference + NMS)
            mem = "%.3gG" % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
            print("%sDone. (%s, %.3fs)" % (s, mem, t2 - t1))

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

                        fourcc     = "mp4v"  # output video codec
                        fps        = vid_cap.get(cv2.CAP_PROP_FPS)
                        w          = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h          = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print("Results saved to %s" % Path(out))
        if platform == "darwin" and not opt.update:  # MacOS
            os.system("open " + save_path)

    print("Done. (%.3fs)" % (time.time() - t0))


def parse_opt():
    scaled_yolov4_dir = os.path.join(pretrained_dir, "scaled_yolov4")
    parser            = argparse.ArgumentParser()
    parser.add_argument(
        "--weights", default=os.path.join(scaled_yolov4_dir, "yolov4_p7_coco.pt"), nargs="+", type=str, help="model.pt path(s)"
    )
    parser.add_argument("--source",       default=ROOT / "inference/images", type=str, help="source")  # file/folder, 0 for webcam
    parser.add_argument("--output",       default=ROOT / "inference/output", type=str, help="output folder")  # output folder
    parser.add_argument("--img-size",     default=1536,                      type=int, help="inference size (pixels)")
    parser.add_argument("--conf-thres",   default=0.4,                       type=float, help="object confidence threshold")
    parser.add_argument("--iou-thres",    default=0.5,                       type=float, help="IOU threshold for NMS")
    parser.add_argument("--device",       default="",                        help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--view-img",     default=True,                      action="store_true", help="display results")
    parser.add_argument("--save-txt",     default=False,                     action="store_true", help="save results to *.txt")
    parser.add_argument("--classes",                                         nargs="+", type=int, help="filter by class: --class 0, or --class 0 2 3")
    parser.add_argument("--agnostic-nms", default=False,                     action="store_true", help="class-agnostic NMS")
    parser.add_argument("--augment",      default=False,                     action="store_true", help="augmented inference")
    parser.add_argument("--update",       default=False,                     action="store_true", help="update all models")
    parser.add_argument("--verbose",      default=True,                      action="store_true", help="")
    opt = parser.parse_args()
    print(opt)
    return opt


def main(opt):
    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in [""]:
                detect(opt)
                strip_optimizer(opt.weights)
        else:
            detect(opt)


# MARK: - Main

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

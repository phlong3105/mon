#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
import platform
import socket
import sys
from pathlib import Path

import click
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLO root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from mon import core
from utils.dataloaders import IMG_FORMATS, LoadImages, LoadScreenshots, LoadStreams, VID_FORMATS
from utils.general import (
    LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2, increment_path, non_max_suppression,
    Profile, scale_boxes, strip_optimizer, xyxy2xywh,
)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode

console       = core.console
_current_file = core.Path(__file__).absolute()
_current_dir  = _current_file.parents[0]


# region Predict

@smart_inference_mode()
def run(opt, nosave: bool = False):
    weights        = opt.weights
    weights        = weights[0] if isinstance(weights, list | tuple) and len(weights) == 1 else weights
    source         = opt.source
    data           = opt.data
    save_dir       = core.Path(opt.save_dir)
    imgsz          = opt.imgsz
    conf_thres     = opt.conf_thres
    iou_thres      = opt.iou_thres
    max_det        = opt.max_det
    classes        = opt.classes
    augment        = opt.augment
    agnostic_nms   = opt.agnostic_nms
    dnn            = opt.dnn
    half           = opt.half
    vid_stride     = opt.vid_stride
    visualize      = opt.visualize
    view_img       = opt.view_img
    save_txt       = opt.save_txt
    save_crop      = opt.save_crop
    save_conf      = opt.save_conf
    line_thickness = opt.line_thickness
    hide_labels    = opt.hide_labels
    hide_conf      = opt.hide_conf
    
    save_img   = not nosave and not source.endswith(".txt")  # save inference images
    is_file    = core.Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url     = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    webcam     = source.isnumeric() or source.endswith(".txt") or (is_url and not is_file)
    screenshot = source.lower().startswith("screen")
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    # save_dir = increment_path(core.Path(project) / name, exist_ok=exist_ok)  # increment run
    # (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    (save_dir / "images" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    
    # Load model
    device = select_device(opt.device)
    model  = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz  = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset  = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs       = len(dataset)
    elif screenshot:
        dataset  = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset  = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im  = torch.from_numpy(im).to(model.device)
            im  = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / core.Path(path).stem, mkdir=True) if visualize else False
            pred      = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(
                prediction = pred,
                conf_thres = conf_thres,
                iou_thres  = iou_thres,
                classes    = classes,
                agnostic   = agnostic_nms,
                max_det    = max_det,
            )

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f"{i}: "
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)

            p         = core.Path(p)  # to Path
            save_path = str(save_dir / "images" / f"{p.stem}.jpg")
            txt_path  = str(save_dir / "labels" / p.stem) + ("" if dataset.mode == "image" else f"_{frame}")  # im.txt
            s        += "%gx%g " % im.shape[2:]  # print string
            gn        = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc       = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n  = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f"{txt_path}.txt", "a") as f:
                            f.write(("%g " * len(line)).rstrip() % line + "\n")

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c     = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f"{names[c]} {conf:.2f}")
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / "crops" / names[c] / f"{p.stem}.jpg", BGR=True)
            
            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == "Linux" and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == "image":
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w   = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h   = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path     = str(core.Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    
    if opt.update:
        strip_optimizer(weights[0])

# endregion


# region Train

@click.command(name="predict", context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.option("--root",       type=str, default=None, help="Project root.")
@click.option("--config",     type=str, default=None, help="Model config.")
@click.option("--weights",    type=str, default=None, help="Weights paths.")
@click.option("--model",      type=str, default=None, help="Model name.")
@click.option("--data",       type=str, default=None, help="Source data directory.")
@click.option("--fullname",   type=str, default=None, help="Save results to root/run/predict/fullname.")
@click.option("--save-dir",   type=str, default=None, help="Optional saving directory.")
@click.option("--device",     type=str, default=None, help="Running devices.")
@click.option("--imgsz",      type=int, default=None, help="Image sizes.")
@click.option("--resize",     is_flag=True)
@click.option("--benchmark",  is_flag=True)
@click.option("--save-image", is_flag=True)
@click.option("--verbose",    is_flag=True)
def main(
    root      : str,
    config    : str,
    weights   : str,
    model     : str,
    data      : str,
    fullname  : str,
    save_dir  : str,
    device    : str,
    imgsz     : int,
    resize    : bool,
    benchmark : bool,
    save_image: bool,
    verbose   : bool,
) -> str:
    hostname = socket.gethostname().lower()
    
    # Get config args
    config = core.parse_config_file(project_root=_current_dir.parent / "config", config=config)
    args   = core.load_config(config)
    
    # Prioritize input args --> config file args
    root     = root      or args["root"]
    root     = core.Path(root)
    weights  = weights   or args["weights"]
    weights  = weights   or args["weights"]
    model    = core.Path(model or args["model"])
    model    = model if model.exists() else _current_dir / "config"  / model.name
    model    = model.config_file()
    data_    = core.Path(args["data"])
    data_    = data_ if data_.exists() else _current_dir.parent / "data" / data_.name
    data_    = data_.config_file()
    data     = data      or args["source"]
    project  = root.name or args["project"]
    fullname = fullname  or args["name"]
    save_dir = save_dir  or root / "run" / "predict" / model
    save_dir = core.Path(save_dir)
    device   = device    or args["device"]
    imgsz    = imgsz     or args["imgsz"]
    verbose  = verbose   or args["verbose"]
    
    # Update arguments
    args["root"]     = root
    args["config"]   = config
    args["weights"]  = core.to_list(weights)
    args["model"]    = str(model)
    args["data"]     = str(data_)
    args["source"]   = data
    args["project"]  = project
    args["name"]     = fullname
    args["save_dir"] = save_dir
    args["device"]   = device
    args["imgsz"]    = core.to_list(imgsz)
    args["verbose"]  = verbose
    
    opt        = argparse.Namespace(**args)
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand

    run(opt)
    return str(opt.save_dir)


if __name__ == "__main__":
    main()

# endregion

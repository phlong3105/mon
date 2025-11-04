#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""DEIM model prediction pipeline for object detection.

References:
    - Paper: "DEIM: DETR with Improved Matching for Fast convergence," CVPR 2025.
    - Code: https://github.com/ShihuaHuang95/DEIM
"""

import os
import sys
from datetime import datetime

import box
import cv2
import torch
import fjson
import albumentations as A
import box
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

import mon
from mon import console, metrics, Path, tfms, optims
import torchvision.transforms.v2 as T

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from engine.core import YAMLConfig

current_file = Path(__file__).absolute()
root_dir     = current_file.parents[0]


# ----- Utils -----
def benchmark(model: nn.Module):
    params, macs, flops = mon.nn.compute_model_stats(model=model)
    mon.log(f"Params    : {params:.4f}")
    mon.log(f"MACs      : {macs:.4f}")
    mon.log(f"FLOPs     : {flops:.4f}")


# ----- Predict -----
class Model(torch.nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.model         = cfg.model.deploy()
        self.postprocessor = cfg.postprocessor.deploy()

    def forward(self, images, orig_target_sizes):
        outputs = self.model(images)
        outputs = self.postprocessor(outputs, orig_target_sizes)
        return outputs


@torch.no_grad()
def predict(args: dict | box.Box) -> str:
    # Start
    mon.rt.print_run_summary(args)

    # Device
    device = mon.create_device(args.device)

    # Seed
    mon.set_random_seed(args.seed)
    
    # Data I/O
    data_name, data_loader = mon.parse_data_loader(args.data, args.root, True, verbose=False)
    
    # Pretrained
    pretrained = args.resume
    if args.weights and args.weights.is_weights_file(exist=True):
        pretrained = args.weights
    if pretrained and pretrained.is_weights_file(exist=True):
        mon.log(f"Pretrained: {pretrained}.")
    else:
        raise ValueError(f"Invalid weights file: {pretrained}.")

    # Model
    cfg_path     = root_dir / "option" / args.cfg
    updated_cfg  = args.updated_cfg
    updated_cfg |= {"resume": str(pretrained)} if pretrained else {}
    updated_cfg |= {
        "device": device,
        "seed"  : args.seed,
    }
    cfg = YAMLConfig(cfg_path=str(cfg_path), root=str(args.root), **updated_cfg)

    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    if pretrained:
        checkpoint = torch.load(pretrained, map_location="cpu")
        if "ema" in checkpoint:
            state = checkpoint["ema"]["module"]
        else:
            state = checkpoint["model"]
    else:
        raise AttributeError("Only support resume to load model.state_dict by now.")

    # Load train mode state and convert to deploy mode
    cfg.model.load_state_dict(state)
    model = Model(cfg)
    model = model.to(device)
    model = model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # Predict
    """
    # COCO JSON Format
    json_path   = args.save_dir / f"{data_name}.json"
    info        = {
        "year"        : f"{datetime.now().year}",
        "version"     : "1",
        "description" : f"{data_name} predictions",
        "contributor" : "Long H. Pham",
        "url"         : "",
        "date_created": f"{datetime.now()}"
    }
    licenses    = []
    categories  = []
    images      = []
    annotations = []
    ann_id      = 0
    """
    timers = mon.TimeProfiler()
    timers.total.tick()
    with mon.create_progress_bar() as pbar:
        for i, datapoint in pbar.track(
            sequence    = enumerate(data_loader),
            total       = len(data_loader),
            description = f"[bright_yellow]Predicting"
        ):
            # Preprocess
            timers.preprocess.tick()
            path   = Path(datapoint["meta"]["path"])
            image  = datapoint["image"]
            h0, w0 = mon.image.imgsz(image)
            size0  = torch.tensor([[w0, h0]]).to(device)
            if args.resize and (h0 != args.imgsz[0] or w0 != args.imgsz[1]):
                # image = mon.resize(image, size=args.imgsz, pad=True)  # Use for: exdark
                image = mon.resize(image, size=args.imgsz)
            image  = image.to(device)
            timers.preprocess.tock()

            # Infer
            timers.infer.tick()
            outputs = model(image, size0)
            timers.infer.tock()

            # Postprocess
            timers.postprocess.tick()
            labels, boxes, scores = outputs
            scores = [s.cpu().numpy().astype(float) for s in scores]  # batch_size = 1
            labels = [l.cpu().numpy().astype(int)   for l in labels]  # batch_size = 1
            boxes  = [b.cpu().numpy().astype(float) for b in  boxes]  # batch_size = 1, XYWH format, change "deploy_out_fmt" in config file.
            timers.postprocess.tock()

            # Save
            if args.save_result:
                out_dir    = mon.rt.parse_output_dir(args.save_dir, data_name, mon.SAVE_LABEL_DIR, path, args.keep_subdirs, args.save_nearby)
                label_path = out_dir / f"{path.stem}.txt"
                label_path.parent.mkdir(parents=True, exist_ok=True)
                with open(str(label_path), "w") as f:
                    for j, img in enumerate(image):
                        ss = scores[j]
                        cs = labels[j][ss >= args.conf_thres]
                        bs = boxes[j][ss >= args.conf_thres]
                        if len(bs) == 0:
                            continue
                        bs = mon.convert_hbb(bbox=bs, fmt=mon.BBoxFormat.VOC2YOLO, imgsz=(h0, w0))
                        for c, b, s in zip(cs, bs, ss):
                            f.write(f"{c} {b[0]} {b[1]} {b[2]} {b[3]} {s}\n")

                """
                # json_path = out_dir.parent / f"{data_name}.json"
                json_path = out_dir.parent / f"{out_dir.stem}.json"
                # Append image
                images.append({"id": i, "file_name": path.name, "height": h0, "width": w0})
                # Append annotations
                if len(boxes) == 0:
                    continue
                for j, (c, b, s) in enumerate(zip(labels, boxes, scores)):
                    annotations.append({
                        "id"         : ann_id,
                        "image_id"   : i,
                        "category_id": int(c),
                        "bbox"       : b[0:4].tolist(),
                        "area"       : float(b[2] * b[3]),
                        "score"      : float(s),
                        "iscrowd"    : 0,
                    })
                    ann_id += 1
                """
    timers.total.tock()

    """
    # Save
    if args.save_result:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        # Write to JSON file
        json_data = {
            "info"       : info,
            "licenses"   : licenses,
            "categories" : categories,
            "images"     : images,
            "annotations": annotations
        }
        with open(str(json_path), "w") as f:
            fjson.dump(json_data, f, float_format=".32f", indent=None)
    """

    # Finish
    timers.print()
    return str(args.save_dir)


# ----- Main -----
def main() -> str:
    cli  = mon.rt.parse_cli_args(root=root_dir)
    data = mon.utils.to_list(cli.data)
    for d in data:
        cli_ = copy.deepcopy(cli)
        cli_.data = d
        args = mon.rt.parse_predict_args(cli=cli_, root=root_dir, model_root=root_dir)
        predict(args)


if __name__ == "__main__":
    main()

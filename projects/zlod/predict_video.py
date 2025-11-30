#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json

import box
import cv2
import numpy as np
import torch

import mon
from mon import albumentations as A
from mon.cv import DEIM
from zlod import GCENet, GCENet_Baseline

current_file   = mon.Path(__file__).absolute()
current_dir    = current_file.parents[0]
project_dir    = current_file.parents[0]

gcenet_weights = project_dir / "zoo/gcenet.pt"
deim_config    = project_dir / "config/deim_dfine_s_coco80.yaml"
deim_weights   = project_dir / "zoo/deim_dfine_s_coco80.pth"
camera_configs = {
    "darkface"      : {
        "iters"     : 8,
        "conf_thres": 0.40,
        "max_fps"   : 1.0,
    },
    "ic_11_01"      : {
        "iters"     : 8,
        "conf_thres": 0.40,
        "max_fps"   : 30.0,
    },
    "lolistreet"    : {
        "iters"     : 8,
        "conf_thres": 0.35,
        "max_fps"   : 1.0,
    },
    "ydld"          : {
        "iters"     : 8,
        "conf_thres": 0.40,
        "max_fps"   : 1.0,
    },
    "ic_04_03"      : {
        "iters"     : 3,
        "conf_thres": 0.35,
        "max_fps"   : 30.0,
    },
    "suwon#11_04_02": {
        "iters"     : 3,
        "conf_thres": 0.35,
        "max_fps"   : 30.0,
    },
    "suwon#54_02_01": {
        "iters"     : 4,
        "conf_thres": 0.35,
        "max_fps"   : 30.0,
    },
    "suwon#56_02_01": {
        "iters"     : 4,
        "conf_thres": 0.35,
        "max_fps"   : 30.0,
    },
    "suwon#59_02_03": {
        "iters"     : 8,
        "conf_thres": 0.35,
        "max_fps"   : 30.0,
    },
    "suwon#60_02_01": {
        "iters"     : 8,
        "conf_thres": 0.35,
        "max_fps"   : 30.0,
    },
    "suwon#71_06_01": {
        "iters"     : 4,
        "conf_thres": 0.35,
        "max_fps"   : 30.0,
    },
    "suwon#77_06_01": {
        "iters"     : 8,
        "conf_thres": 0.35,
        "max_fps"   : 30.0,
    },
}


# ----- Utils -----
def postprocess_outputs(outputs, conf_thres=0.50):
    labels, boxes, scores = outputs
    labels = labels.cpu().numpy().astype(np.int32)[0]    # batch_size = 1
    boxes  =  boxes.cpu().numpy().astype(np.float32)[0]  # batch_size = 1, XYXY format
    scores = scores.cpu().numpy().astype(np.float32)[0]  # batch_size = 1
    #
    labels = labels[scores >= conf_thres]
    boxes  =  boxes[scores >= conf_thres]
    scores = scores[scores >= conf_thres]
    return labels, boxes, scores


def write_results_txt(file_path, labels, boxes, scores):
    with open(file_path, "w") as f:
        if len(boxes) == 0:
            return
        for c, b, s in zip(labels, boxes, scores):
            f.write(f"{c} {b[0]} {b[1]} {b[2]} {b[3]} {s}\n")


def draw_bboxes(image, outputs, classes, conf_thres=0.50):
    labels, boxes, scores = postprocess_outputs(outputs, conf_thres)
    for c, b, s in zip(labels, boxes, scores):
        if c >= len(classes):
            continue
        image = mon.dtypes.draw_bbox(
            image     = image,
            bbox      = b,
            label     = "",             # f"{c}",
            color     = [49, 3, 150],   # classes[c]["color"],
            thickness = 2,
            fill      = False,
        )
    return image


# ----- Predict -----
@torch.no_grad()
def predict(args: dict | box.Box) -> str:
    # Parse input arguments
    data_dir       = project_dir / "data" / args.data_dir
    origin_video   = data_dir / "origin.mp4"
    enhanced_video = data_dir / "enhanced.mp4"
    classes_file   = data_dir / "classes.yaml"
    pred_dir       = data_dir / "pred"
    pred_image_dir = pred_dir / "image"
    pred_label_dir = pred_dir / "label"
    pred_json      = pred_dir / "pred.json"
    video_wrt      = None
    imgsz          = args.imgsz
    conf_thres     = camera_configs[str(data_dir.stem)]["conf_thres"]
    device         = torch.device(args.device) if args.device != "cpu" else "cpu"
    
    # Build models
    iters  = camera_configs[str(data_dir.stem)]["iters"]
    gcenet = GCENet(iters=iters, weights=gcenet_weights)
    gcenet = gcenet.to(device)
    gcenet = gcenet.eval()
    
    deim_cfg = mon.rt.load_config(deim_config)
    deim     = DEIM(
        cfg         = deim_cfg.cfg,
        weights     = deim_weights,
        root        = current_dir,
        device      = device,
        seed        = deim_cfg.seed,
        updated_cfg = deim_cfg.updated_cfg,
    )
    deim = deim.to(device)
    deim = deim.eval()
    for deim_param in deim.parameters():
        deim_param.requires_grad = False
    
    # Data I/O
    transform = A.Compose([
        A.ResizeDivisibleBy(height=imgsz, width=imgsz, divisor=32),
        A.Normalize(normalization="min_max"),
        A.ToTensorV2(transpose_mask=True),
    ], additional_targets={"enhanced": "image"})
    # Video reader
    origin_cap   = cv2.VideoCapture(str(origin_video))
    enhanced_cap = cv2.VideoCapture(str(enhanced_video))
    if args.early_stop:
        num_frames = 1800
    else:
        num_frames = int(origin_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Read classes from YAML file
    classes = mon.rt.load_config(classes_file, verbose=False)
    classes = classes.get("classes", [])
    
    # Predict
    predictions       = []
    debug_predictions = []
    timers = mon.TimeProfiler()
    timers.total.tick()
    with mon.create_progress_bar() as pbar:
        for i in pbar.track(
            sequence    = range(0, num_frames),
            total       = num_frames,
            description = f"[bright_yellow]Predicting"
        ):
            # Input
            ret, image    = origin_cap.read()
            _  , enhanced = enhanced_cap.read()
            if not ret:
                break
            image    = cv2.cvtColor(image,    cv2.COLOR_BGR2RGB)
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
            
            # Preprocess
            timers.preprocess.tick()
            h0, w0          = mon.image.imgsz(image)
            size0           = torch.tensor([[w0, h0]]).to(device)
            augmented       = transform(image=image, enhanced=enhanced)
            image_tensor    =    augmented["image"].unsqueeze(0).to(device)  # Add batch dimension and to device
            enhanced_tensor = augmented["enhanced"].unsqueeze(0).to(device)  # Add batch dimension and to device
            timers.preprocess.tock()
            
            # Infer
            timers.infer.tick()
            if args.lle:
                enhanced_tensor = gcenet(image_tensor, inference=False)["output"]
                # enhanced_tensor = gcenet(image_tensor, inference=False)[-1]
            outputs = deim(enhanced_tensor, size0)
            if args.debug:
                debug_outputs = deim(image_tensor, size0)
            timers.infer.tock()
            
            # Postprocess
            timers.postprocess.tick()
            labels, boxes, scores = postprocess_outputs(outputs, 0.25)
            enhanced = mon.image.to_array(enhanced_tensor)
            h1, w1   = mon.image.imgsz(enhanced)
            if (h1, w1) != (h0, w0):
                enhanced = cv2.resize(enhanced, (w0, h0))
            timers.postprocess.tock()
            
            # Save predictions to JSON file
            boxes_ = mon.bbox.convert(bbox=boxes, fmt=mon.BBoxFormat.VOC2COCO, imgsz=(h0, w0))
            for c, b, s in zip(labels, boxes_, scores):
                predictions.append({
                    "image_id"   : i,
                    "category_id": int(c),
                    "bbox"       : [
                        round(float(b[0]), 32),
                        round(float(b[1]), 32),
                        round(float(b[2]), 32),
                        round(float(b[3]), 32)
                    ],
                    "score"      : float(s),
                })
            # Save predictions to text files
            pred_label_path = pred_label_dir / f"{i}.txt"
            pred_label_path.parent.mkdir(parents=True, exist_ok=True)
            write_results_txt(pred_label_path, labels, boxes, scores)
            # Save image with bboxes
            output_image = draw_bboxes(enhanced, outputs, classes, conf_thres)
            
            # Save debug
            if args.debug:
                d_labels, d_boxes, d_scores = postprocess_outputs(debug_outputs)
                # Save predictions to JSON file
                d_boxes_ = mon.bbox.convert(bbox=d_boxes, fmt=mon.BBoxFormat.VOC2COCO, imgsz=(h0, w0))
                for c, b, s in zip(d_labels, d_boxes_, d_scores):
                    debug_predictions.append({
                        "image_id"   : i,
                        "category_id": int(c),
                        "bbox"       : [
                            round(float(b[0]), 32),
                            round(float(b[1]), 32),
                            round(float(b[2]), 32),
                            round(float(b[3]), 32)
                        ],
                        "score"      : float(s),
                    })
                # Save image with bboxes
                image        = draw_bboxes(image, debug_outputs, classes)
                output_image = cv2.hconcat([image, output_image])
            
            pred_image_path = pred_image_dir / f"{i}.jpg"
            pred_image_path.parent.mkdir(parents=True, exist_ok=True)
            mon.image.save(output_image, str(pred_image_path))
            if args.save_video:
                if video_wrt is None:
                    video_file = pred_dir / f"{data_dir.stem}_pred.mp4"
                    video_fps  = camera_configs[str(data_dir.stem)]["max_fps"]
                    fourcc     = cv2.VideoWriter_fourcc(*"mp4v")
                    h2, w2     = mon.image.imgsz(output_image)
                    video_wrt  = cv2.VideoWriter(str(video_file), fourcc, video_fps, (w2, h2))
                video_wrt.write(cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
            
    timers.total.tock()
    
    # Write to JSON file
    with open(pred_json, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=None)
    mon.log(f"Enhanced images COCO results:")
    

# ----- Main -----
def main() -> str:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",   type=str, default="suwon#59_02_03", help="Path to image folder")
    parser.add_argument("--imgsz",      type=int, default=640,    help="Image size for preprocessing")
    parser.add_argument("--device",     type=str, default="cuda:1", help="Device to run the model on (e.g., 'cuda' or 'cpu')")
    parser.add_argument("--lle",        action="store_true")
    parser.add_argument("--early-stop", action="store_true", default=True)
    parser.add_argument("--debug",      action="store_true", default=True)
    parser.add_argument("--save-video", action="store_true", default=True)
    args = parser.parse_args()
    predict(args)


if __name__ == "__main__":
    main()

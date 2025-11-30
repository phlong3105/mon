#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json

import box
import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import mon
from mon import albumentations as A
from mon.cv import DEIM
from zlod import GCENet


current_file   = mon.Path(__file__).absolute()
current_dir    = current_file.parents[0]
project_dir    = current_file.parents[0]

gcenet_weights = project_dir / "zoo/gcenet.pt"
deim_config    = project_dir / "config/deim_dfine_s_coco80.yaml"
deim_weights   = project_dir / "zoo/deim_dfine_s_coco80.pth"
camera_configs = {
    "darkface"  : {
        "iters"     : 8,
        "conf_thres": 0.40,
        "max_fps"   : 1.0,
    },
    "ic_11_01"  : {
        "iters"     : 8,
        "conf_thres": 0.40,
        "max_fps"   : 30.0,
    },
    "lolistreet": {
        "iters"     : 8,
        "conf_thres": 0.35,
        "max_fps"   : 1.0,
    },
    "ydld"      : {
        "iters"     : 8,
        "conf_thres": 0.40,
        "max_fps"   : 1.0,
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
        image = mon.dtypes.draw_bbox(
            image     = image,
            bbox      = b,
            label     = f"{c}",
            color     = classes[c]["color"],
            thickness = 2,
            fill      = False,
        )
    return image


def measure_metric(input_json: mon.Path, target_json: mon.Path):
    if not input_json and not mon.Path(input_json).is_json_file(exist=True):
        raise FileNotFoundError(f"[input_json] does not exist: {input_json}.")
    if not target_json or not mon.Path(target_json).is_json_file(exist=True):
        raise FileNotFoundError(f"[target_json] does not exist: {target_json}.")

    coco_gt   = COCO(str(target_json))
    coco_dt   = coco_gt.loadRes(str(input_json))
    imgIds    = sorted(coco_gt.getImgIds())

    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.imgIds = imgIds
    # coco_eval.params.catIds = [1]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    results = {
        "AP"    : coco_eval.stats[0],
        "AP50"  : coco_eval.stats[1],
        "AP75"  : coco_eval.stats[2],
        # "APs"   : coco_eval.stats[3],
        # "APm"   : coco_eval.stats[4],
        # "APl"   : coco_eval.stats[5],
        # "AR@1"  : coco_eval.stats[6],
        # "AR@10" : coco_eval.stats[7],
        # "AR@100": coco_eval.stats[8],
        # "ARs"   : coco_eval.stats[9],
        # "ARm"   : coco_eval.stats[10],
        # "ARl"   : coco_eval.stats[11],
    }
    
    # Print results
    message = ""
    # Headers
    for m, v in results.items():
        if v:
            message += f"{f'{m}':<10}\t"
    message += "\n"
    # Values
    for i, (m, v) in enumerate(results.items()):
        if v:
            if i == len(results) - 1:
                message += f"{v:.10f}\n"
            else:
                message += f"{v:.10f}\t"
    mon.log(f"{message}\n")
    
    return results


# ----- Predict -----
@torch.no_grad()
def predict(args: dict | box.Box) -> str:
    # Parse input arguments
    data_dir       = project_dir / "data" / args.data_dir
    image_dir      = data_dir / "image"
    depth_dir      = data_dir / "depth"
    classes_file   = data_dir / "classes.yaml"
    gt_json        = data_dir / "gt.json"
    pred_dir       = data_dir / "pred"
    pred_image_dir = pred_dir / "image"
    pred_label_dir = pred_dir / "label"
    pred_json      = pred_dir / "pred.json"
    debug_json     = pred_dir / "debug.json"
    video_out      = None
    imgsz          = args.imgsz
    iters          = camera_configs[str(data_dir.stem)]["iters"]
    conf_thres     = camera_configs[str(data_dir.stem)]["conf_thres"]
    device         = torch.device(args.device) if args.device != "cpu" else "cpu"
    
    # Build models
    gcenet = GCENet(iters, weights=gcenet_weights)
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
    ])  #, additional_targets={"depth": "image"})
    # List images
    image_files = list(image_dir.rglob("*"))
    image_files = sorted([f for f in image_files if f.is_image_file()])
    # Read classes from YAML file
    classes = mon.rt.load_config(classes_file, verbose=False)
    classes = classes.get("classes", [])
    
    # Predict
    predictions       = []
    debug_predictions = []
    timers = mon.TimeProfiler()
    timers.total.tick()
    with mon.create_progress_bar() as pbar:
        for i, image_file in pbar.track(
            sequence    = enumerate(image_files),
            total       = len(image_files),
            description = f"[bright_yellow]Predicting"
        ):
            # Input
            image      = cv2.imread(str(image_file), cv2.IMREAD_COLOR)
            image      = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # depth      = None
            # depth_file = depth_dir / image_file.name
            # if depth_file.is_image_file(exist=True):
            #     depth = cv2.imread(str(depth_file), cv2.IMREAD_GRAYSCALE)
            #     depth = depth[:, :, None]
            if image is None:
                mon.log_error(f"Warning: Could not read image {image_file}. Skipping.")
                continue
            
            # Preprocess
            timers.preprocess.tick()
            h0, w0       = mon.image.imgsz(image)
            size0        = torch.tensor([[w0, h0]]).to(device)
            # augmented    = transform(image=image, depth=depth)
            augmented    = transform(image=image)
            image_tensor = augmented["image"].unsqueeze(0).to(device)  # Add batch dimension and to device
            # depth_tensor = augmented["depth"].unsqueeze(0).to(device)  # Add batch dimension and to device
            timers.preprocess.tock()
            
            # Infer
            timers.infer.tick()
            enhanced = gcenet(image_tensor, inference=True)["output"]
            # enhanced = enhanced[-1]
            outputs  = deim(enhanced, size0)
            if args.debug:
                debug_outputs = deim(image_tensor, size0)
            timers.infer.tock()
            
            # Postprocess
            timers.postprocess.tick()
            labels, boxes, scores = postprocess_outputs(outputs, 0.25)
            enhanced = mon.image.to_array(enhanced)
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
            pred_label_path = pred_label_dir / f"{image_file.stem}.txt"
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
            
            pred_image_path = pred_image_dir / image_file.name
            pred_image_path.parent.mkdir(parents=True, exist_ok=True)
            mon.image.save(output_image, str(pred_image_path))
            if args.save_video:
                if video_out is None:
                    video_file = pred_dir / f"{data_dir.stem}_pred.mp4"
                    video_file.parent.mkdir(parents=True, exist_ok=True)
                    video_fps  = camera_configs[str(data_dir.stem)]["max_fps"]
                    fourcc     = cv2.VideoWriter_fourcc(*"mp4v")
                    h2, w2     = mon.image.imgsz(output_image)
                    video_out  = cv2.VideoWriter(str(video_file), fourcc, video_fps, (w2, h2))
                video_out.write(cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
            
    timers.total.tock()
    
    # Write to JSON file
    with open(pred_json, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=None)
    mon.log(f"Enhanced images COCO results:")
    if args.debug:
        with open(debug_json, "w", encoding="utf-8") as f:
            json.dump(debug_predictions, f, indent=None)
        mon.log(f"Low-light images COCO results:")
     
    # Measure metrics if ground truth is available
    if gt_json.is_json_file(exist=True):
        measure_metric(input_json=pred_json, target_json=gt_json)
        if args.debug:
            measure_metric(input_json=debug_json, target_json=gt_json)
            
    # Finish
    # timers.print()


# ----- Main -----
def main() -> str:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",   type=str, default="ic_11_01", help="Path to image folder")
    parser.add_argument("--imgsz",      type=int, default=640,    help="Image size for preprocessing")
    parser.add_argument("--device",     type=str, default="cuda", help="Device to run the model on (e.g., 'cuda' or 'cpu')")
    parser.add_argument("--debug",      action="store_true", default=True)
    parser.add_argument("--save-video", action="store_true", default=True)
    args = parser.parse_args()
    predict(args)


if __name__ == "__main__":
    main()

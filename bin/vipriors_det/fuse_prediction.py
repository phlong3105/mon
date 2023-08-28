#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Fuse multiple prediction files togethers."""

from __future__ import annotations

import click
import cv2

import mon
from ensemble_boxes import *


# region Function

@click.command()
@click.option("--image-dir",  default=mon.DATA_DIR/"vipriors/delftbikes/test/images", type=click.Path(exists=True),  help="Image directory.")
@click.option("--input-dir",  default=mon.RUN_DIR/"predict/vipriors_det/prediction/", type=click.Path(exists=True),  help="Prediction directory.")
@click.option("--output-dir", default=mon.RUN_DIR/"predict/vipriors_det/submission/labels/", type=click.Path(exists=False), help="Submission directory.")
@click.option("--fuse",       default="wbf", type=click.Choice(["nms", "soft-nms", "nmw", "wbf"], case_sensitive=False), help="Fusion option.")
@click.option("--verbose",    is_flag=True)
def fuse_prediction(
    image_dir : mon.Path,
    input_dir : mon.Path,
    output_dir: mon.Path,
    fuse      : str,
    verbose   : bool,
):
    assert image_dir is not None and mon.Path(image_dir).is_dir()
    assert input_dir is not None and mon.Path(input_dir).is_dir()
    
    image_dir    = mon.Path(image_dir)
    input_dir    = mon.Path(input_dir)
    output_dir   = mon.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    subdir_names = [d.stem for d in input_dir.subdirs()]
    input_files  = []
    with mon.get_progress_bar() as pbar:
        for subdir in pbar.track(
            sequence    = subdir_names,
            description = f"[bright_yellow] Listing prediction files"
        ):
            pred_dir   = input_dir/subdir/"labels"
            pred_files = list(pred_dir.rglob("*"))
            pred_files = [f for f in pred_files if f.is_txt_file()]
            pred_files = sorted(pred_files)
            input_files.append(pred_files)
    
    with mon.get_progress_bar() as pbar:
        for i in pbar.track(
            sequence    = range(len(input_files[0])),
            total       = len(input_files[0]),
            description = f"[bright_yellow] Fusing predictions"
        ):
            image_file = image_dir / f"{input_files[0][i].stem}.jpg"
            assert image_file.is_image_file()
            image = cv2.imread(str(image_file))
            h, w, c = image.shape
            
            output_file_name = input_files[0][i].name
            output_file      = output_dir/output_file_name
            boxes_list       = []
            scores_list      = []
            labels_list      = []
            for j in range(len(input_files)):
                assert output_file_name == input_files[j][i].name
                with open(input_files[j][i], "r") as input_f:
                    lines  = input_f.read().splitlines()
                    lines  = [x.strip().split(" ") for x in lines]
                    boxes  = []
                    scores = []
                    labels = []
                    for l in lines:
                        l[0] = int(l[0])
                        l[3] = float(l[3]) * w
                        l[4] = float(l[4]) * h
                        l[1] = (float(l[1]) * w) - (l[3] / 2)
                        l[2] = (float(l[2]) * h) - (l[4] / 2)
                        l[5] = float(l[5])
                        x1   = l[1] / w
                        y1   = l[2] / h
                        x2   = (x1 + w) / w
                        y2   = (y1 + h) / h
                        boxes.append([x1, y1, x2, y2])
                        scores.append(l[5])
                        labels.append(l[0])
                boxes_list.append(boxes)
                scores_list.append(scores)
                labels_list.append(labels)
            
            iou_thr      = 0.5
            skip_box_thr = 0.0001
            sigma        = 0.1
            if fuse in ["nms"]:
                boxes, scores, labels = nms(
                    boxes   = boxes_list,
                    scores  = scores_list,
                    labels  = labels_list,
                    iou_thr = iou_thr,
                )
            elif fuse in ["soft-nms"]:
                boxes, scores, labels = soft_nms(
                    boxes   = boxes_list,
                    scores  = scores_list,
                    labels  = labels_list,
                    iou_thr = iou_thr,
                    sigma   = sigma,
                    thresh  = skip_box_thr,
                )
            elif fuse in ["nmw"]:
                boxes, scores, labels = non_maximum_weighted(
                    boxes_list   = boxes_list,
                    scores_list  = scores_list,
                    labels_list  = labels_list,
                    iou_thr      = iou_thr,
                    skip_box_thr = skip_box_thr,
                )
            else:
                boxes, scores, labels = weighted_boxes_fusion(
                    boxes_list   = boxes_list,
                    scores_list  = scores_list,
                    labels_list  = labels_list,
                    iou_thr      = iou_thr,
                    skip_box_thr = skip_box_thr,
                )
            
            with open(output_file, "w") as output_f:
                for i in range(len(boxes)):
                    b    = boxes[i]
                    b[0] = b[0] * w
                    b[1] = b[1] * h
                    b[2] = b[2] * w
                    b[3] = b[3] * h
                    b[2] = b[2] - b[0]
                    b[3] = b[3] - b[1]
                    b[0] = b[2] / 2
                    b[1] = b[3] / 2
                    b[0] = b[0] / w
                    b[1] = b[1] / h
                    b[2] = b[2] / w
                    b[3] = b[3] / h
                    s    = scores[i]
                    l    = int(labels[i])
                    output_f.write(
                        f"{l} {b[0]} {b[1]} {b[2]} {b[3]} {s}\n"
                    )
                
# endregion


# region Main

if __name__ == "__main__":
    fuse_prediction()

# endregion

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from ensemble_boxes import weighted_boxes_fusion
from munch import Munch

from one.core import progress_bar

CURRENT_DIR = Path(__file__).resolve().parent.absolute()
RUNS_DIR    = CURRENT_DIR / "runs"


# H1: - Functional -------------------------------------------------------------

def generate_submission(args: dict | Munch | argparse.Namespace):
    if isinstance(args, dict):
        args = Munch.fromDict(args)
    
    source      = Path(args.source)
    output_file = Path(args.output_file)
    output_data = []
    
    file_list   = list(sorted(os.listdir(os.path.join(source, "labels"))))
    file_list   = [source / "labels" / f for f in file_list]
    # file_list   = list(source.rglob("*.txt"))
    # file_list   = sorted(file_list)
    
    with progress_bar() as pbar:
        for i in pbar.track(
            range(len(file_list)),
            description=f"[bright_yellow]Processing files"
        ):
            path  = file_list[i]
            lines = open(path, "r").read().splitlines()
            data  = []
            for l in lines:
                d = l.split(" ")
                data.append(
                    {
                        "image_id"   : i,
                        "category_id": int(d[0]) + 1,
                        "bbox"       : [float(d[1]),
                                        float(d[2]),
                                        float(d[3]) - float(d[1]),
                                        float(d[4]) - float(d[2])],
                        "score"      : float(d[5])
                    }
                )
            data = sorted(data, key=lambda x: x["score"], reverse=True)

            output_data.extend(data)
            
    with open(output_file, "w") as f:
        json.dump(output_data, f)
    

def generate_ensemble_submission(args: dict | Munch | argparse.Namespace):
    if isinstance(args, dict):
        args = Munch.fromDict(args)
    
    output_file = Path(args.output_file)
    output_data = []
    dirs        = [
        # RUNS_DIR / "detect" / "yolov7-d6-delftbikes-multiscale-01",
        # RUNS_DIR / "detect" / "yolov7-d6-delftbikes-multiscale-02",
        # RUNS_DIR / "detect" / "yolov7-d6-delftbikes-multiscale-03",
        # RUNS_DIR / "detect" / "yolov7-d6-delftbikes-multiscale-04",
        # 
        # RUNS_DIR / "detect" / "yolov7-e6-delftbikes-multiscale-01",
        # RUNS_DIR / "detect" / "yolov7-e6-delftbikes-multiscale-02",
        # RUNS_DIR / "detect" / "yolov7-e6-delftbikes-multiscale-03",
        # RUNS_DIR / "detect" / "yolov7-e6-delftbikes-multiscale-04",
        # 
        # RUNS_DIR / "detect" / "yolov7-e6e-delftbikes-multiscale-01",
        # RUNS_DIR / "detect" / "yolov7-e6e-delftbikes-multiscale-02",
        # RUNS_DIR / "detect" / "yolov7-e6e-delftbikes-multiscale-03",
        # RUNS_DIR / "detect" / "yolov7-e6e-delftbikes-multiscale-04",
        # 
        # RUNS_DIR / "detect" / "yolov7-w6-delftbikes-multiscale-01",
        # RUNS_DIR / "detect" / "yolov7-w6-delftbikes-multiscale-02",
        # RUNS_DIR / "detect" / "yolov7-w6-delftbikes-multiscale-03",
        # RUNS_DIR / "detect" / "yolov7-w6-delftbikes-multiscale-04",
        
        RUNS_DIR / "detect" / "yolov7-d6-delftbikes-multiscale",
        RUNS_DIR / "detect" / "yolov7-e6-delftbikes-multiscale",
        RUNS_DIR / "detect" / "yolov7-e6e-delftbikes-multiscale",
        RUNS_DIR / "detect" / "yolov7-w6-delftbikes-multiscale",
    ]
    
    txts = [list(d.rglob("labels/*.txt")) for d in dirs]
    n    = len(txts[0])
    if not all(n == len(t) for t in txts):
        raise ValueError(f"Number of .txt files among folders does not match.")
    
    with progress_bar() as pbar:
        for i in pbar.track(
            range(n),
            description=f"[bright_yellow]Ensembling files"
        ):
            boxes_list  = []
            scores_list = []
            labels_list = []
            h, w, c     = 0, 0, 0
            for j in range(len(txts)):
                boxes  = []
                scores = []
                labels = []
                p      = txts[j][i]
                lines  = open(p, "r").read().splitlines()
                for l in lines:
                    d       = l.split(" ")
                    h, w, c = int(d[6]), int(d[7]), int(d[8])
                    x1 = float(d[1]) / w
                    y1 = float(d[2]) / h
                    x2 = float(d[3]) / w
                    y2 = float(d[4]) / h
                    boxes.append([x1, y1, x2, y2])
                    scores.append(float(d[5]))
                    labels.append(int(d[0]))
                
                boxes_list.append(boxes)
                scores_list.append(scores)
                labels_list.append(labels)
    
            weights      = [1, 1, 1, 1]
            iou_thr      = 0.5
            skip_box_thr = 0.001
            sigma        = 0.1
            boxes, scores, labels = weighted_boxes_fusion(
                boxes_list   = boxes_list,
                scores_list  = scores_list,
                labels_list  = labels_list,
                weights      = weights,
                iou_thr      = iou_thr,
                skip_box_thr = skip_box_thr,
            )
            
            data = []
            for k in range(len(boxes)):
                b    = boxes[k]
                b[0] = b[0] * w
                b[1] = b[1] * h
                b[2] = b[2] * w
                b[3] = b[3] * h
                data.append(
                    {
                        "image_id": i,
                        "category_id": int(labels[k] + 1),
                        "bbox": [
                            b[0],
                            b[1],
                            b[2] - b[0],
                            b[3] - b[1]
                        ],
                        "score": float(scores[k])
                    }
                )
            data = sorted(data, key=lambda x: x["score"], reverse=True)
            output_data.extend(data)
         
    with open(output_file, "w") as f:
        json.dump(output_data, f)


# H1: - Main -------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",         default="generate_submission",                               type=str)
    parser.add_argument("--source",      default=CURRENT_DIR/"runs"/"detect"/"yolov7-delftbikes-multiscale2", type=str, help="Directory containing YOLO results .txt")
    parser.add_argument("--output-file", default=CURRENT_DIR/"runs"/"detect"/"submission.json",       type=str, help="Submission .json file")
    args   = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.run == "generate_submission":
        generate_submission(args)
    if args.run == "generate_ensemble_submission":
        generate_ensemble_submission(args)

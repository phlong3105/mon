#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DelftBikes datasets and datamodules.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import torch

from one.constants import DATA_DIR
from one.core import create_dirs
from one.core import Path_
from one.core import progress_bar
from one.data import load_from_file
from one.vision.shape import box_xyxy_to_cxcywh_norm


# H1: - Functional -------------------------------------------------------------

def generate_train_val(
    root       : Path_,
    json_file  : Path_,
    val_size   : int,
    yolo_labels: bool = True
):
    root       = Path(root)
    json_file  = Path(json_file)
    src_path   = root / "trainval" / "images"
    train_path = root / "train"    / "images"
    val_path   = root / "val"      / "images"
    img_list   = os.listdir(src_path)
    val_list   = img_list[:val_size]
    train_list = img_list[val_size:]
    json_data  = load_from_file(open(root / "trainval" / json_file))
    
    # Val set generation
    create_dirs(paths=[val_path])
    for img in val_list:
        shutil.copy(os.path.join(src_path, img), val_path)
    
    valset_list = os.listdir(val_path)
    if len(valset_list) == 1000:
        print("Val images are successfully generated.")
    
    val_dict = {}
    for im_name in valset_list:
        val_dict[im_name] = json_data[im_name]
    
    with open(os.path.join(root, "val", "val_annotations.json"), "w") as outfile:
        json.dump(val_dict, outfile)
    
    if yolo_labels:
        generate_yolo_labels(
            root      = os.path.join(root, "val"),
            json_file = "val_annotations.json"
        )
    print("Val labels are successfully generated.")
    
    # Train set generation
    create_dirs(paths=[train_path])
    for img in train_list:
        shutil.copy(os.path.join(src_path, img), train_path)
        
    trainset_list = os.listdir(train_path)
    if len(trainset_list) == 7000:
        print("Train images are successfully generated.")
        
    train_dict = {}
    for im_name in train_list:
        train_dict[im_name] = json_data[im_name]
    
    with open(os.path.join(root, "train", "train_annotations.json"), "w") as outfile:
        json.dump(train_dict, outfile)
    
    if yolo_labels:
        generate_yolo_labels(
            root      = os.path.join(root, "train"),
            json_file = "train_annotations.json"
        )
    print("Train labels are successfully generated.")


def generate_yolo_labels(root: Path_, json_file: Path_):
    root        = Path(root)
    json_file   = Path(json_file)
    # images    = list(sorted(os.listdir(os.path.join(root, "images"))))
    json_data   = load_from_file(root / json_file)
    labels_path = root / "yolo_labels"
    
    create_dirs(paths=[labels_path])

    with progress_bar () as pbar:
        for k, v in pbar.track(
            json_data.items(), description=f"[red]Processing labels"
        ):
            image_path = root / "images" / k
            image      = v["image"]
            channels   = image["channels"]
            height     = image["height"]
            width      = image["width"]
    
            ids   = []
            boxes = []
            names = []
            confs = []
            for idx, i in enumerate(v["parts"], 0):
                label = v["parts"][i]
                if label["object_state"] != "absent":
                    loc = label["absolute_bounding_box"]
                    x1  = loc["left"]
                    x2  = loc["left"] + loc["width"]
                    y1  = loc["top"]
                    y2  = loc["top"] + loc["height"]
                    boxes.append(torch.FloatTensor([x1, y1, x2, y2]))
                    ids.append(idx)
                    names.append(label["part_name"])
                    confs.append(label["trust"])
            
            boxes = torch.stack(boxes, 0)
            boxes = box_xyxy_to_cxcywh_norm(boxes, height, width)
            
            yolo_file = labels_path / k.replace(".jpg", ".txt")
            with open(yolo_file, mode="w") as f:
                for i, b in enumerate(boxes):
                    b = b.numpy()
                    f.write(f"{ids[i]} {b[0]} {b[1]} {b[2]} {b[3]} {confs[i]}\n")
        

# H1: - Test -------------------------------------------------------------------

def parse_args(cfg):
    pass


# H1: - Main -------------------------------------------------------------------

if __name__ == "__main__":
    """
    generate_train_val(
        root        = DATA_DIR / "delftbikes",
        json_file   = "trainval_annotations.json",
        val_size    = 1000,
        yolo_labels = True
    )
    """
    generate_yolo_labels(
        root      = DATA_DIR / "delftbikes" / "train",
        json_file = "train_annotations.json"
    )

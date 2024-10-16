#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This script resize images."""

from __future__ import annotations

import cv2

import mon


def arrange_image(split: str = "test"):
    root_dir    = mon.DATA_DIR / f"enhance/llie/Sony/Sony"
    input_dir   = mon.DATA_DIR / f"enhance/llie/Sony/{split}/image"
    target_dir  = mon.DATA_DIR / f"enhance/llie/Sony/{split}/ref"
    text_file   = mon.DATA_DIR / f"enhance/llie/Sony/Sony_{split}_list.txt"
    with open(str(text_file), "r") as in_file:
        l = in_file.read().splitlines()
        l = [x.strip().split(" ") for x in l]
    
    with mon.get_progress_bar() as pbar:
        for i in pbar.track(
            sequence    = range(len(l)),
            total       = len(l),
            description = f"[bright_yellow] Converting"
        ):
            image_path = mon.Path(l[i][0])
            ref_path   = mon.Path(l[i][1])
            image_path = root_dir / f"short" / image_path.name
            ref_path   = root_dir / f"long"  / ref_path.name
            # Read image
            image      = mon.read_image(image_path)
            ref        = mon.read_image(ref_path)
            image      = mon.resize(image, 512, 32, "short")
            ref        = mon.resize(ref,   512, 32, "short")
            image      = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            ref        = cv2.cvtColor(ref,   cv2.COLOR_RGB2BGR)
            # Save image
            new_image_path = input_dir  / f"{image_path.stem}.jpg"
            new_ref_path   = target_dir / f"{image_path.stem}.jpg"
            new_image_path.parent.mkdir(parents=True, exist_ok=True)
            new_ref_path.parent.mkdir(parents=True,   exist_ok=True)
            cv2.imwrite(str(new_image_path), image)
            cv2.imwrite(str(new_ref_path),   ref)


arrange_image("val")

#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Reference: https://y-t-g.github.io/tutorials/bg-images-for-yolo/

from __future__ import annotations

import os

import click
import requests
from pycocotools.coco import COCO

import mon


# region Function

@click.command(name="main", context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.option("--save-dir",   type=click.Path(exists=False), default=mon.DATA_DIR / "aic/aic24-fisheye8k/coco", help="Output .json file.")
@click.option("--num_images", type=int, default=800,  help="Number of downloading images.")
@click.option("--classes" ,   type=str, default="bus, bicycle, car, motorcycle, person, truck, train", help="Detecting classes.")
@click.option("--verbose",    is_flag=True)
def gen_bg_img(
    save_dir  : str | mon.Path,
    num_images: int,
    classes   : str,
    verbose   : bool = True,
):
    # assert save_dir is not None and mon.Path(save_dir).is_dir()
    save_dir = mon.Path(save_dir)
    mon.delete_dir(save_dir)
    save_dir.parent.mkdir(parents=True, exist_ok=True)
    
    assert classes not in [None, ""]
    classes = classes.split(",")

    coco = COCO("instances_train2017.json")
    
    # Specify a list of classes to exclude.
    # Background images will not contain these.
    # These should be classes included in training.
    exc_cat_ids = coco.getCatIds(catNms=classes)
    
    # Get the corresponding image ids and images using loadImgs
    exc_img_ids = coco.getImgIds(catIds=exc_cat_ids)
    
    # Get all image ids
    all_cat_ids = coco.getCatIds(catNms=[""])
    all_img_ids = coco.getImgIds(catIds=all_cat_ids)
    
    # Remove img ids of classes that are included in training
    bg_img_ids  = set(all_img_ids) - set(exc_img_ids)
    
    # Get background image metadata
    bg_images   = coco.loadImgs(bg_img_ids)
    
    # Create dirs
    os.makedirs(str(save_dir / "images"), exist_ok=True)
    os.makedirs(str(save_dir / "labels"), exist_ok=True)
    
    # Save the images into a local folder
    with mon.get_progress_bar() as pbar:
        for im in pbar.track(
            sequence    = bg_images[:num_images],
            total       = num_images,
            description = f"[bright_yellow] Downloading background images"
        ):
            img_data = requests.get(im["coco_url"]).content
            # Save the image
            with open(str(save_dir / "images" / im["file_name"]), "wb") as handler:
                handler.write(img_data)
            # Save the corresponding blank label txt file
            with open(str(save_dir / "labels" / (im["file_name"][:-3] + "txt")), "wb") as handler:
                pass

# endregion


# region Main

if __name__ == "__main__":
    gen_bg_img()

# endregion

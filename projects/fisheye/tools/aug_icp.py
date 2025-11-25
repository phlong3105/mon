#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Apply Inplace Copy-Paste augmentation for static cameras (ICP) on a dataset."""

import mon
from fisheye import icp

current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]
root_dir     = current_file.parents[1]
if root_dir.has_subdir("data"):
    data_dir = root_dir / "data"
else:
    data_dir = root_dir


# ----- Convert -----
def apply_icp_all(data: str, runs: int = 1):
    data_root = data_dir / data
    for subdir in sorted(data_root.subdirs()):
        image_dir = data_dir / data / subdir.stem / "image"
        label_dir = data_dir / data / subdir.stem / "label"

        augment   = icp.ICPAugmentation(
            image_dir      = image_dir,
            label_dir      = label_dir,
            num_classes    = 5,
            sam_model      = "sam2.1_l.pt",
            ratio          = 1,
            iou_thres      = 0.00001,
            max_tries      = 100,
            style_transfer = False,
            harmonization  = True,
        )
        for _ in range(runs):
            augment.process()


def apply_icp_one(data: str, subdir: str, runs: int = 1):
    image_dir = data_dir / data / subdir / "image"
    label_dir = data_dir / data / subdir / "label"
    
    augment   = icp.ICPAugmentation(
        image_dir      = image_dir,
        label_dir      = label_dir,
        num_classes    = 5,
        sam_model      = "sam2.1_l.pt",
        ratio          = 1,
        iou_thres      = 0.00001,
        max_tries      = 100,
        style_transfer = False,
        harmonization  = False,
        shadow         = False,
    )
    for _ in range(runs):
        augment.process()


# ----- Main -----
if __name__ == "__main__":
    # Utils
    # icp.group_image_and_label_files("fisheye8k/train")
    # icp.concat_image_and_label_files("fisheye8k/train/groups")
    
    # Augment
    # apply_icp_all("fisheye8k/train/groups", 1)
    # apply_icp_one("fisheye8k/train/groups", "camera1_A", 1)
    apply_icp_one("aicp", "camera17_A", 1)

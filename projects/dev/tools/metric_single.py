#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Metric Evaluation Script.

This script provides a simple interface for measuring IQA metrics.
"""

from __future__ import annotations

__all__ = []

import argparse
import sys

import cv2
import pyiqa
import torch

from mon.core import create_progress_bar, log, log_error, Path
from mon.dataset import transform as T

current_file = Path(__file__).normalize()
current_dir = current_file.parents[0]


# ==============================================================================
# region MAIN
# ==============================================================================

def main(args: argparse.Namespace):
    # Validate inputs
    data_dir = Path(args.data_dir).normalize()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Define and read target file
    device = torch.device("cuda:0")
    excluded_filenames = ["image.jpg", "target.jpg", "depth.jpg"]
    target_file = data_dir / "target.jpg"
    target = cv2.imread(str(target_file))

    # Define metrics
    all_metrics = pyiqa.default_model_configs.DEFAULT_CONFIGS
    metrics = ["psnr", "ssim", "ssimc", "lpips", "niqe", "pi"]
    metrics_func = {}
    for i, m in enumerate(metrics):
        if m in all_metrics:
            metrics_func[m] = pyiqa.create_metric(
                metric_name=m, as_loss=False, device=device,
            )
        else:
            log_error(f"Unsupported metric: {m}. Skipping...")

    # Define transforms
    transforms = T.Compose([
        T.Normalize(normalization="min_max"),
        T.ToTensorV2(transpose_mask=True),
    ], additional_targets={"target": "image"})
    if args.resize:
        h = w = args.imgsz
        transforms = T.Resize(height=h, width=w) + transforms

    # Loop through all predicted images in the data directory
    image_files = sorted(list(data_dir.rglob(f"*.jpg")))
    image_files = [i for i in image_files if i.name not in excluded_filenames]
    with create_progress_bar() as pbar:
        for i, image_file in pbar.track(
            sequence=enumerate(image_files),
            total=len(image_files),
            description=f"[bright_yellow]Measuring",
        ):
            # Read image file
            image = cv2.imread(str(image_file))

            # Transform image and target
            transformed = transforms(image=image, target=target)
            image_t = transformed["image"].unsqueeze(0).to(device)
            target_t = transformed["target"].unsqueeze(0).to(device)

            # Sometimes image and target may have different orientations
            # (H, W) vs (W, H). We check the image and target sizes and
            # transpose the image if needed.
            image_sz = image_t.shape[-2:]
            target_sz = target_t.shape[-2:]
            if image_sz[0] == target_sz[1]:
                image_t = image_t.transpose(2, 3)

            # Measure metric
            results = {}
            for m in metrics:
                if all_metrics[m]["metric_mode"] == "FR":
                    results[m] = metrics_func[m](image_t, target_t)
                else:
                    results[m] = metrics_func[m](image_t)

            # Log results
            message = f"{image_file.stem:24s} "
            for k, v in results.items():
                message += f"| {k}: {v.item():06.4f} "
            log(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("main")
    parser.add_argument("--data-dir", type=str, default="/home/longpham/10_workspace/11_code/mon/projects/dev/run/assets/sice_112/")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--resize", action="store_true")
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return args


if __name__ == "__main__":
    main(parse_args())

# endregion

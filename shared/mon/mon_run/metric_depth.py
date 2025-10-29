#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Measures depth metrics for a given model and dataset."""

import argparse
import logging

import albumentations as A
import cv2
import matplotlib
import numpy as np

import mon

mon.disable_print()

current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]
cmap         = matplotlib.colormaps.get_cmap("Spectral_r")
_METRICS     = ["abs_rel", "sq_rel", "rmse", "rmse_log", "mae", "delta1", "delta2", "delta3"]


# ----- Metrics -----
def compute_metrics(
    pred      : np.ndarray,
    target    : np.ndarray,
    valid_mask: np.ndarray = None,
    normalize : bool       = True
) -> dict:
    # Input validation
    if pred.shape != target.shape:
        raise ValueError("Predicted and target depth maps must have the same shape.")

    pred   =   pred.astype(np.float32)
    target = target.astype(np.float32)

    # Create valid_mask if not provided
    if valid_mask is None:
        valid_mask = (target > 0) & (~np.isnan(pred)) & (~np.isnan(target))

    if not np.any(valid_mask):
        raise ValueError("No valid pixels found in the depth maps.")

    # Normalize predicted depth map to target's range
    if normalize:
        valid_target           = target[valid_mask]
        target_min, target_max = np.min(valid_target), np.max(valid_target)

        # Avoid division by zero in normalization
        if target_max == target_min:
            raise ValueError("Target depth map has no range (min equals max).")

        # Min-max normalization of predicted depths
        valid_pred         = pred[valid_mask]
        pred_min, pred_max = np.min(valid_pred), np.max(valid_pred)
        if pred_max != pred_min:  # Only normalize if predicted map has a range
            pred = target_min + (target_max - target_min) * (pred - pred_min) / (pred_max - pred_min)
        else:
            # If predicted map is constant, scale to target's mean or min
            pred = np.full_like(pred, target_min)

        # Update min/max of normalized predicted map
        valid_pred         = pred[valid_mask]
        pred_min, pred_max = np.min(valid_pred), np.max(valid_pred)

    # Flatten arrays and apply mask
    pred_flat   = pred[valid_mask]
    target_flat = target[valid_mask]

    # Compute differences
    diff       = pred_flat - target_flat
    abs_diff   = np.abs(diff)
    # Absolute Relative Error
    abs_rel    = np.mean(abs_diff / target_flat)
    # Squared Relative Error
    sq_rel     = np.mean((diff ** 2) / target_flat)
    # RMSE
    rmse       = np.sqrt(np.mean(diff ** 2))
    # RMSE log
    log_pred   = np.log(np.clip(pred_flat,   1e-10, None))  # Avoid log(0)
    log_target = np.log(np.clip(target_flat, 1e-10, None))
    rmse_log   = np.sqrt(np.mean((log_pred - log_target) ** 2))
    # MAE
    mae        = np.mean(abs_diff)
    # Threshold accuracies
    thresh     = np.maximum(pred_flat / target_flat, target_flat / pred_flat)
    delta1     = np.mean(thresh < 1.25)
    delta2     = np.mean(thresh < 1.25 ** 2)
    delta3     = np.mean(thresh < 1.25 ** 3)

    return {
        "abs_rel" : abs_rel,
        "sq_rel"  : sq_rel,
        "rmse"    : rmse,
        "rmse_log": rmse_log,
        "mae"     : mae,
        "delta1"  : delta1,
        "delta2"  : delta2,
        "delta3"  : delta3
    }


def measure_depth_metrics(
    input_dir : mon.Path,
    target_dir: mon.Path,
    model     : str,
    data      : str,
    imgsz     : int,
    resize    : bool,
    normalize : bool,
    use_color : bool,
    verbose   : bool,
) -> dict:
    # Parse input and target directories
    input_dir  = mon.Path(input_dir)
    target_dir = mon.Path(target_dir)

    # List image files
    image_files = list(input_dir.rglob("*"))
    image_files = [f for f in image_files if f.is_image_file()]
    image_files = sorted(image_files)
    num_items   = 0
    
    # Parse arguments
    metric      = [m.lower() for m in _METRICS]
    values      = {m: []     for m in metric}
    results     = {}
    h, w        = mon.image.imgsz(imgsz)
    
    # Prepare transform
    base_transform = A.Compose([])
    
    # Measuring
    description = f"[bright_yellow]Measuring {model} | {data}"
    with mon.create_progress_bar(transient=not verbose) as pbar:
        for image_file in pbar.track(
            sequence    = image_files,
            total       = len(image_files),
            description = description
        ):
            # Image
            image  = mon.image.load_image(path=image_file, flags=cv2.IMREAD_COLOR)
            h0, w0 = mon.image.imgsz(image)
            h2, w2 = h, w
            if use_color:
                image = (cmap(image)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

            # Target
            target      = None
            target_file = None
            need_resize = resize
            for ext in mon.ImageExtension.values():
                temp = target_dir / f"{image_file.stem}{ext}"
                if temp.exists():
                    target_file = temp
            if target_file and target_file.exists():  # Has target file
                target = mon.image.load_image(path=target_file, flags=cv2.IMREAD_COLOR)
                h1, w1 = mon.image.imgsz(target)
                if h1 != h0 or w1 != w0:  # Mismatch size between image and target
                    h2, w2      = h1, w1
                    need_resize = True
                if use_color:
                    target = (cmap(target)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
            else:
                raise FileNotFoundError(f"``target_file`` does not exist: {target_file}.")
            
            # Transform
            if need_resize:
                transform = A.Compose([
                    A.Resize(height=h2, width=w2)
                ], additional_targets={"target": "image"})
                augmented = transform(image=image, target=target)
                image     = augmented["image"]
                target    = augmented["target"]
            
            # Measure metric
            measured_results = compute_metrics(image, target, normalize=normalize)
            for k, v in measured_results.items():
                if k in values:
                    values[k].append(v)

            num_items += 1

    for m, v in values.items():
        if len(v) > 0:
            results[m] = float(sum(v) / num_items)
        else:
            results[m] = None
    return results


# ----- Main -----
def main(
    input_dir  : mon.Path,
    target_dir : mon.Path,
    result_file: mon.Path,
    arch       : str,
    model      : str,
    data       : str,
    device     : int | list[int] | str,
    imgsz      : int,
    resize     : bool,
    normalize  : bool,
    use_color  : bool,
    save_txt   : bool,
    verbose    : bool,
):
    if not verbose:
        logger = logging.getLogger()
        logger.disabled = True
    mon.console.rule(f"[bold red] {model}")
    mon.console.log(f"[bold green]Model: {model}")
    mon.console.log(f"[bold red]Data : {data}")
    mon.console.log(f"[bold]Device: {device}")

    input_dir  = mon.Path(input_dir)  if input_dir  else None
    target_dir = mon.Path(target_dir) if target_dir else None

    if not input_dir or not input_dir.is_dir():
        raise ValueError(f"``input_dir`` does not exist: {input_dir}.")

    results = measure_depth_metrics(
        input_dir  = input_dir,
        target_dir = target_dir,
        model      = model,
        data       = data,
        imgsz      = imgsz,
        resize     = resize,
        normalize  = normalize,
        use_color  = use_color,
        verbose    = verbose,
    )

    # Show results
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
    print(f"{message}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="metric")
    parser.add_argument("--input-dir",   type=str, help="Input image directory.")
    parser.add_argument("--target-dir",  type=str, help="Ground-truth image directory.")
    parser.add_argument("--result-file", type=str, help="Result file.")
    parser.add_argument("--arch",        type=str, help="Model's architecture.")
    parser.add_argument("--model",       type=str, help="Model's fullname.")
    parser.add_argument("--data",        type=str, help="Source data name.")
    parser.add_argument("--device",      type=str, help="Running devices.")
    parser.add_argument("--imgsz",       type=int, default=512)
    parser.add_argument("--resize",      action="store_true")
    parser.add_argument("--normalize",   action="store_true")
    parser.add_argument("--use-color",   action="store_true")
    parser.add_argument("--save-txt",    action="store_true")
    parser.add_argument("--verbose",     action="store_true")
    args = parser.parse_args()
    main(**vars(args))

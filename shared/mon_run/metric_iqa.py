#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Measures image quality assessment metrics for a given model and dataset."""

import argparse
import logging

import albumentations as A
import pyiqa
import pyiqa.default_model_configs
import pyiqa.models.inference_model
import torch

import mon

mon.disable_print()

current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]
_METRICS     = pyiqa.default_model_configs.DEFAULT_CONFIGS


# ----- PyIQA -----
def measure_metric_pyiqa(
    input_dir  : mon.Path,
    target_dir : mon.Path,
    arch       : str,
    model      : str,
    data       : str,
    device     : int | list[int] | str,
    imgsz      : int,
    resize     : bool,
    metric     : list[str],
    use_gt_mean: bool,
    verbose    : bool,
) -> dict:
    """Measure metrics using :mod:`pyiqa` package."""
    # Parse input and target directories
    input_dir  = mon.Path(input_dir)
    target_dir = mon.Path(target_dir) if target_dir else input_dir.replace("image", "ref")

    # Parse arguments
    device  = device[0] if len(device) == 1 else device
    device  = torch.device(("cpu" if not torch.cuda.is_available() else device))
    metric  = list(_METRICS.names()) if ("all" in metric or "*" in metric) else metric
    metric  = [m.lower() for m in metric]
    values  = {m: []     for m in metric}
    results = {}
    h, w    = mon.image.imgsz(imgsz)
    
    # Parse metrics
    metric_f = {}
    for i, m in enumerate(metric):
        if m in _METRICS:
            metric_f[m] = pyiqa.create_metric(metric_name=m, as_loss=False, device=device)
        
    # Data I/O
    transform = A.Compose([
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], normalization="min_max"),
        A.ToTensorV2(transpose_mask=True),
    ])
    if resize:
        transform = A.Resize(height=h, width=w) + transform
    dataloader = mon.data.DataLoader(
        dataset = mon.data.ImageEvalDataset(
            input_dir  = input_dir,
            target_dir = target_dir,
            transform  = transform,
            verbose    = verbose,
        ), batch_size = 1
    )
    
    # Measuring
    description = f"[bright_yellow]Measuring {model} | {data} (GT Mean)" if use_gt_mean else f"[bright_yellow]Measuring {model} | {data}"
    with mon.create_progress_bar(transient=not verbose) as pbar:
        for i, datapoint in pbar.track(
            sequence    = enumerate(dataloader),
            total       = len(dataloader),
            description = description
        ):
            image  = datapoint["image"]
            target = datapoint.get("target", None)
            if target in [ [], [None], None ]:
                target = None
            elif image.shape != target.shape:
                image = image.permute(0, 1, 3, 2)
            
            # Move to device
            image  =  image.to(device=device)
            target = target.to(device=device) if target is not None else None
            
            # Measure metric
            for m in metric:
                if target is None and _METRICS[m]["metric_mode"] == "FR":
                    continue
                elif target is not None and _METRICS[m]["metric_mode"] == "FR":
                    values[m].append(metric_f[m](image, target))
                else:
                    values[m].append(metric_f[m](image))

    for m, v in values.items():
        if len(v) > 0:
            results[m] = float(sum(v) / len(v))
        else:
            results[m] = None
    return results


def update_best_results(results: dict, new_values: dict) -> dict:
    for m, v in new_values.items():
        if m in _METRICS:
            lower_better = _METRICS[m].get("lower_better", False)
            if m not in results:
                results[m] = v
            elif results[m] is None:
                results[m] = v
            elif v:
                results[m] = min(results[m], v) if lower_better else max(results[m], v)
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
    metric     : list[str],
    use_gt_mean: bool,
    backend    : list[str],
    save_txt   : bool,
    verbose    : bool,
):
    if not verbose:
        logger = logging.getLogger()
        logger.disabled = True
    mon.console.rule(f"[bold red] {model}")
    mon.console.log(f"[bold green]Model : {model}")
    mon.console.log(f"[bold red]Data  : {data}")
    mon.console.log(f"[bold]Device: {device}")

    input_dir  = mon.Path(input_dir)  if input_dir  else None
    target_dir = mon.Path(target_dir) if target_dir else None

    if not input_dir or not input_dir.is_dir():
        raise ValueError(f"``input_dir`` does not exist: {input_dir}.")

    results         = {}
    results_gt_mean = {}
    backend         = [backend] if not isinstance(backend, list) else backend
    backend         = [str(b).lower() for b in backend]

    for b in backend:
        if b in ["pyiqa"]:
            metric_values = measure_metric_pyiqa(
                input_dir   = input_dir,
                target_dir  = target_dir,
                arch        = arch,
                model       = model,
                data        = data,
                device      = device,
                imgsz       = imgsz,
                resize      = resize,
                metric      = metric,
                use_gt_mean = False,
                verbose     = verbose,
            )
            results = update_best_results(results, metric_values)
            if use_gt_mean:
                metric_values_gt_mean = measure_metric_pyiqa(
                    input_dir   = input_dir,
                    target_dir  = target_dir,
                    arch        = arch,
                    model       = model,
                    data        = data,
                    device      = device,
                    imgsz       = imgsz,
                    resize      = resize,
                    metric      = metric,
                    use_gt_mean = True,
                    verbose     = verbose,
                )
                results_gt_mean = update_best_results(results_gt_mean, metric_values_gt_mean)
        else:
            mon.console.log(f"`{backend}` is not supported!")
    
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
    for i, (m, v) in enumerate(results_gt_mean.items()):
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
    parser.add_argument("--metric",      type=str, action="append", help="Measuring metric.")
    parser.add_argument("--use-gt-mean", action="store_true")
    parser.add_argument("--backend",     choices=["pyiqa"], default="pyiqa")
    parser.add_argument("--save-txt",    action="store_true")
    parser.add_argument("--verbose",     action="store_true")
    args = parser.parse_args()
    main(**vars(args))

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Metric Pipeline.

This script is used to measure metrics for a given model and dataset.
"""

from __future__ import annotations

import logging

import click
import pyiqa
import torch

import mon

console  = mon.console
_METRICS = pyiqa.DEFAULT_CONFIGS


# region Metric

def measure_metric_pyiqa(
    input_dir  : mon.Path,
    target_dir : mon.Path | None,
    result_file: mon.Path | str,
    arch       : str,
    model      : str,
    data       : str,
    imgsz      : int,
    device     : int | list[int] | str,
    resize     : bool,
    metric     : list[str],
    use_gt_mean: bool,
    save_txt   : bool,
    verbose    : bool,
) -> dict:
    """Measure metrics using :mod:`pyiqa` package."""
    assert input_dir and mon.Path(input_dir).is_dir()
    # if target_dir:
    #     assert mon.Path(target_dir).is_dir()
    if result_file:
        assert (mon.Path(result_file).is_dir()
                or mon.Path(result_file).is_file()
                or isinstance(result_file, str))
        result_file = mon.Path(result_file)
    
    # Parse input and target directories
    input_dir  = mon.Path(input_dir)
    target_dir = mon.Path(target_dir) if target_dir else input_dir.replace("image", "ref")
    
    # Parse result file
    result_file = mon.Path(result_file) if result_file else None
    if save_txt and result_file and result_file.is_dir():
        result_file /= "metric.txt"
        result_file.parent.mkdir(parents=True, exist_ok=True)
    
    # List image files
    image_files = list(input_dir.rglob("*"))
    image_files = [f for f in image_files if f.is_image_file()]
    image_files = sorted(image_files)
    num_items   = len(image_files)
    
    # Parse arguments
    device      = device[0] if len(device) == 1 else device
    device      = torch.device(("cpu" if not torch.cuda.is_available() else device))
    metric      = list(_METRICS.keys()) if ("all" in metric or "*" in metric) else metric
    metric      = [m.lower() for m in metric]
    values      = {m: []     for m in metric}
    results     = {}
    h, w        = mon.get_image_size(imgsz)
    
    # Parse metrics
    metric_f    = {}
    mon.disable_print()
    for i, m in enumerate(metric):
        if m not in _METRICS:
            continue
        metric_f[m] = pyiqa.InferenceModel(
            metric_name = m,
            as_loss     = False,
            device      = device,
        )
    mon.enable_print()
    need_target = any(_METRICS[m]["metric_mode"] == "FR" for m in metric)
    
    # Measuring
    description = f"[bright_yellow] Measuring {model} {data} (GT Mean)" if use_gt_mean else f"[bright_yellow] Measuring {model} {data}"
    with mon.get_progress_bar(transient=not verbose) as pbar:
        for image_file in pbar.track(
            sequence    = image_files,
            total       = len(image_files),
            description = description
        ):
            # Image
            image  = mon.read_image(path=image_file, to_tensor=True, normalize=True).to(device=device)
            h0, w0 = mon.get_image_size(image)
            if torch.any(image.isnan()):
                continue
            if resize:  # Force resize
                image = mon.resize(image, (h, w))
            
            # Target
            has_target  = need_target
            target_file = None
            for ext in mon.IMAGE_FILE_FORMATS:
                temp = target_dir / f"{image_file.stem}{ext}"
                if temp.exists():
                    target_file = temp
            if target_file and target_file.exists():  # Has target file
                target = mon.read_image(path=target_file, to_tensor=True, normalize=True).to(device=device)
                h1, w1 = mon.get_image_size(target)
                if resize:  # Force resize
                    target = mon.resize(target, (h, w))
                elif h1 != h0 or w1 != w0:  # Mismatch size between image and target
                    image  = mon.resize(image, (h1, w1))
            else:
                has_target = False
            
            # GT mean
            if use_gt_mean and has_target:
                image = mon.scale_gt_mean(image, target)
            
            # Measure metric
            for m in metric:
                if m not in _METRICS:
                    continue
                if not has_target and _METRICS[m]["metric_mode"] == "FR":
                    continue
                elif has_target and _METRICS[m]["metric_mode"] == "FR":
                    values[m].append(metric_f[m](image, target))
                else:
                    values[m].append(metric_f[m](image))

    for m, v in values.items():
        if len(v) > 0:
            results[m] = float(sum(v) / num_items)
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

# endregion


# region Main

@click.command(name="metric", context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.option("--input-dir",   type=click.Path(exists=True),  default=None, help="Image directory.")
@click.option("--target-dir",  type=click.Path(exists=False), default=None, help="Ground-truth directory.")
@click.option("--result-file", type=str,                      default=None, help="Result file.")
@click.option("--arch",        type=str,                      default=None, help="Model's architecture.")
@click.option("--model",       type=str,                      default=None, help="Model's fullname.")
@click.option("--data",        type=str,                      default=None, help="Source data.")
@click.option("--device",      type=str,                      default=None, help="Running devices.")
@click.option("--imgsz",       type=int,                      default=512)
@click.option("--resize",      is_flag=True)
@click.option("--metric",      type=str, multiple=True, help="Measuring metric.")
@click.option("--use-gt-mean", is_flag=True)
@click.option("--backend",     type=click.Choice(["pyiqa"], case_sensitive=False), default=["pyiqa"], multiple=True)
@click.option("--save-txt",    is_flag=True)
@click.option("--verbose",     is_flag=True)
def main(
    input_dir  : mon.Path,
    target_dir : mon.Path,
    result_file: mon.Path | str,
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
    input_dir       = mon.Path(input_dir)
    target_dir      = mon.Path(target_dir)
    results         = {}
    results_gt_mean = {}
    
    if not verbose:
        logger = logging.getLogger()
        logger.disabled = True
    console.rule(f"[bold red] {model}")
    
    for b in backend:
        if b in ["pyiqa"]:
            metric_values = measure_metric_pyiqa(
                input_dir   = input_dir,
                target_dir  = target_dir,
                result_file = result_file,
                arch        = arch,
                model       = model,
                data        = data,
                device      = device,
                imgsz       = imgsz,
                resize      = resize,
                metric      = metric,
                use_gt_mean = False,
                save_txt    = save_txt,
                verbose     = verbose,
            )
            results = update_best_results(results, metric_values)
            if use_gt_mean:
                metric_values_gt_mean = measure_metric_pyiqa(
                    input_dir   = input_dir,
                    target_dir  = target_dir,
                    result_file = result_file,
                    arch        = arch,
                    model       = model,
                    data        = data,
                    device      = device,
                    imgsz       = imgsz,
                    resize      = resize,
                    metric      = metric,
                    use_gt_mean = True,
                    save_txt    = save_txt,
                    verbose     = verbose,
                )
                results_gt_mean = update_best_results(results_gt_mean, metric_values_gt_mean)
        else:
            console.log(f"`{backend}` is not supported!")
    
    # Show results
    # console.rule(f"[bold red] {model}")
    console.log(f"[bold green]Model: {model}")
    console.log(f"[bold red]Data : {input_dir.name}")
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
    main()

# endregion

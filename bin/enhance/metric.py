#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Measure metrics for image enhancement methods."""

from __future__ import annotations

import logging
import os

import click
import piqa
import pyiqa
import torch

import mon
from mon import METRIC_CONFIG

console = mon.console


# region Function

def measure_metric_piqa(
    image_dir     : mon.Path,
    target_dir    : mon.Path | None,
    result_file   : mon.Path | str,
    name          : str,
    variant       : int | str | None,
    image_size    : int,
    resize        : bool,
    metric        : list[str],
    test_y_channel: bool,
    save_txt      : bool,
    append_results: bool,
    show_results  : bool,
    verbose       : bool,
) -> dict:
    """Measure metrics."""
    _METRICS = {
        # "fid"    : piqa.FID,
        "fsim"   : piqa.FSIM,   
        "haarpsi": piqa.HaarPSI,
        "lpips"  : piqa.LPIPS,  
        "mdsi"   : piqa.MDSI,   
        "ms-gmsd": piqa.MS_GMSD,
        "ms-ssim": piqa.MS_SSIM,
        "psnr"   : piqa.PSNR,   
        "ssim"   : piqa.SSIM,   
        "tv"     : piqa.TV,     
        "vsi"    : piqa.VSI,    
    }

    assert image_dir is not None and mon.Path(image_dir).is_dir()
    # if target_dir is not None:
    #     assert mon.Path(target_dir).is_dir()
    if result_file is not None:
        assert (mon.Path(result_file).is_dir()
                or mon.Path(result_file).is_file()
                or isinstance(result_file, str))
        result_file = mon.Path(result_file)
        
    image_dir   = mon.Path(image_dir)
    target_dir  = mon.Path(target_dir) \
        if target_dir is not None \
        else mon.Path(str(image_dir).replace("low", "high"))
    
    result_file = mon.Path(result_file) if result_file is not None else None
    if save_txt and result_file is not None and result_file.is_dir():
        result_file /= "metric.txt"
        result_file.parent.mkdir(parents=True, exist_ok=True)
    
    image_files  = list(image_dir.rglob("*"))
    image_files  = [f for f in image_files if f.is_image_file()]
    image_files  = sorted(image_files)
    num_items    = len(image_files)
    
    device       = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    metric       = _METRICS if ("all" in metric or "*" in metric) else metric
    metric       = [m.lower() for m in metric]
    values       = {m: []     for m in metric}
    results      = {}
    metric_f     = {}
    for i, m in enumerate(metric):
        if m in _METRICS:
            metric_f[m] = _METRICS[m]().to(device=device)
        
    need_target = any(m in _METRICS and METRIC_CONFIG[m]["metric_mode"] == "FR" for m in metric)
   
    # Measuring
    h, w = mon.get_hw(image_size)
    with mon.get_progress_bar(transient=not verbose) as pbar:
        for image_file in pbar.track(
            sequence    = image_files,
            total       = len(image_files),
            description = f"[bright_yellow] Measuring"
        ):
            image = mon.read_image(path=image_file, to_rgb=True, to_tensor=True, normalize=True).to(device=device)
            if torch.any(image.isnan()):
                continue
            if resize:
                image = mon.resize(input=image, size=[h, w])
            
            has_target  = need_target
            target_file = None
            for ext in mon.ImageFormat.values():
                temp = target_dir / f"{image_file.stem}{ext}"
                if temp.exists():
                    target_file = temp
            if target_file is not None and target_file.exists():
                target = mon.read_image(path=target_file, to_rgb=True, to_tensor=True, normalize=True).to(device=device)
                if resize:
                    target = mon.resize(input=target, size=[h, w])
            else:
                has_target = False
            
            for m in metric:
                if m not in _METRICS:
                    continue
                if not has_target and METRIC_CONFIG[m]["metric_mode"] == "FR":
                    continue
                elif has_target and METRIC_CONFIG[m]["metric_mode"] == "FR":
                    values[m].append(float(metric_f[m](image, target)))
                else:
                    values[m].append(float(metric_f[m](image)))

    for m, v in values.items():
        if len(v) > 0:
            results[m] = float(sum(v) / num_items)
        else:
            results[m] = None
    return results


def measure_metric_pyiqa(
    image_dir     : mon.Path,
    target_dir    : mon.Path | None,
    result_file   : mon.Path | str,
    name          : str,
    variant       : int | str | None,
    image_size    : int,
    resize        : bool,
    metric        : list[str],
    test_y_channel: bool,
    save_txt      : bool,
    append_results: bool,
    show_results  : bool,
    verbose       : bool,
) -> dict:
    """Measure metrics using :mod:`pyiqa` package."""
    _METRICS = list(pyiqa.DEFAULT_CONFIGS.keys())

    assert image_dir is not None and mon.Path(image_dir).is_dir()
    # if target_dir is not None:
    #     assert mon.Path(target_dir).is_dir()
    if result_file is not None:
        assert (mon.Path(result_file).is_dir()
                or mon.Path(result_file).is_file()
                or isinstance(result_file, str))
        result_file = mon.Path(result_file)
        
    image_dir  = mon.Path(image_dir)
    target_dir = mon.Path(target_dir) \
        if target_dir is not None \
        else mon.Path(str(image_dir).replace("low", "high"))
    
    result_file = mon.Path(result_file) if result_file is not None else None
    if save_txt and result_file is not None and result_file.is_dir():
        result_file /= "metric.txt"
        result_file.parent.mkdir(parents=True, exist_ok=True)
    
    image_files = list(image_dir.rglob("*"))
    image_files = [f for f in image_files if f.is_image_file()]
    image_files = sorted(image_files)
    num_items   = len(image_files)
    
    device      = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    metric      = _METRICS if ("all" in metric or "*" in metric) else metric
    metric      = [m.lower() for m in metric]
    values      = {m: []     for m in metric}
    results     = {}
    metric_f    = {}

    mon.disable_print()
    for i, m in enumerate(metric):
        if m not in _METRICS:
            continue
        if "test_y_channel" in pyiqa.DEFAULT_CONFIGS[m]["metric_opts"]:
            metric_f[m] = pyiqa.InferenceModel(
                metric_name    = m,
                as_loss        = False,
                # test_y_channel = test_y_channel,
                device         = device,
            )
        else:
            metric_f[m] = pyiqa.InferenceModel(
                metric_name = m,
                as_loss     = False,
                device      = device,
            )
    mon.enable_print()
    need_target = any(METRIC_CONFIG[m]["metric_mode"] == "FR" for m in metric)
    
    # Measuring
    h, w = mon.get_hw(image_size)
    with mon.get_progress_bar(transient=not verbose) as pbar:
        for image_file in pbar.track(
            sequence    = image_files,
            total       = len(image_files),
            description = f"[bright_yellow] Measuring"
        ):
            image = mon.read_image(path=image_file, to_rgb=True, to_tensor=True, normalize=True).to(device=device)
            if torch.any(image.isnan()):
                continue
            if resize:
                image = mon.resize(input=image, size=[h, w])
            
            has_target  = need_target
            target_file = None
            for ext in mon.ImageFormat.values():
                temp = target_dir / f"{image_file.stem}{ext}"
                if temp.exists():
                    target_file = temp
            if target_file is not None and target_file.exists():
                target = mon.read_image(path=target_file, to_rgb=True, to_tensor=True, normalize=True).to(device=device)
                if resize:
                    target = mon.resize(input=target, size=[h, w])
            else:
                has_target = False
            
            for m in metric:
                if m not in _METRICS:
                    continue
                if not has_target and METRIC_CONFIG[m]["metric_mode"] == "FR":
                    continue
                elif has_target and METRIC_CONFIG[m]["metric_mode"] == "FR":
                    values[m].append(metric_f[m](image, target))
                else:
                    values[m].append(metric_f[m](image))

    for m, v in values.items():
        if len(v) > 0:
            results[m] = float(sum(v) / num_items)
        else:
            results[m] = None
    return results


def update_results(results: dict, new_values: dict) -> dict:
    for m, v in new_values.items():
        if m in METRIC_CONFIG:
            lower_better = METRIC_CONFIG[m]["lower_better"]
            if m not in results:
                results[m] = v
            elif results[m] is None:
                results[m] = v
            elif v is not None:
                results[m] = min(results[m], v) if lower_better else max(results[m], v)

    return results


@click.command()
@click.option("--image-dir",      default=mon.DATA_DIR/"", type=click.Path(exists=True),  help="Image directory.")
@click.option("--target-dir",     default=None,            type=click.Path(exists=False), help="Ground-truth directory.")
@click.option("--result-file",    default=None,            type=str, help="Result file.")
@click.option("--name",           default=None,            type=str, help="Model name.")
@click.option("--variant",        default=None,            type=str, help="Model variant.")
@click.option("--image-size",     default=512, type=int)
@click.option("--resize",         is_flag=True)
@click.option("--metric",         multiple=True, type=str, help="Measuring metric.")
@click.option("--test-y-channel", is_flag=True)
@click.option("--backend",        multiple=True, default=["piqa", "pyiqa"], type=click.Choice(["piqa", "pyiqa"], case_sensitive=False))
@click.option("--save-txt",       is_flag=True)
@click.option("--append-results", is_flag=True)
@click.option("--show-results",   is_flag=True)
@click.option("--verbose",        is_flag=True)
def measure_metric(
    image_dir     : mon.Path,
    target_dir    : mon.Path | None,
    result_file   : mon.Path | str,
    name          : str,
    variant       : int | str | None,
    image_size    : int,
    resize        : bool,
    metric        : list[str],
    test_y_channel: bool,
    backend       : list[str],
    save_txt      : bool,
    append_results: bool,
    show_results  : bool,
    verbose       : bool,
):
    results = {}

    if not verbose:
        logger = logging.getLogger()
        logger.disabled = True

    for b in backend:
        if b in ["piqa"]:
            new_values = measure_metric_piqa(
                image_dir      = image_dir,
                target_dir     = target_dir,
                result_file    = result_file,
                name           = name,
                variant        = variant,
                image_size     = image_size,
                resize         = resize,
                metric         = metric,
                test_y_channel = test_y_channel,
                save_txt       = save_txt,
                append_results = append_results,
                show_results   = show_results,
                verbose        = verbose,
            )
            results = update_results(results, new_values)
        elif b in ["pyiqa"]:
            new_values = measure_metric_pyiqa(
                image_dir      = image_dir,
                target_dir     = target_dir,
                result_file    = result_file,
                name           = name,
                variant        = variant,
                image_size     = image_size,
                resize         = resize,
                metric         = metric,
                test_y_channel = test_y_channel,
                save_txt       = save_txt,
                append_results = append_results,
                show_results   = show_results,
                verbose        = verbose,
            )
            results = update_results(results, new_values)
        else:
            console.log(f"`{backend}` is not supported!")

    # Show results
    variant       = variant if variant not in [None, "", "none"] else None
    model_variant = f"{name}-{variant}" if variant is not None else f"{name}"
    if verbose:
        console.rule(f"[bold red] {model_variant}")
        console.log(f"{model_variant}")
        console.log(f"{image_dir.name}")
        for m, v in results.items():
            console.log(f"{m:<10}: {v:.10f}")
    if show_results:
        message = ""
        # Headers
        if not append_results:
            for m, v in results.items():
                message += f"{f'{m}':<10}\t"
            message += "\n"
        # Values
        for i, (m, v) in enumerate(results.items()):
            if i == len(results) - 1:
                message += f"{0:.10f}"   if v is None else f"{v:.10f}"
            else:
                message += f"{0:.10f}\t" if v is None else f"{v:.10f}\t"

        if verbose:
            console.log(f"{message}")
        if not append_results:
            print(f"COPY THIS:")
        print(message)
    
    # Save results
    if save_txt:
        if not append_results:
            mon.delete_files(regex=result_file.name, path=result_file.parent)
        with open(str(result_file), "a") as f:
            if os.stat(str(result_file)).st_size == 0:
                f.write(f"{'model':<10}\t{'data':<10}\t")
                
                for m, v in results.items():
                    f.write(f"{f'{m}':<10}\t")
            f.write(f"{f'{model_variant}':<10}\t{f'{image_dir.name}':<10}\t")
            for m, v in results.items():
                f.write(f"{v:.10f}\t")
            f.write(f"\n")

# endregion


# region Main

if __name__ == "__main__":
    measure_metric()

# endregion

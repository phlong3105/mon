#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Measure metrics for image enhancement methods."""

from __future__ import annotations

import os

import click
import piqa
import pyiqa
import torch

import mon

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
    verbose       : bool,
):
    """Measure metrics."""
    _METRICS = {
        # "fid"    : {"module": piqa.FID,     "metric_mode": "FR", },
        "fsim"   : {"module": piqa.FSIM,    "metric_mode": "FR", },
        "haarpsi": {"module": piqa.HaarPSI, "metric_mode": "FR", },
        "lpips"  : {"module": piqa.LPIPS,   "metric_mode": "FR", },
        "mdsi"   : {"module": piqa.MDSI,    "metric_mode": "FR", },
        "ms-gmsd": {"module": piqa.MS_GMSD, "metric_mode": "FR", },
        "ms-ssim": {"module": piqa.MS_SSIM, "metric_mode": "FR", },
        "psnr"   : {"module": piqa.PSNR,    "metric_mode": "FR", },
        "ssim"   : {"module": piqa.SSIM,    "metric_mode": "FR", },
        "tv"     : {"module": piqa.TV,      "metric_mode": "NR", },
        "vsi"    : {"module": piqa.VSI,     "metric_mode": "FR", },
    }

    variant       = variant if variant not in [None, "", "none"] else None
    model_variant = f"{name}-{variant}" if variant is not None else f"{name}"
    console.rule(f"[bold red] {model_variant}")
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
    metric_f     = {}
    for i, m in enumerate(metric):
        if m in _METRICS:
            metric_f[m] = _METRICS[m]["module"]().to(device=device)
        
    need_target = any(m in _METRICS and _METRICS[m]["metric_mode"] == "FR" for m in metric)
   
    # Measuring
    h, w = mon.get_hw(image_size)
    with mon.get_progress_bar() as pbar:
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
                if not has_target and _METRICS[m]["metric_mode"] == "FR":
                    continue
                elif has_target and _METRICS[m]["metric_mode"] == "FR":
                    values[m].append(float(metric_f[m](image, target)))
                else:
                    values[m].append(float(metric_f[m](image)))
    
    # Show results
    if append_results:
        console.log(f"{model_variant}")
        console.log(f"{image_dir.name}")
        console.log(f"backend: pyiqa")
        message = ""
        for m, v in values.items():
            message += f"{f'{m}':<10}\t"
        message += "\n"
        for m, v in values.items():
            avg = float(sum(v) / num_items)
            message += f"{avg:.10f}\t"
        console.log(f"{message}")
        print(f"COPY THIS:")
        print(message)
    else:
        console.log(f"{model_variant}")
        console.log(f"{image_dir.name}")
        console.log(f"backend: piqa")
        for m, v in values.items():
            avg = float(sum(v) / num_items)
            console.log(f"{m:<10}: {avg:.10f}")
    
    # Save results
    if save_txt:
        if not append_results:
            mon.delete_files(regex=result_file.name, path=result_file.parent)
        with open(str(result_file), "a") as f:
            if os.stat(str(result_file)).st_size == 0:
                f.write(f"{'model':<10}\t{'data':<10}\t")
                for m, v in values.items():
                    f.write(f"{f'{m}':<10}\t")
            f.write(f"{f'{model_variant}':<10}\t{f'{image_dir.name}':<10}\t")
            for m, v in values.items():
                avg = float(sum(v) / num_items)
                f.write(f"{avg:.10f}\t")
            f.write(f"\n")
            # f.write(f"{model_name}\n")
            # f.write(f"{image_dir.name}\n")
            # for m, v in values.items():
            #     avg = float(sum(v) / num_items)
            #     f.write(f"{m:<10}: {avg:.9f}\n")


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
    verbose       : bool,
):
    """Measure metrics using :mod:`pyiqa` package."""
    _FULL_REFERENCE_METRICS = [
        "ahiq",
        "ckdn",
        "cw_ssim",
        "dists",
        "fsim",
        "gmsd",
        "lpips",
        "lpips-vgg",
        "mad",
        "ms_ssim",
        "nlpd",
        "pieapp",
        "psnr",
        "psnry",
        "ssim",
        "ssimc",
        "vif",
        "vsi",
        "wadiqam",
    ]
    _NON_REFERENCE_METRICS  = [
        "brisque",
        "clipiqa",
        "clipiqa+",
        "clipiqa+_rn50_512",
        "clipiqa+_vitL14_512",
        "cnniqa",
        "dbcnn",
        "fid",
        "hyperiqa",
        "ilniqe",
        "ilniqe",
        "maniqa",
        "maniqa-kadid",
        "maniqa-koniq",
        "musiq",
        "musiq-ava",
        "musiq-koniq",
        "musiq-paq2piq",
        "musiq-spaq",
        "nima",
        "nima-vgg16-ava",
        "niqe",
        "nrqm",
        "paq2piq",
        "pi",
        "pieapp",
        "tres",
        "tres-flive",
        "tres-koniq",
        "uranker",
    ]
    _METRICS = _NON_REFERENCE_METRICS + _FULL_REFERENCE_METRICS

    variant       = variant if variant not in [None, "", "none"] else None
    model_variant = f"{name}-{variant}" if variant is not None else f"{name}"
    console.rule(f"[bold red] {model_variant}")
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
    
    image_files = list(image_dir.rglob("*"))
    image_files = [f for f in image_files if f.is_image_file()]
    image_files = sorted(image_files)
    num_items   = len(image_files)
    
    device      = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    metric      = _METRICS if ("all" in metric or "*" in metric) else metric
    metric      = [m.lower() for m in metric]
    values      = {m: []     for m in metric}
    metric_f    = {}
    for i, m in enumerate(metric):
        if m not in _METRICS:
            continue
        metric_config = pyiqa.DEFAULT_CONFIGS[m]
        if "test_y_channel" in metric_config["metric_opts"]:
            metric_f[m] = pyiqa.create_metric(
                metric_name    = m,
                as_loss        = False,
                # test_y_channel = test_y_channel,
                device         = device,
            )
        else:
            metric_f[m] = pyiqa.create_metric(
                metric_name = m,
                as_loss     = False,
                device      = device,
            )
        
    need_target = any(m in _FULL_REFERENCE_METRICS for m in metric)
    
    # Measuring
    h, w = mon.get_hw(image_size)
    with mon.get_progress_bar() as pbar:
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
                metric_config = pyiqa.DEFAULT_CONFIGS[m]
                if not has_target and metric_config["metric_mode"] == "FR":
                    continue
                elif has_target and metric_config["metric_mode"] == "FR":
                    values[m].append(metric_f[m](image, target))
                else:
                    values[m].append(metric_f[m](image))
    
    # Show results
    if append_results:
        console.log(f"{model_variant}")
        console.log(f"{image_dir.name}")
        console.log(f"backend: pyiqa")
        message = ""
        for m, v in values.items():
            message += f"{f'{m}':<10} \t"
        message += "\n"
        for m, v in values.items():
            avg      = float(sum(v) / num_items)
            message += f"{avg:.10f} \t"
        console.log(f"{message}")
        print("COPY THIS:")
        print(message)
    else:
        console.log(f"{model_variant}")
        console.log(f"{image_dir.name}")
        console.log(f"backend: pyiqa")
        for m, v in values.items():
            avg = float(sum(v) / num_items)
            console.log(f"{m:<10}: {avg:.10f}")
    
    # Save results
    if save_txt:
        if not append_results:
            mon.delete_files(regex=result_file.name, path=result_file.parent)
        with open(str(result_file), "a") as f:
            if os.stat(str(result_file)).st_size == 0:
                f.write(f"{'model':<10}\t{'data':<10}\t")
                for m, v in values.items():
                    f.write(f"{f'{m}':<10}\t")
            f.write(f"{f'{model_variant}':<10}\t{f'{image_dir.name}':<10}\t")
            for m, v in values.items():
                avg = float(sum(v) / num_items)
                f.write(f"{avg:.10f}\t")
            f.write(f"\n")
                

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
@click.option("--backend",        default="pyiqa", type=click.Choice(["piqa", "pyiqa"], case_sensitive=False))
@click.option("--save-txt",       is_flag=True)
@click.option("--append-results", is_flag=True)
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
    backend       : str,
    save_txt      : bool,
    append_results: bool,
    verbose       : bool,
):
    if backend in ["piqa"]:
        measure_metric_piqa(
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
            verbose        = verbose,
        )
    elif backend in ["pyiqa"]:
        measure_metric_pyiqa(
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
            verbose        = verbose,
        )
    else:
        console.log(f"`{backend}` is not supported!")
    
# endregion


# region Main

if __name__ == "__main__":
    measure_metric()

# endregion

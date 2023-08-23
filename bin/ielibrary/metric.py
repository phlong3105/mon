#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Measure metrics for image enhancement methods."""

from __future__ import annotations

import click
import pyiqa

import mon

console = mon.console


# region Function
_FULL_REFERENCE_METRICS = ["lpips", "psnr", "ssim"]
_NON_REFERENCE_METRICS  = ["brisque", "nique", "unique"]
_METRICS                = _FULL_REFERENCE_METRICS + _NON_REFERENCE_METRICS


@click.command()
@click.option("--image-dir",   default=mon.DATA_DIR/"", type=click.Path(exists=True),  help="Image directory.")
@click.option("--target-dir",  default=mon.DATA_DIR/"", type=click.Path(exists=False), help="Ground-truth directory.")
@click.option("--result-file", default=None,            type=str, help="Result file.")
@click.option("--img-size", "-image_size", default=512, type=int)
@click.option(
    "--metric", "-m",
    multiple = True,
    default  = ["psnr", "ssim"],
    type     = click.Choice(["all", "*"] + _METRICS, case_sensitive = False),
    help     = "Measuring metric."
)
@click.option("--test-y-channel", is_flag=True)
@click.option("--color-space",    default="ycrcb", type=str)
@click.option("--verbose",        is_flag=True)
def measure_metric(
    image_dir     : mon.Path,
    target_dir    : mon.Path,
    result_file   : mon.Path | str,
    img_size      : int,
    metric        : list[str],
    test_y_channel: bool,
    color_space   : str,
    verbose       : bool,
):
    console.rule("[bold red]1. INITIALIZATION")
    assert image_dir is not None and mon.Path(image_dir).is_dir()
    if target_dir is not None:
        assert mon.Path(target_dir).is_dir()
    if result_file is not None:
        assert (mon.Path(result_file).is_dir()
                or mon.Path(result_file).is_file()
                or isinstance(result_file, str))
    
    image_dir   = mon.Path(image_dir)
    target_dir  = mon.Path(target_dir)  if target_dir  is not None else None
    
    result_file = mon.Path(result_file) if result_file is not None else None
    if result_file is not None and result_file.is_dir():
        result_file /= "metric.csv"
    result_file.mkdir(parents=True, exist_ok=True)
    
    image_files = list(image_dir.rglob("*"))
    image_files = [f for f in image_files if f.is_image_file()]
    image_files = sorted(image_files)
    num_items   = len(image_files)
    
    metric      = _METRICS if ("all" in metric or "*" in metric) else metric
    metric      = [m.lower() for m in metric]
    values      = {m: []     for m in metric}
    metric_f    = {}
    for i, m in enumerate(metric):
        metric_f[m] = pyiqa.create_metric(
            metric_name    = m,
            test_y_channel = test_y_channel,
            color_space    = color_space,
        )
        
    need_target = any(m in _FULL_REFERENCE_METRICS for m in metric)
    if need_target and target_dir is None:
        raise ValueError(f"Require target images.")
    console.log("[green]Done")
    
    console.rule("[bold red]2. MEASURING")
    with mon.get_progress_bar() as pbar:
        for image_file in pbar.track(
            sequence    = image_files,
            total       = len(image_files),
            description = f"[bright_yellow] Measuring"
        ):
            image = mon.read_image(path=image_file, to_rgb=True, to_tensor=True, normalize=True)
            image = mon.resize(input=image, size=img_size)
            if need_target:
                target_file = target_dir / image_file.name
                target      = mon.read_image(path=target_file, to_rgb=True, to_tensor=True, normalize=True)
                target      = mon.resize(input=target, size=img_size)
            for m in metric:
                if need_target:
                    values[m].append(metric_f[m](image, target))
                else:
                    values[m].append(metric_f[m](image))
    console.log("[green]Done")
    
    console.rule("[bold red]3. DISPLAY")
    for m, v in values.items():
        avg = float(sum(v) / num_items)
        console.log(f"{m:<10}: {avg:.9f}")
    console.log("[green]Done")
    
# endregion


# region Main

if __name__ == "__main__":
    measure_metric()

# endregion

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Measure metrics for image enhancement methods."""

from __future__ import annotations

import click

import mon
from mon.core.file import json


# region Function

@click.command()
@click.option("--image-dir",  default=mon.DATA_DIR/"", type=click.Path(exists=True),  help="Image directory.")
@click.option("--label-dir",  default=mon.DATA_DIR/"", type=click.Path(exists=False), help="Label directory.")
@click.option("--output-dir", default="",              type=click.Path(exists=False), help="Output directory.")
@click.option("--metric",     default="",              type=click.Path(exists=False), help="Output directory.")
@click.option("--verbose",    is_flag=True)
def measure_metric(
    image_dir : mon.Path,
    label_dir : mon.Path,
    output_dir: mon.Path,
    verbose   : bool
):
    assert image_dir is not None and mon.Path(image_dir).is_dir()
    if label_dir is not None:
        assert mon.Path(label_dir).is_dir()
    if output_dir is not None:
        assert mon.Path(output_dir).is_dir()
    
    image_dir  = mon.Path(image_dir)
    label_dir  = mon.Path(label_dir or image_dir.parent / f"labels")
    output_dir = mon.Path(output_dir or f".")
    label_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    image_files = list(image_dir.rglob("*"))
    image_files = [f for f in image_files if f.is_image_file()]
    image_files = sorted(image_files)
    
# endregion


# region Main

if __name__ == "__main__":
    measure_metric()

# endregion

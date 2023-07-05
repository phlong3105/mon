#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Fuse multiple prediction files togethers."""

from __future__ import annotations

import click

import mon
from mon.foundation.file import json


# region Function

@click.command()
@click.option("--input-dir",  default=mon.RUN_DIR/"predict/delftbikes/prediction/", type=click.Path(exists=True),  help="Prediction directory.")
@click.option("--output-dir", default=mon.RUN_DIR/"predict/delftbikes/submission/labels/", type=click.Path(exists=False), help="Submission directory.")
@click.option("--verbose",    is_flag=True)
def fuse_prediction(
    input_dir : mon.Path,
    output_dir: mon.Path,
    verbose   : bool
):
    assert input_dir  is not None and mon.Path(input_dir).is_dir()
   
    input_dir    = mon.Path(input_dir)
    output_dir   = mon.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    subdir_names = [d.stem for d in input_dir.subdirs()]
    input_files  = []
    with mon.get_progress_bar() as pbar:
        for subdir in pbar.track(
            sequence    = subdir_names,
            description = f"[bright_yellow] Listing prediction files"
        ):
            pred_dir   = input_dir/subdir/"labels"
            pred_files = list(pred_dir.rglob("*"))
            pred_files = [f for f in pred_files if f.is_txt_file()]
            pred_files = sorted(pred_files)
            input_files.append(pred_files)
    
    with mon.get_progress_bar() as pbar:
        for i in pbar.track(
            sequence    = range(len(input_files[0])),
            total       = len(input_files[0]),
            description = f"[bright_yellow] Fusing predictions"
        ):
            output_file_name = input_files[0][i].name
            output_file      = output_dir/output_file_name
            output_lines     = []
            for j in range(len(input_files)):
                assert output_file_name == input_files[j][i].name
                with open(input_files[j][i], "r") as input_f:
                    output_lines.extend(input_f.read().splitlines())
            
            with open(output_file, "w") as output_f:
                for line in output_lines:
                    output_f.write(line + "\n")
                
# endregion


# region Main

if __name__ == "__main__":
    fuse_prediction()

# endregion

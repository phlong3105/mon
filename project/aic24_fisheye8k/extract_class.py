#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import click

import mon


# region Function

@click.command(name="main", context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.option("--image-dir", type=click.Path(exists=True),  default=mon.DATA_DIR/"aic/aic24-fisheye8k/train", help="Image directory.")
@click.option("--label-dir", type=click.Path(exists=True),  default=mon.DATA_DIR/"aic/aic24-fisheye8k/train", help="Bounding bbox directory.")
@click.option("--classes",   type=str,                      default="0,2,4", help="Class ID to keep.")
@click.option("--verbose",   is_flag=True)
def extract_class(
    image_dir : mon.Path,
    label_dir : mon.Path,
    classes   : str,
    verbose   : bool,
):
    assert image_dir is not None and mon.Path(image_dir).is_dir()
    assert label_dir is not None and mon.Path(label_dir).is_dir()
    
    image_dir   = mon.Path(image_dir)
    label_dir   = mon.Path(label_dir)
    data_name   = image_dir.name
    classes     = classes.replace(" ", "")
    classes_str = classes
    classes     = [int(c) for c in classes.split(",")]
    print(classes)
    
    label_files = list(label_dir.rglob("*/labels/*.txt"))
    label_files = [f for f in label_files if f.is_txt_file()]
    label_files = sorted(label_files)
    with mon.get_progress_bar() as pbar:
        for i in pbar.track(
            sequence    = range(len(label_files)),
            total       = len(label_files),
            description = f"[bright_yellow] Processing {data_name}"
        ):
            label_file = label_files[i]
            with open(label_file, "r") as in_file:
                ls = in_file.read().splitlines()
                ls = [l.strip().split(" ") for l in ls]
                ls = [l for l in ls if len(l) >= 5]
                
            output_file = label_file.parent.parent / f"labels-{classes_str}" / label_file.name
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w") as out_file:
                for l in ls:
                    for j, c in enumerate(classes):
                        if l[0] == c:
                            l[0] = j
                            out_file.write(" ".join(l) + "\n")
              
# endregion


# region Main

if __name__ == "__main__":
    extract_class()

# endregion

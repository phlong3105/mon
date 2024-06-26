#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements visual comparison pipeline."""

from __future__ import annotations

import math

import click
import cv2
import numpy as np
from matplotlib import pyplot as plt

import mon

console = mon.console

plt.rcParams["savefig.bbox"] = "tight"


_INCLUDE_DATASETS = [
    "dicm",
    "fusion",
    "lime",
    "lol-v1",
    "lol-v2-real",
    "lol-v2-syn",
    "mef",
    "npe",
    "vv",
]
_EXCLUDE_DATASETS = [
    "darkface",
    "deepupe",
    "exdark",
    "sice",
    "sice-part2",
]
_INCLUDE_DIRS     = [
    "input",
    "ground-truth",
    "enlightengan",
    "zerodce",
    "zerodce++",
    #
    "zerodcev2-0101",
    "zerodcev2-0102",
    "zerodcev2-0111",
    "zerodcev2-0112",
    "zerodcev2-0201",
    "zerodcev2-0202",
    "zerodcev2-0211",
    "zerodcev2-0212",
    '''
    #
    "zerodcev2-0000",
    "zerodcev2-0001",
    "zerodcev2-0002",
    #
    "zerodcev2-0010",
    "zerodcev2-0011",
    "zerodcev2-0012",
    #
    "zerodcev2-0100",
    "zerodcev2-0101",
    "zerodcev2-0102",
    #
    "zerodcev2-0110",
    "zerodcev2-0111",
    "zerodcev2-0112",
    #
    "zerodcev2-0200",
    "zerodcev2-0201",
    "zerodcev2-0202",
    #
    "zerodcev2-0210",
    "zerodcev2-0211",
    "zerodcev2-0212",
    '''
]
_EXCLUDE_DIRS     = [
    "compare",
]


# region Function

def list_images(input_dir: mon.Path, verbose: bool):
    assert input_dir is not None and mon.Path(input_dir).is_dir()
    input_dir = mon.Path(input_dir)
    
    # List all sub-directories
    subdirs       = sorted(input_dir.subdirs())
    dataset_names = []
    image_grid    = {
        "input"       : "",
        "ground-truth": "",
    }
    with mon.get_progress_bar() as pbar:
        for sd in pbar.track(
            sequence    = subdirs,
            total       = len(subdirs),
            description = f"[bright_yellow] Listing datasets"
        ):
            if sd.name not in image_grid \
                and sd.name in _INCLUDE_DIRS \
                and sd.name not in _EXCLUDE_DIRS \
                and "compare" not in sd.name:
                image_grid[sd.name] = ""
            for dataset_dir in sd.subdirs():
                if dataset_dir.name not in dataset_names \
                    and dataset_dir.name in _INCLUDE_DATASETS \
                    and dataset_dir.name not in _EXCLUDE_DATASETS:
                    dataset_names.append(dataset_dir.name)
    dataset_names = sorted(dataset_names)
    if verbose:
        console.log(list(image_grid.keys()))
        console.log(dataset_names)
    
    # Listing image names
    image_stem_dict = {}
    with mon.get_progress_bar() as pbar:
        for dn in pbar.track(
            sequence    = dataset_names,
            total       = len(dataset_names),
            description = f"[bright_yellow] Listing images"
        ):
            image_paths         = input_dir.glob(f"*/{dn}/*")
            image_paths         = [p for p in image_paths if p.is_image_file() ]
            image_stem_dict[dn] = list(set([p.stem for p in image_paths]))
    
    return subdirs, dataset_names, image_grid, image_stem_dict


def plot_cv2(
    input_dir : mon.Path,
    output_dir: mon.Path | str,
    image_size: int | bool,
    num_cols  : int,
    verbose   : bool
):
    subdirs, dataset_names, image_grid, image_stem_dict = list_images(input_dir, verbose)
    
    if output_dir is not None:
        output_dir = mon.Path(output_dir)
        if output_dir.exists():
            mon.delete_dir(paths=output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Visualize images
    with mon.get_progress_bar() as pbar:
        for dn in pbar.track(
            sequence    = dataset_names,
            total       = len(dataset_names),
            description = f"[bright_yellow] Visualizing"
        ):
            for image_stem in image_stem_dict[dn]:
                image_shape = None
                image_dtype = None
                # Read images
                for k, _ in image_grid.items():
                    path = None
                    for ext in mon.IMAGE_FILE_FORMATS:
                        temp = input_dir / k / dn / f"{image_stem}{ext}"
                        if temp.exists():
                            path = temp
                    if path is not None and path.exists() and path.is_image_file():
                        image       = cv2.imread(str(path))[..., ::-1]
                        image_dtype = image.dtype
                        if k != "zerodce++":
                            image_shape = image.shape
                    else:
                        image = None
                    image_grid[k] = image
                    
                # Resize
                for k, v in image_grid.items():
                    if v is not None and image_size is not None:
                        h, w          = mon.get_hw(image_size)
                        image_grid[k] = cv2.resize(v, [w, h])
                        image_shape   = v.shape
                
                # Handle empty images
                for k, v in image_grid.items():
                    if v is None:
                        image_grid[k] = np.full(image_shape, 255, dtype=image_dtype)
                    elif k == "zerodce++" and image_size is None and image_shape is not None:
                        image_grid[k] = cv2.resize(v, [image_shape[1], image_shape[0]])
                
                # Add texts
                for k, v in image_grid.items():
                    top    = 50  # shape[0] = rows
                    bottom = top
                    left   = 10  # shape[1] = cols
                    right  = left
                    v      = cv2.copyMakeBorder(v, top, bottom, left, right, cv2.BORDER_CONSTANT, None, [255, 255, 255])
                    #
                    textsize = cv2.getTextSize(k, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                    if textsize is not None:
                        text_x = round((v.shape[1] - textsize[0]) / 2)
                        text_y = textsize[1] + 10  # round((v.shape[0] + textsize[1]) / 2)
                        cv2.putText(v, k, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                    #
                    image_grid[k] = v
                    image_shape   = v.shape
                
                # Display images
                num_rows          = math.ceil(len(image_grid) / num_cols)
                image_grid_values = list(image_grid.values())
                row_images        = []
                for i in range(num_rows):
                    row_image = None
                    for j in range(num_cols):
                        idx = i * num_cols + j
                        if j == 0:
                            row_image   = image_grid_values[idx]
                            continue
                        if idx < len(image_grid):
                            row_image   = cv2.hconcat([row_image, image_grid_values[idx]])
                        else:
                            empty_image = np.full(image_shape, 255, dtype=image_dtype)
                            row_image   = cv2.hconcat([row_image, empty_image])
                    row_images.append(row_image)
                output = cv2.vconcat(row_images)
                output = output[..., ::-1]
                #
                if output_dir is not None:
                    result_path = output_dir / dn / f"{image_stem}.png"
                    result_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(result_path), output)
                #
                if verbose:
                    cv2.imshow("Output", output)
                    cv2.waitKey(1)
                

def plot_cv2_diff(
    input_dir : mon.Path,
    output_dir: mon.Path | str,
    image_size: int | bool,
    num_cols  : int,
    mode      : str,
    ref       : str,
    verbose   : bool
):
    subdirs, dataset_names, image_grid, image_stem_dict = list_images(input_dir, verbose)
    
    if output_dir is not None:
        output_dir = mon.Path(output_dir)
        # if output_dir.exists():
        #     mon.delete_dir(paths=output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Visualize images
    with mon.get_progress_bar() as pbar:
        for dn in pbar.track(
            sequence    = dataset_names,
            total       = len(dataset_names),
            description = f"[bright_yellow] Visualizing"
        ):
            for image_stem in image_stem_dict[dn]:
                image_shape = None
                image_dtype = None
                # Read images
                for k, _ in image_grid.items():
                    path = None
                    for ext in mon.IMAGE_FILE_FORMATS:
                        temp = input_dir / k / dn / f"{image_stem}{ext}"
                        if temp.exists():
                            path = temp
                    if path is not None and path.exists() and path.is_image_file():
                        image       = cv2.imread(str(path))[..., ::-1]
                        image_dtype = image.dtype
                        if k != "zerodce++":
                            image_shape = image.shape
                    else:
                        image = None
                    image_grid[k] = image
                    
                # Resize
                for k, v in image_grid.items():
                    if v is not None and image_size is not None:
                        h, w          = mon.get_hw(image_size)
                        image_grid[k] = cv2.resize(v, [w, h])
                        image_shape   = v.shape
                
                # Handle empty images
                for k, v in image_grid.items():
                    if v is None:
                        image_grid[k] = np.full(image_shape, 255, dtype=image_dtype)
                    elif k == "zerodce++" and image_size is None and image_shape is not None:
                        image_grid[k] = cv2.resize(v, [image_shape[1], image_shape[0]])
                
                plot_diff = (mode == "diff" and image_grid[ref].shape != ())
                if plot_diff:
                    # Difference
                    ref_image = cv2.cvtColor(image_grid[ref], cv2.COLOR_RGB2GRAY)
                    for k, v in image_grid.items():
                        if k != ref:
                            v    = cv2.cvtColor(v, cv2.COLOR_RGB2GRAY)
                            diff = cv2.subtract(v, ref_image)
                            diff = (diff * 255).astype("uint8")
                            diff = cv2.merge([diff, diff, diff])
                            image_grid[k] = diff.astype("uint8")
                        
                    # Add texts
                    for k, v in image_grid.items():
                        top    = 50  # shape[0] = rows
                        bottom = top
                        left   = 10  # shape[1] = cols
                        right  = left
                        v      = cv2.copyMakeBorder(v, top, bottom, left, right, cv2.BORDER_CONSTANT, None, [255, 255, 255])
                        #
                        textsize = cv2.getTextSize(k, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
                        if textsize is not None:
                            text_x = round((v.shape[1] - textsize[0]) / 2)
                            text_y = textsize[1] + 10  # round((v.shape[0] + textsize[1]) / 2)
                            cv2.putText(v, k, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                        #
                        image_grid[k] = v
                        image_shape   = v.shape
                
                    # Display images
                    num_rows          = math.ceil(len(image_grid) / num_cols)
                    image_grid_values = list(image_grid.values())
                    row_images        = []
                    for i in range(num_rows):
                        row_image = None
                        for j in range(num_cols):
                            idx = i * num_cols + j
                            if j == 0:
                                row_image   = image_grid_values[idx]
                                continue
                            if idx < len(image_grid):
                                row_image   = cv2.hconcat([row_image, image_grid_values[idx]])
                            else:
                                empty_image = np.full(image_shape, 255, dtype=image_dtype)
                                row_image   = cv2.hconcat([row_image, empty_image])
                        row_images.append(row_image)
                    output = cv2.vconcat(row_images)
                    output = output[..., ::-1]
                    #
                    if output_dir is not None:
                        result_path = output_dir / dn / f"{image_stem}-diff.png"
                        result_path.parent.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(result_path), output)
                    #
                    if verbose:
                        cv2.imshow("Output", output)
                        cv2.waitKey(1)
                        

def plot_matplotlib(
    input_dir : mon.Path,
    output_dir: mon.Path | str,
    image_size: int | bool,
    num_cols  : int,
    verbose   : bool
):
    subdirs, dataset_names, image_grid, image_stem_dict = list_images(input_dir, verbose)
    
    if output_dir is not None:
        output_dir = mon.Path(output_dir)
        if output_dir.exists():
            mon.delete_dir(paths=output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Visualize images
    first_image = True
    with mon.get_progress_bar() as pbar:
        for dn in pbar.track(
            sequence    = dataset_names,
            total       = len(dataset_names),
            description = f"[bright_yellow] Visualizing"
        ):
            for image_stem in image_stem_dict[dn]:
                image_shape = None
                image_dtype = None
                for k, _ in image_grid.items():
                    path = None
                    for ext in mon.IMAGE_FILE_FORMATS:
                        temp = input_dir / k / dn / f"{image_stem}{ext}"
                        if temp.exists():
                            path = temp
                    if path is not None and path.exists() and path.is_image_file():
                        image = cv2.imread(str(path))[..., ::-1]
                        if image_size is not None:
                            h, w  = mon.get_hw(image_size)
                            image = cv2.resize(image, [w, h])
                        image_grid[k] = image
                        if image_shape is None:
                            image_shape = image.shape
                            image_dtype = image.dtype
                    else:
                        image_grid[k] = None
                
                num_rows = math.ceil(len(image_grid) / num_cols)
                h, w, c  = image_shape
                px       = 1 / plt.rcParams["figure.dpi"]  # pixel in inches
                fig_w    = num_cols * w * px
                fig_h    = (num_rows * h * px) if num_rows <= 1 else ((num_rows - 1) * h * px)
                fig, axs = plt.subplots(
                    nrows   = num_rows,
                    ncols   = num_cols,
                    layout  = "constrained",
                    # num     = 0,
                    figsize = (fig_w, fig_h),
                )
                # fig.tight_layout()
                # plt.tight_layout()
                image_grid_keys = list(image_grid.keys())
                for i, a in enumerate(axs.flatten()):
                    k = image_grid_keys[i] if i < len(image_grid_keys) else None
                    v = image_grid[k] if k is not None else None
                    if v is None:
                        v    = np.zeros(image_shape, dtype=np.uint8)
                        v[:] = 255
                        v    = v.astype(image_dtype)
                    a.imshow(v)
                    # a.axis("off")
                    # a.set_title(k)
                    a.xaxis.set_ticklabels([])
                    a.yaxis.set_ticklabels([])
                    a.set_xlabel(k or "", loc="center")
                # plt.subplots_adjust(wspace=0.0, hspace=0.0)
                
                if output_dir is not None:
                    result_path = output_dir / dn / f"{image_stem}.png"
                    result_path.parent.mkdir(parents=True, exist_ok=True)
                    plt.savefig(result_path, dpi=500)
                
                if verbose:
                    if first_image:
                        manager = plt.get_current_fig_manager()
                        manager.full_screen_toggle()
                        first_image = False
                    plt.show()
                    plt.pause(1)
                plt.close(fig)


@click.command(context_settings=dict(
    ignore_unknown_options = True,
    allow_extra_args       = True,
))
@click.option(
    "--input-dir",
    default = mon.RUN_DIR / "predict/vision/enhance/llie",
    type    = click.Path(exists=True),
    help    = "Image directory."
)
@click.option(
    "--output-dir",
    default = mon.RUN_DIR / "predict/vision/enhance/llie/compare",
    type    = click.Path(exists=False),
    help    = "Save results location."
)
@click.option("--image-size", default=None,    type=int)
@click.option("--num-cols",   default=8,       type=int)
@click.option("--mode",       default="diff",  type=click.Choice(["image", "diff"],     case_sensitive=False))
@click.option("--ref",        default="input", type=click.Choice(_INCLUDE_DIRS,                 case_sensitive=False))
@click.option("--backend",    default="cv2",   type=click.Choice(["cv2", "matplotlib"], case_sensitive=False))
@click.option("--verbose",    is_flag=True)
@click.pass_context
def main(
    ctx,
    input_dir : mon.Path,
    output_dir: mon.Path | str,
    image_size: int | bool,
    num_cols  : int,
    mode      : str,
    ref       : str,
    backend   : str,
    verbose   : bool
):
    model_kwargs = {
        k.lstrip("--"): ctx.args[i + 1]
            if not (i + 1 >= len(ctx.args) or ctx.args[i + 1].startswith("--"))
            else True for i, k in enumerate(ctx.args) if k.startswith("--")
    }
    
    if backend in ["cv2"]:
        plot_cv2(
            input_dir  = input_dir,
            output_dir = output_dir,
            image_size = image_size,
            num_cols   = num_cols,
            verbose    = verbose,
        )
        if mode == "diff":
            plot_cv2_diff(
                input_dir  = input_dir,
                output_dir = output_dir,
                image_size = image_size,
                num_cols   = num_cols,
                mode       = mode,
                ref        = ref,
                verbose    = verbose,
            )
    elif backend in ["matplotlib"]:
        plot_matplotlib(
            input_dir  = input_dir,
            output_dir = output_dir,
            image_size = image_size,
            num_cols   = num_cols,
            verbose    = verbose,
        )
    else:
        raise ValueError
                
# endregion


# region Main

if __name__ == "__main__":
    main()

# endregion

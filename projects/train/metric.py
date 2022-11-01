#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Measure performance script.
"""

from __future__ import annotations

import argparse
import socket

import pyiqa
from munch import Munch
from torch.nn import functional as F
from one.constants import DATA_DIR
from one.constants import PROJECTS_DIR
from one.core import console
from one.core import InterpolationMode
from one.core import is_image_file
from one.core import Paths_
from one.core import progress_bar
from one.nn import MAELoss
from one.vision.acquisition import read_image
from one.vision.transformation import resize


# H1: - Measure ----------------------------------------------------------------

def measure(args: Munch | dict):
    args = Munch.fromDict(args)

    # H2: - Initialization -----------------------------------------------------
    console.rule("[bold red]1. INITIALIZATION")
    console.log(f"Machine: {args.hostname}")
    pred_files  : Paths_ = args.pred_files
    target_files: Paths_ = args.target_files
   
    # Metrics
    # brisque = pyiqa.create_metric(metric_name="brisque")
    # ilniqe  = pyiqa.create_metric(metric_name="ilniqe")
    mae     = F.l1_loss
    niqe    = pyiqa.create_metric(metric_name="niqe")
    psnr    = pyiqa.create_metric(metric_name="psnr")
    ssim    = pyiqa.create_metric(metric_name="ssim")

    # Values
    num_items      = 0
    # brisque_values = []
    # ilniqe_values  = []
    mae_values     = []
    niqe_values    = []
    psnr_values    = []
    ssim_values    = []
    console.log("[green]Done")
    
    # H2: - Measure ------------------------------------------------------------
    console.rule("[bold red]2. MEASURING")
    with progress_bar() as pbar:
        for i, pred_file in pbar.track(
            enumerate(pred_files),
            total       = len(pred_files),
            description = f"[bright_yellow] Measuring"
        ):
            if not is_image_file(pred_file):
                continue
            
            pred   = read_image(str(pred_file))
            target = None
            if target_files is not None:
                target = read_image(str(target_files[i]))
                
            if args.shape is not None:
                pred = resize(
                    image         = pred,
                    size          = args.shape,
                    interpolation = InterpolationMode.BICUBIC,
                    antialias     = True,
                )
                if target is not None:
                    target = resize(
                        image         = target,
                        size          = args.shape,
                        interpolation = InterpolationMode.BICUBIC,
                        antialias     = True,
                    )

            niqe_values.append(niqe(pred))
            if target is not None:
                # brisque_values.append(brisque(pred))
                # ilniqe_values.append(ilniqe(pred))
                mae_values.append(mae(pred * 255, target * 255, reduction="mean"))
                psnr_values.append(psnr(pred, target))
                ssim_values.append(ssim(pred, target))
            num_items += 1
    console.log("[green]Done")
    
    # H2: - Display ------------------------------------------------------------
    console.rule("[bold red]3. DISPLAY")
    # avg_brisque = sum(brisque_values) / num_items
    # avg_ilniqe  = sum(ilniqe_values)  / num_items
    avg_mae     = sum(mae_values)     / num_items
    avg_niqe    = sum(niqe_values)    / num_items
    avg_psnr    = sum(psnr_values)    / num_items
    avg_ssim    = sum(ssim_values)    / num_items
    
    # avg_brisque = float(avg_brisque)
    # avg_ilniqe  = float(avg_ilniqe)
    avg_mae     = float(avg_mae)
    avg_niqe    = float(avg_niqe)
    avg_psnr    = float(avg_psnr)
    avg_ssim    = float(avg_ssim)
    
    # console.log(f"brisque: {avg_brisque:.9f}")
    # console.log(f"ilniqe : {avg_ilniqe :.9f}")
    console.log(f"mae    : {avg_mae    :.9f}")
    console.log(f"niqe   : {avg_niqe   :.9f}")
    console.log(f"psnr   : {avg_psnr   :.9f}")
    console.log(f"ssim   : {avg_ssim   :.9f}")
    console.log("[green]Done")


# H1: - Main -------------------------------------------------------------------

hosts = {
    "lp-labdesktop01-ubuntu": {
        # "pred"    : PROJECTS_DIR / "train" / "runs" / "infer" / "lol226" / "zeroadce-e-tiny-lol226-0" / "dcim",
        "pred"    : PROJECTS_DIR / "train" / "runs" / "infer" / "sice" / "zeroadce-a-lol4k-0",
        # "target"  : None,
        "target"  : True,
        # "img_size": None,
        "img_size": (3, 400, 600),
        "save"    : True,
        "verbose" : False,
	},
    "lp-imac.local": {
        # "pred"    : PROJECTS_DIR / "train" / "runs" / "infer" / "lol226" / "zeroadce-c-tiny-lol226-0",
        "pred"    : PROJECTS_DIR / "train" / "runs" / "infer" / "sice" / "zeroadce-e-large-lol226-0",
        "target"  : True,
        # "target"  : None,
        # "img_size": None,
        "img_size": (3, 400, 600),
        "save"    : True,
        "verbose" : False,
	},
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred",     type=str,            help="Prediction source.")
    parser.add_argument("--target",   type=str,            help="Ground-truth source.")
    parser.add_argument("--img-size", type=int, nargs="+", help="Image sizes.")
    parser.add_argument("--save",     action="store_true", help="Save.")
    parser.add_argument("--verbose",  action="store_true", help="Display results.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    hostname     = socket.gethostname().lower()
    host_args    = Munch(hosts[hostname])
    
    input_args   = vars(parse_args())
    pred         = input_args.get("pred",     None) or host_args.get("pred",     None)
    target       = input_args.get("target",   None) or host_args.get("target",   None)
    shape        = input_args.get("img_size", None) or host_args.get("img_size", None)
    save         = input_args.get("save",     None) or host_args.get("save",     None)
    verbose      = input_args.get("verbose",  None) or host_args.get("verbose",  None)

    pred_files   = list(pred.rglob("*"))
    pred_files   = [f for f in pred_files if is_image_file(f)]
    
    target_files = None
    if target is not None:
        target_files = []
        for i, f in enumerate(pred_files):
            stem = str(f.parent.stem)
            file = DATA_DIR / "sice" / "part2_900x1200_low" / "high" / f"{stem}.jpg"
            target_files.append(file)

    args = Munch(
        hostname     = hostname,
        pred_files   = pred_files,
        target_files = target_files,
        shape        = shape,
        save         = save,
        verbose      = verbose,
    )
    measure(args=args)
